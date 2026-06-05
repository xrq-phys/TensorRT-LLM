# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
cuDNN MXFP8 SDPA backend for visual generation models.

Requires ``nvidia-cudnn-frontend`` with cuDNN MXFP8 SDPA support and Blackwell GPUs
(sm100/sm103).

Q/K: block-scaled MXFP8 via ``torch.ops.trtllm.mxfp8_quantize`` - same approach
as the CuTe DSL backend. V: block-scaled MXFP8 over the sequence dimension. Layout:
HND [B, H, S, D].
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import PredefinedAttentionMask
from ._mxfp8_quant import _MXFP8_BLOCK, quantize_mxfp8_qk, quantize_mxfp8_v
from .interface import AttentionBackend, AttentionTensorLayout

_cudnn_import_error: Optional[BaseException] = None
try:
    import cudnn  # type: ignore[import]
except (ImportError, OSError) as _e:
    cudnn = None
    _cudnn_import_error = _e

_cudnn_handle: Optional[Any] = None


def _check_cudnn_available() -> None:
    if _cudnn_import_error is None:
        return
    raise ImportError(
        "cuDNN Python frontend is not available. "
        "Install it with: pip install nvidia-cudnn-frontend\n"
        f"Import error: {_cudnn_import_error}"
    ) from _cudnn_import_error


def _get_cudnn_handle() -> Any:
    global _cudnn_handle
    if _cudnn_handle is None:
        _cudnn_handle = cudnn.create_handle()
    return _cudnn_handle


def _torch_to_cudnn_dtype(dt: torch.dtype) -> Any:
    _check_cudnn_available()
    _map = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
    }
    if dt not in _map:
        raise ValueError(f"No cudnn.data_type mapping for {dt}")
    return _map[dt]


@dataclass
class _GraphBundle:
    graph: Any
    workspace_size: int
    q_t: Any
    k_t: Any
    v_t: Any
    sf_q_t: Any
    sf_k_t: Any
    sf_v_t: Any
    o_t: Any
    stats_t: Any
    amax_o_t: Any


_GRAPH_CACHE: Dict[Tuple, _GraphBundle] = {}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_up(x: int, multiple: int) -> int:
    return _ceil_div(x, multiple) * multiple


def _stride(*dims: int) -> list[int]:
    acc = 1
    strides = []
    for dim in reversed(dims):
        strides.append(acc)
        acc *= dim
    return list(reversed(strides))


def _build_graph(
    b: int,
    h: int,
    hkv: int,
    sq: int,
    skv: int,
    d: int,
    is_causal: bool,
    sm_scale: float,
    out_dtype: torch.dtype,
    device: torch.device,
    *,
    with_lse: bool,
) -> _GraphBundle:
    out_cudnn_dt = _torch_to_cudnn_dtype(out_dtype)
    fp8 = cudnn.data_type.FP8_E4M3
    e8m0 = cudnn.data_type.FP8_E8M0
    s_q_pad = _pad_up(sq, 128)
    s_kv_pad = _pad_up(skv, 128)
    qk_scale_cols = _pad_up(_ceil_div(d, _MXFP8_BLOCK), 4)
    v_scale_rows = _pad_up(_ceil_div(skv, _MXFP8_BLOCK), 4)
    v_d_pad = _pad_up(d, 128)

    graph = cudnn.pygraph(
        io_data_type=fp8,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        name="mxfp8_sdpa",
    )

    q_t = graph.tensor(name="q", dim=[b, h, sq, d], stride=_stride(b, h, sq, d), data_type=fp8)
    k_t = graph.tensor(
        name="k", dim=[b, hkv, skv, d], stride=_stride(b, hkv, skv, d), data_type=fp8
    )
    v_t = graph.tensor(
        name="v", dim=[b, hkv, skv, d], stride=_stride(b, hkv, skv, d), data_type=fp8
    )

    sf_q_t = graph.tensor(
        name="sf_q",
        dim=[b, h, s_q_pad, qk_scale_cols],
        stride=_stride(b, h, s_q_pad, qk_scale_cols),
        data_type=e8m0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_k_t = graph.tensor(
        name="sf_k",
        dim=[b, hkv, s_kv_pad, qk_scale_cols],
        stride=_stride(b, hkv, s_kv_pad, qk_scale_cols),
        data_type=e8m0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )
    sf_v_t = graph.tensor(
        name="sf_v",
        dim=[b, hkv, v_scale_rows, v_d_pad],
        stride=[
            hkv * v_scale_rows * v_d_pad,
            v_scale_rows * v_d_pad,
            1,
            v_scale_rows,
        ],
        data_type=e8m0,
        reordering_type=cudnn.tensor_reordering.F8_128x4,
    )

    o_t, stats_t, amax_o_t = graph.sdpa_mxfp8(
        q=q_t,
        k=k_t,
        v=v_t,
        descale_q=sf_q_t,
        descale_k=sf_k_t,
        descale_v=sf_v_t,
        attn_scale=sm_scale,
        use_causal_mask=is_causal,
        generate_stats=True,
    )

    o_t.set_output(True).set_dim([b, h, sq, d]).set_stride(_stride(b, h, sq, d)).set_data_type(
        out_cudnn_dt
    )
    stats_t.set_output(True).set_dim([b, h, sq, 1]).set_stride(_stride(b, h, sq, 1)).set_data_type(
        cudnn.data_type.FLOAT
    )
    amax_o_t.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(
        cudnn.data_type.FLOAT
    )

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    return _GraphBundle(
        graph=graph,
        workspace_size=graph.get_workspace_size(),
        q_t=q_t,
        k_t=k_t,
        v_t=v_t,
        sf_q_t=sf_q_t,
        sf_k_t=sf_k_t,
        sf_v_t=sf_v_t,
        o_t=o_t,
        stats_t=stats_t,
        amax_o_t=amax_o_t,
    )


@torch.compiler.disable
def _get_or_build_graph(
    b: int,
    h: int,
    hkv: int,
    sq: int,
    skv: int,
    d: int,
    is_causal: bool,
    sm_scale: float,
    out_dtype: torch.dtype,
    device: torch.device,
    *,
    with_lse: bool,
) -> _GraphBundle:
    key = (b, h, hkv, sq, skv, d, is_causal, sm_scale, out_dtype, with_lse)
    if key not in _GRAPH_CACHE:
        logger.debug(
            f"[CuDNNAttention] compiling graph: b={b} h={h} hkv={hkv} sq={sq} skv={skv} "
            f"d={d} causal={is_causal} dtype={out_dtype} lse={with_lse}"
        )
        _GRAPH_CACHE[key] = _build_graph(
            b,
            h,
            hkv,
            sq,
            skv,
            d,
            is_causal,
            sm_scale,
            out_dtype,
            device,
            with_lse=with_lse,
        )
    return _GRAPH_CACHE[key]


@torch.compiler.disable
def _execute_graph(bundle: _GraphBundle, tensor_map: dict, device: torch.device) -> None:
    handle = _get_cudnn_handle()
    cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream().cuda_stream)
    workspace = torch.empty(bundle.workspace_size, dtype=torch.uint8, device=device)
    bundle.graph.execute(tensor_map, workspace, handle=handle)


class CuDNNAttention(AttentionBackend):
    """cuDNN MXFP8 SDPA backend for visual generation.

    Q/K/V are quantized to MXFP8 (FP8 e4m3fn + E8M0 per-32-element block scales)
    using ``torch.ops.trtllm.mxfp8_quantize``.

    Requires cuDNN MXFP8 SDPA support on Blackwell GPUs.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        quant_attention_config: Optional[QuantAttentionConfig] = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype or torch.bfloat16
        self.quant_attention_config = quant_attention_config
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.HND

    def _run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        *,
        with_lse: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _check_cudnn_available()

        b, h, sq, d = q.shape
        _, hkv, skv, _ = k.shape
        out_dtype = self.dtype if q.dtype in (torch.float16, torch.bfloat16) else q.dtype

        q_fp8, q_sf = quantize_mxfp8_qk(q)
        k_fp8, k_sf = quantize_mxfp8_qk(k)
        v_fp8, v_sf = quantize_mxfp8_v(v)

        bundle = _get_or_build_graph(
            b,
            h,
            hkv,
            sq,
            skv,
            d,
            is_causal=is_causal,
            sm_scale=self.scale,
            out_dtype=out_dtype,
            device=q.device,
            with_lse=with_lse,
        )

        output = torch.empty(b, h, sq, d, dtype=out_dtype, device=q.device)
        stats_raw = torch.empty(b, h, sq, 1, dtype=torch.float32, device=q.device)
        amax_o = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=q.device)

        tensor_map: dict = {
            bundle.q_t: q_fp8,
            bundle.k_t: k_fp8,
            bundle.v_t: v_fp8,
            bundle.sf_q_t: q_sf,
            bundle.sf_k_t: k_sf,
            bundle.sf_v_t: v_sf,
            bundle.o_t: output,
            bundle.stats_t: stats_raw,
            bundle.amax_o_t: amax_o,
        }

        _execute_graph(bundle, tensor_map, q.device)

        lse: Optional[torch.Tensor] = None
        if with_lse:
            # cuDNN returns [B, H, S, 1]; reshape to [B, S, H] to match other backends.
            lse = stats_raw.squeeze(-1).permute(0, 2, 1).contiguous()

        return output, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        output, _ = self.forward_with_lse(q, k, v, attention_mask=attention_mask, **kwargs)
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, H, S, D]
            lse:    [B, S, H] float32
        """
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        output, lse = self._run(q, k, v, is_causal=is_causal, with_lse=True)
        if lse is None:
            raise RuntimeError("cuDNN graph didn't produce LSE")
        return output, lse

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout
