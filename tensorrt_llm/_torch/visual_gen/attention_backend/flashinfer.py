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
FlashInfer ragged DiT attention backend for visual generation models.

This backend calls FlashInfer's trtllm_ragged_attention_deepseek kernel
directly and does not use the LLM attention backend or metadata path.
"""

import math
from typing import Optional

import torch

from tensorrt_llm.models.modeling_utils import QuantConfig

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

try:
    import flashinfer

    if not torch.cuda.is_available():
        raise ImportError("FlashInfer ragged DiT attention requires CUDA.")

    # TODO: Implement fallback for non-SM10X HWs. What is needed:
    # - Use a different entry point (batch_prefill_with_kv_cache)
    # - Add fallback for quantization type (quantization not available for non-SM10X)
    # - Make fallback decision visible to callers.
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if cc_major != 10:
        raise ImportError(
            f"FlashInfer ragged DiT attention requires SM100/SM103, got SM{cc_major}{cc_minor}."
        )

    _trtllm_ragged_attention_deepseek = flashinfer.prefill.trtllm_ragged_attention_deepseek
except (ImportError, AttributeError, RuntimeError) as exc:
    raise ImportError("FlashInfer ragged DiT attention is not available.") from exc

_WORKSPACE_SIZE = 256 * 1024 * 1024
_WORKSPACES: dict[torch.device, torch.Tensor] = {}


def _get_workspace(device: torch.device) -> torch.Tensor:
    workspace = _WORKSPACES.get(device)
    if workspace is None:
        workspace = torch.zeros(_WORKSPACE_SIZE, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = workspace
    return workspace


@torch.compile
def _to_float8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    amax = x.float().abs().amax().clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_fp8 = (x.float() * scale).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    return x_fp8, scale.float().reciprocal()


class FlashInferAttention(AttentionBackend):
    """
    FlashInfer ragged DiT attention wrapper.

    Supported ``AttentionConfig.quantization_type`` values:
        - ``no_quant``: pass Q/K/V tensors directly
        - ``qkv_fp8``: quantize Q/K/V to FP8 E4M3
        - ``qk_bf16_v_fp8``: cast Q/K to BF16 and quantize V to FP8 E4M3
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        dtype: Optional[torch.dtype] = None,
        attention_config=None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.quant_config = quant_config
        self.dtype = dtype
        self.quantization_type = getattr(attention_config, "quantization_type", "no_quant")
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        if self.quantization_type == "qkv_fp8":
            q_fp8, q_inv_scale = _to_float8(q)
            k_fp8, k_inv_scale = _to_float8(k)
            v_fp8, v_inv_scale = _to_float8(v)
            bmm1_scale = self.scale * q_inv_scale * k_inv_scale
            return q_fp8, k_fp8, v_fp8, float(bmm1_scale), float(v_inv_scale)

        if self.quantization_type == "qk_bf16_v_fp8":
            v_fp8, v_inv_scale = _to_float8(v)
            return (
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v_fp8,
                self.scale,
                float(v_inv_scale),
            )

        return q, k, v, self.scale, 1.0

    @torch.compiler.disable
    def _forward_flashinfer(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        q_len = q.shape[1]
        kv_len = k.shape[1]
        total_q = batch_size * q_len
        total_kv = batch_size * kv_len

        q = q.reshape(total_q, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(total_kv, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(total_kv, self.num_kv_heads, self.head_dim).contiguous()
        q, k, v, bmm1_scale, bmm2_scale = self._prepare_inputs(q, k, v)

        q_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=q.device)
        kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=q.device)
        qo_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=q.device), q_lens.cumsum(0).int()]
        )
        kv_indptr = torch.cat(
            [torch.zeros(1, dtype=torch.int32, device=q.device), kv_lens.cumsum(0).int()]
        )

        output, _lse = _trtllm_ragged_attention_deepseek(
            q,
            k,
            v,
            _get_workspace(q.device),
            kv_lens,
            q_len,
            kv_len,
            bmm1_scale,
            bmm2_scale,
            -1,
            batch_size,
            -1,
            qo_indptr,
            kv_indptr,
            False,
            causal,
            True,
        )
        return output.reshape(batch_size, q_len, self.num_heads, self.head_dim)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        return self._forward_flashinfer(q, k, v, is_causal)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
