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
CuTe DSL (NVIDIA kernels) Backend for Visual Generation Models

Uses CUTE DSL JIT kernel published in CUTLASS, or its pre-compiled derivatives.
Expects NHD layout ([B, S, H, D]) and supports float16/bfloat16.

Cute kernel source: tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/blackwell/attention/cubins
(https://github.com/nvidia/cutlass at commit e406c186f510a15091cce01f782020ceb7ba8eb5)
"""

import math
from typing import Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_cute_dsl_import_error = None
try:
    import tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention as cute_dsl
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention.fmha import (
        _cute_runtime_import_error,
    )

    if _cute_runtime_import_error is not None:
        raise ImportError(_cute_runtime_import_error)
except (ImportError, OSError) as e:
    cute_dsl = None
    _cute_dsl_import_error = e


class CuTeDSLAttention(AttentionBackend):
    """
    CuTe DSL (NVIDIA kernels) backend for diffusion models.

    Uses pre-compiled cubin kernels when allowed & present.
    Otherwise compiles the kernels with CuTe DSL JIT.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        allow_cubins: bool = True,
        context_quantization_mode: str = "NO_QUANT",
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.allow_cubins = allow_cubins
        if context_quantization_mode != "NO_QUANT":
            if self.head_dim in [128] and context_quantization_mode in ["QK16PV8"]:
                pass
            else:
                logger.critical(
                    "Backend CUTEDSL only supports quantizing head_dim=128 w/ QK16PV8. "
                    f"Ignoring the supplied option: {context_quantization_mode=}"
                )
                context_quantization_mode = "NO_QUANT"
        self.context_quantization_mode = context_quantization_mode
        self.scale = 1.0 / math.sqrt(head_dim)

        # CuTe DSL expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.dtype]:
        """Cast inputs to CuTeDSL-compatible dtype and resolve causal flag."""
        if _cute_dsl_import_error is not None:
            raise ImportError(
                f"CuTe DSL kernels are not available. Import error: {_cute_dsl_import_error}"
            ) from _cute_dsl_import_error

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # The JIT source code only supports float16 and bfloat16
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return q, k, v, is_causal, origin_dtype

    # cute_dsl.cute_dsl_fmha_fwd is already decorated with @torch.compiler.disable
    # Allow torch.compile to fuse preceding linear/norm with quantization of V / seq-preprocess
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, num_heads, _ = q.shape
        _, seq_len_kv, _, value_head_dim = v.shape
        out = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            value_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )

        # Options that instructs quantization of V
        scale_v = kwargs.get("scale_v", 1.0)
        if self.context_quantization_mode in ["QK16PV8"]:
            v_qscale = 448.0 / v.abs().amax().clamp(min=1e-3)
            v = (v * v_qscale).to(torch.float8_e4m3fn)
            scale_v = scale_v / v_qscale

        # Sequence preproc.
        qo_indptr_host = [i * seq_len_q for i in range(batch_size + 1)]
        qo_indptr = torch.tensor(qo_indptr_host).to(device=q.device, dtype=torch.int32)
        kv_indptr_host = [i * seq_len_kv for i in range(batch_size + 1)]
        kv_indptr = torch.tensor(kv_indptr_host).to(device=q.device, dtype=torch.int32)

        cute_dsl.cute_dsl_fmha_fwd(
            q.flatten(0, 1).contiguous(),
            k.flatten(0, 1).contiguous(),
            v.flatten(0, 1).contiguous(),
            out.flatten(0, 1),
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse.flatten(0, 1).contiguous(),
            scale_q=kwargs.get("scale_q", 1.0),
            scale_k=kwargs.get("scale_k", 1.0),
            scale_v=scale_v,
            scale_o=kwargs.get("scale_o", 1.0),
            max_qo_len=seq_len_q,
            max_kv_len=seq_len_kv,
            is_persistent=False,
            allow_cubins=self.allow_cubins,
            skip_softmax_threshold_scale_factor=kwargs.get("skip_softmax_threshold_scale_factor"),
        )
        return out, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using CuTe DSL (NVIDIA kernels).

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            attention_mask: Attention mask type (CAUSAL or FULL)

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
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
        Forward pass returning both output and log-sum-exp (LSE).

        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            lse:    [batch_size, num_heads, seq_len] - log-sum-exp per query position,
                    always in float32. Used for numerically stable combination of
                    partial attention results in Attention2D parallelism.
        """
        q, k, v, is_causal, origin_dtype = self._prepare_inputs(q, k, v, attention_mask)
        output, lse = self._fwd(q, k, v, is_causal, **kwargs)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output, lse.transpose(1, 2)

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
