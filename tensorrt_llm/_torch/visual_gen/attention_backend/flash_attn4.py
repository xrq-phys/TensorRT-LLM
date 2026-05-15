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
Flash Attention 4 Backend for Visual Generation Models

Uses Flash Attention 4 CUTE DSL JIT kernel or other CUTE DSL derivatives.
Expects NHD layout ([B, S, H, D]) and supports float16/bfloat16.

Cute kernel source: tensorrt_llm/_torch/visual_gen/jit_kernels/flash_attention/cute/
(https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute
at commit ea8f73506369d7cdd498396474107a978858138c)
"""

import math
from typing import Optional, Tuple

import torch

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_flash_attn_fwd_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.jit_kernels import cute_precompiled as _cute_precompiled
    from tensorrt_llm._torch.visual_gen.jit_kernels.flash_attention.cute.interface import (
        _flash_attn_fwd,
    )
except (ImportError, OSError) as e:
    _flash_attn_fwd = None
    _cute_precompiled = None
    _flash_attn_fwd_import_error = e


class FlashAttn4Attention(AttentionBackend):
    """
    Flash Attention 4 backend for diffusion models.

    Uses precompiled CuTe DSL FMHA kernels when present.
    Otherwise calls flash_attn.cute.interface._flash_attn_fwd, which:
    - Expects [B, S, H, D] (NHD) format
    - Supports float16 and bfloat16 (auto-casts other dtypes)
    - Supports both self-attention and cross-attention (different Q/KV lengths)
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        allow_precompiled_mods: bool = True,
        context_quantization_mode: str = "NO_QUANT",
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.allow_precompiled_mods = allow_precompiled_mods
        self.context_quantization_mode = context_quantization_mode
        self.scale = 1.0 / math.sqrt(head_dim)

        # FA4 expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    @torch.compiler.disable
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calls _flash_attn_fwd with torch.compile disabled. Returns (output, lse)."""
        output, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=self.scale,
            causal=is_causal,
            window_size_left=None,
            window_size_right=None,
            learnable_sink=None,
            softcap=0.0,
            pack_gqa=None,
            mask_mod=None,
            block_sparse_tensors=None,
            return_lse=True,
        )
        return output, lse

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.dtype]:
        """Cast inputs to FA4-compatible dtype and resolve causal flag."""
        if _flash_attn_fwd is None:
            raise ImportError(
                f"FlashAttention 4 is not available. Import error: {_flash_attn_fwd_import_error}"
            ) from _flash_attn_fwd_import_error

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # FA4 only supports float16 and bfloat16
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return q, k, v, is_causal, origin_dtype

    def _fwd_precompiled(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        if not self.allow_precompiled_mods:
            return None

        batch_size, seq_len_q, num_heads, head_dim = q.shape
        _, seq_len_kv, _, value_head_dim = v.shape
        gpu_arch = torch.cuda.get_device_capability(q.device)
        gpu_arch = f"sm_{gpu_arch[0]}{gpu_arch[1]}a"

        # Options that instructs quantization of V
        scale_v = kwargs.get("scale_v", 1.0)
        if self.context_quantization_mode in ["QK16PV8"]:
            v_qscale = 448.0 / v.abs().amax()
            v = (v * v_qscale).to(torch.float8_e4m3fn)
            scale_v = scale_v / float(v_qscale.item())

        skip_softmax_threshold_scale_factor = kwargs.get("skip_softmax_threshold_scale_factor")
        try:
            kernel_fn = _cute_precompiled.get_cute_dsl_fmha_kernel(
                q.dtype,
                v.dtype,
                q.dtype,
                head_dim,
                is_causal,
                is_persistent=True,
                varlen=False,
                enable_tvm_ffi=True,
                with_lse=True,
                enable_skip_softmax=(
                    skip_softmax_threshold_scale_factor is not None
                    and skip_softmax_threshold_scale_factor > 0
                ),
                gpu_arch=gpu_arch,
            )
        except (FileNotFoundError, ImportError, ValueError):
            return None

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
            num_heads,
            seq_len_q,
            dtype=torch.float32,
            device=q.device,
        )

        _cute_precompiled.cute_dsl_fmha_context_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            out,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse,
            scale_q=kwargs.get("scale_q", 1.0),
            scale_k=kwargs.get("scale_k", 1.0),
            scale_v=scale_v,
            scale_o=kwargs.get("scale_o", 1.0),
            max_qo_len=seq_len_q,
            max_kv_len=seq_len_kv,
            kernel_fn=kernel_fn,
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
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
        Forward pass using Flash Attention 4.

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

        result = self._fwd_precompiled(q, k, v, is_causal, **kwargs)
        if result is None:
            output, lse = self._fwd(q, k, v, is_causal)
        else:
            output, lse = result

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output, lse

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
