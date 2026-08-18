# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer CuTe DSL FMHA backend for visual generation models.

This adapter calls FlashInfer's precompiled ragged-prefill CuTe DSL kernels.
Visual-generation tensors use a dense ``[B, S, H, D]`` layout, so the adapter
flattens the batch and supplies uniform sequence offsets to FlashInfer.
"""

import math
from typing import Optional

import torch

from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_FP8_E4M3_MAX = 448.0  # FP8 e4m3 max magnitude

_flashinfer_cute_dsl_import_error = None
try:
    from flashinfer.attention.cute_dsl.fmha import cute_dsl_fmha_ragged_prefill
except (ImportError, OSError) as e:
    cute_dsl_fmha_ragged_prefill = None
    _flashinfer_cute_dsl_import_error = e


class FlashInferCuTeDSLAttention(AttentionBackend):
    """FlashInfer CuTe DSL FMHA backend for diffusion models."""

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
        self.dtype = dtype
        if quant_attention_config is not None and quant_attention_config.qk_dtype not in (
            "bf16",
            "fp16",
        ):
            raise ValueError(
                "FlashInfer CuTe DSL attention only supports the QK16PV8 recipe "
                f"(qk_dtype in bf16/fp16); got qk_dtype={quant_attention_config.qk_dtype!r}."
            )
        self.quant_attention_config = quant_attention_config
        self.scale = 1.0 / math.sqrt(head_dim)
        self._preferred_layout = AttentionTensorLayout.NHD
        self._indptr_cache: dict[tuple[torch.device, int, int], torch.Tensor] = {}

    def _get_indptr(
        self,
        device: torch.device,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        cache_key = (device, batch_size, seq_len)
        indptr = self._indptr_cache.get(cache_key)
        if indptr is None:
            indptr = torch.arange(
                batch_size + 1,
                device=device,
                dtype=torch.int32,
            )
            indptr *= seq_len
            self._indptr_cache[cache_key] = indptr
        return indptr

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.dtype]:
        if cute_dsl_fmha_ragged_prefill is None:
            raise ImportError(
                "FlashInfer CuTe DSL FMHA is not available. "
                f"Import error: {_flashinfer_cute_dsl_import_error}"
            ) from _flashinfer_cute_dsl_import_error

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        return q, k, v, is_causal, origin_dtype

    @torch.compiler.disable
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]
        num_kv_heads = k.shape[2]
        value_head_dim = v.shape[-1]

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})."
            )
        if k.shape[:3] != v.shape[:3]:
            raise ValueError(
                f"K and V batch/sequence/head dimensions must match, got "
                f"{tuple(k.shape[:3])} and {tuple(v.shape[:3])}."
            )

        q_flat = q.contiguous().view(batch_size * seq_len_q, num_heads, head_dim)
        k_flat = k.contiguous().view(batch_size * seq_len_kv, num_kv_heads, head_dim)
        v_flat = v.contiguous().view(batch_size * seq_len_kv, num_kv_heads, value_head_dim)
        # QK16PV8: Q/K stay in BF16/FP16 while V is quantized to FP8 (e4m3, per tensor).
        # The dequant factor is folded into the kernel's `scale_v`.
        scale_v = 1.0
        if self.quant_attention_config is not None:
            v_qscale = _FP8_E4M3_MAX / v_flat.abs().amax().clamp(min=1e-3)
            v_flat = (v_flat * v_qscale).to(torch.float8_e4m3fn)
            scale_v = float(v_qscale.reciprocal())
        output = torch.empty(
            batch_size * seq_len_q,
            num_heads,
            value_head_dim,
            device=q.device,
            dtype=q.dtype,
        )
        lse = torch.empty(
            batch_size * seq_len_q,
            num_heads,
            device=q.device,
            dtype=torch.float32,
        )
        qo_indptr = self._get_indptr(q.device, batch_size, seq_len_q)
        kv_indptr = self._get_indptr(q.device, batch_size, seq_len_kv)

        cute_dsl_fmha_ragged_prefill(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            o=output,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse,
            scale_v=scale_v,
            max_qo_len=seq_len_q,
            max_kv_len=seq_len_kv,
        )
        return (
            output.view(batch_size, seq_len_q, num_heads, value_head_dim),
            lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2),
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        output, _ = self.forward_with_lse(
            q,
            k,
            v,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
        )
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_padding_mask is not None:
            raise NotImplementedError(
                "FlashInfer CuTe DSL attention does not support padded dense batches."
            )
        q, k, v, is_causal, origin_dtype = self._prepare_inputs(q, k, v, attention_mask)
        output, lse = self._fwd(q, k, v, is_causal)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output, lse

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout
