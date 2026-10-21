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
"""Attention over a ``CyclicKVCache``, built on a backend that attends to a K/V window."""

from typing import Optional

import torch

from ....attention.backends.interface import PredefinedAttentionMask
from ...cache import CyclicKVCache
from .interface import AttentionBackend, AttentionTensorLayout


class CyclicKVAttention(AttentionBackend):
    """Wraps a backend whose ``support_lse()`` is true and whose layout is NHD.

    With a ``kv_cache`` the new tokens' K/V are staged in the cache and every causal block of
    them attends to the cache's fixed region and its window. Those are at most two ranges of
    the cache's buffer, which the inner backend attends to one at a time; the two partial
    results are merged by their LSE. Without a ``kv_cache`` this is the inner backend.

    A backend whose ``support_kv_window()`` is true attends to a range for every block in one
    call. Any other backend is called once per block and range, on a slice of the buffer:
    correct but slow, and not capturable in a CUDA graph, since the slices are taken on the
    host.

    It sits innermost among the wrappers of ``wrap_parallel_attention``, so the cache holds
    what one rank attends to: with Ulysses, its heads of the whole sequence.
    """

    def __init__(self, inner_backend: AttentionBackend, layer_idx: int) -> None:
        if not inner_backend.support_lse():
            raise ValueError(
                f"{type(inner_backend).__name__} does not return the LSE that merges the two "
                "ranges of the cache; use CUTEDSL, FA4, FLASHINFER or CUDNN."
            )
        if inner_backend.preferred_layout != AttentionTensorLayout.NHD:
            raise ValueError(f"{type(inner_backend).__name__} does not take the NHD layout.")
        self.inner_backend = inner_backend
        self.layer_idx = layer_idx

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_cache: Optional[CyclicKVCache] = None,
        causal_block_size: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attention, over the cache when there is one.

        Args:
            kv_cache: A ``CyclicKVCache``. When given, ``k``/``v`` are the new tokens only:
                they are written at ``staging_offset`` and each causal block attends to the
                cache's fixed region, the ``window_tokens`` before the block, and the new
                tokens up to the block's end. ``batch_size`` must be 1.
            seq_len: With ``kv_cache``, the number of real tokens. Rows past it are padding
                added so the sequence splits evenly across ranks; they are neither written to
                the cache nor attended, and their output rows are zero.
            causal_block_size: With ``kv_cache``, cuts the real tokens into consecutive causal
                blocks: full attention within a block, causal across blocks. Absent: one block.
        """
        if kv_cache is None:
            if causal_block_size is not None:
                raise NotImplementedError(
                    "causal_block_size is only implemented over a K/V cache; pass kv_cache."
                )
            return self.inner_backend.forward(q, k, v, **kwargs)
        return self._forward_with_kv_cache(q, k, v, kv_cache, causal_block_size, **kwargs)

    def _forward_with_kv_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: CyclicKVCache,
        causal_block_size: Optional[int],
        *,
        seq_len: Optional[int] = None,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """``q``, ``k``, ``v`` are ``[1, S, H, D]`` / ``[1, S, H_kv, D]``: one video."""
        if attention_mask != PredefinedAttentionMask.FULL:
            raise NotImplementedError("K/V cache attention is full attention over the cache.")
        batch, num_rows, num_heads, head_dim = q.shape
        if batch != 1 or k.shape[1] != num_rows or v.shape != k.shape:
            raise ValueError(
                "K/V cache attention takes one video: q, k, v of [1, S, heads, head_dim]."
            )
        if k.shape[2] != kv_cache.num_kv_heads:
            raise ValueError(
                f"k has {k.shape[2]} heads but the cache holds {kv_cache.num_kv_heads} per rank."
            )
        num_tokens = num_rows if seq_len is None else seq_len
        if not 0 < num_tokens <= num_rows:
            raise ValueError(
                f"seq_len {seq_len} outside (0, {num_rows}]: it counts the real tokens; "
                "rows past it are padding."
            )
        if q.dtype != kv_cache.dtype:
            raise TypeError(f"q is {q.dtype} but the cache holds {kv_cache.dtype}")
        block_size, num_blocks = kv_cache.causal_blocks(num_tokens, causal_block_size)
        kv_cache.write_staged(self.layer_idx, k[0, :num_tokens], v[0, :num_tokens])
        k_buffer, v_buffer = kv_cache.kv_buffers(self.layer_idx)
        q_blocks = q[0, :num_tokens].reshape(num_blocks, block_size, num_heads, head_dim)
        if self.inner_backend.support_kv_window():
            (out_a, lse_a), (out_b, lse_b) = (
                self._attend_ranges(q_blocks, k_buffer[None], v_buffer[None], ranges, **kwargs)
                for ranges in kv_cache.kv_ranges(num_blocks, block_size)
            )
        else:
            starts, lengths = kv_cache.kv_ranges_host(num_blocks, block_size)
            (out_a, lse_a), (out_b, lse_b) = (
                self._attend_ranges_per_block(
                    q_blocks, k_buffer, v_buffer, starts[j], lengths[j], **kwargs
                )
                for j in range(2)
            )
        merged_lse = torch.logaddexp(lse_a, lse_b)
        merged = out_a.float() * torch.exp(lse_a - merged_lse).unsqueeze(-1) + out_b.float() * (
            torch.exp(lse_b - merged_lse).unsqueeze(-1)
        )
        result = torch.zeros(num_rows, num_heads, head_dim, dtype=q.dtype, device=q.device)
        result[:num_tokens] = merged.reshape(num_tokens, num_heads, head_dim).to(q.dtype)
        return result.unsqueeze(0)

    def _attend_ranges(
        self,
        q_blocks: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        ranges,
        **kwargs,
    ):
        """One attention call: every block of ``q_blocks`` over its range of the buffer.
        Returns the output ``[blocks, S, H, D]`` and the LSE ``[blocks, S, H]``, ``-inf`` where
        the range is empty.

        A backend that attends to both ranges of a block in one call would replace the two
        calls this module makes, one per range.
        """
        kv_start, kv_len = ranges
        out, lse = self.inner_backend.forward_with_lse(
            q_blocks, k_buffer, v_buffer, kv_start=kv_start, kv_len=kv_len, **kwargs
        )
        return out, lse.transpose(1, 2)

    def _attend_ranges_per_block(
        self,
        q_blocks: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        starts,
        lengths,
        **kwargs,
    ):
        """``_attend_ranges`` for a backend without ``support_kv_window()``: one call per block
        over its slice of the buffer, ``starts``/``lengths`` being host arrays."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{type(self.inner_backend).__name__} attends to the cache per block on the "
                "host, which a CUDA graph cannot capture; use CUTEDSL."
            )
        num_blocks, block_size, num_heads, head_dim = q_blocks.shape
        out = torch.zeros_like(q_blocks)
        lse = torch.full((num_blocks, block_size, num_heads), float("-inf"), device=q_blocks.device)
        for i, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
            if length == 0:
                continue
            out_i, lse_i = self.inner_backend.forward_with_lse(
                q_blocks[i : i + 1],
                k_buffer[None, start : start + length],
                v_buffer[None, start : start + length],
                **{**kwargs, "batch_size": 1, "seq_len": block_size, "seq_len_kv": length},
            )
            out[i] = out_i[0]
            lse[i] = lse_i[0].transpose(0, 1)
        return out, lse

    def forward_with_lse(self, q, k, v, **kwargs):
        if kwargs.get("kv_cache") is not None:
            raise NotImplementedError("forward_with_lse does not support a K/V cache.")
        return self.inner_backend.forward_with_lse(q, k, v, **kwargs)

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self.inner_backend.preferred_layout

    def support_lse(self) -> bool:
        return self.inner_backend.support_lse()

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
