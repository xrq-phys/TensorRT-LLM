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
"""References for attention over a cyclic K/V cache.

``dense_window_reference`` knows nothing about the cache: it keeps every token of the rollout
in plain tensors and selects the visible ones by position. ``ReferenceWindowAttention`` is a
backend that attends to ranges of a shared K/V buffer in torch, a stand-in for a kernel.
"""

import math

import torch

from tensorrt_llm._torch.visual_gen.attention.backends.interface import (
    AttentionBackend,
    AttentionTensorLayout,
)


def visible_positions(start: int, end: int, fixed: int, window: int) -> torch.Tensor:
    """Logical key positions of a causal block ``[start, end)``.

    The fixed region, and everything from ``window`` before the block to its end.
    """
    first = max(0, start - window)
    pos = torch.arange(end)
    return pos[(pos < fixed) | (pos >= first)]


def attend(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, sm_scale: float):
    """fp32 attention of ``q`` ``[T, H_q, D]`` over ``keys``/``values`` ``[S, H_kv, D]``.

    The output in ``q``'s dtype and the log-sum-exp ``[H_q, T]``.
    """
    rep = q.shape[1] // keys.shape[1]
    kx = keys.repeat_interleave(rep, dim=1).float()
    vx = values.repeat_interleave(rep, dim=1).float()
    scores = torch.einsum("thd,shd->hts", q.float(), kx) * sm_scale
    out = torch.einsum("hts,shd->thd", scores.softmax(dim=-1), vx)
    return out.to(q.dtype), torch.logsumexp(scores, dim=-1)


def dense_window_reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    start: int,
    fixed: int,
    window: int,
    block_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Attention of the ``T`` tokens of ``q`` at logical positions ``[start, start + T)``.

    ``keys``/``values`` hold every token of the rollout up to ``start + T``, indexed by
    logical position (the staged tokens at their positions). Block ``j`` of ``block_size``
    tokens sees the fixed region, the ``window`` tokens before its first token and
    everything up to its own end.
    """
    outs = []
    for lo in range(0, q.shape[0], block_size):
        s, e = start + lo, start + lo + block_size
        pos = visible_positions(s, e, fixed, window).to(q.device)
        outs.append(attend(q[lo : lo + block_size], keys[pos], values[pos], sm_scale)[0])
    return torch.cat(outs)


class ReferenceWindowAttention(AttentionBackend):
    """A backend attending to ranges of a shared K/V buffer, in torch."""

    def __init__(self, head_dim: int) -> None:
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward_with_lse(self, q, k, v, *, kv_start, kv_len, **kwargs):
        batch, seq_len, heads, _ = q.shape
        out = torch.zeros_like(q)
        lse = torch.full((batch, heads, seq_len), float("-inf"), device=q.device)
        for i, (start, length) in enumerate(zip(kv_start.tolist(), kv_len.tolist())):
            if length:
                rows = slice(start, start + length)
                out[i], lse[i] = attend(q[i], k[0, rows], v[0, rows], self.scale)
        return out, lse

    def forward(self, q, k, v, **kwargs):
        return self.forward_with_lse(q, k, v, **kwargs)[0]

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return AttentionTensorLayout.NHD

    @classmethod
    def support_lse(cls) -> bool:
        return True

    def support_kv_window(self) -> bool:
        return True
