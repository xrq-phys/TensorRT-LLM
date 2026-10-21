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
"""K/V cache for a causal video rollout: one contiguous cyclic buffer per layer.

One rollout is one logical sequence of tokens

    [ fixed | old | window | staged | future ]

and a causal block of new tokens only ever attends to two ranges of it: the fixed region,
and the ``window_tokens`` before the block through the end of the block. So the cache holds
the fixed region and a ring that is big enough for the window and the staged tokens, and a
token past the fixed region is stored at ``fixed + (position - fixed) % ring``. Nothing is
evicted or copied: staging the next tokens overwrites the ones that fell out of the window.

Physically a range of the ring may wrap around the end of the buffer, but the ring starts
right after the fixed region, so what a block attends to is always at most two contiguous
ranges of the buffer: the fixed region together with the wrapped part, and the rest.
``kv_ranges`` returns those two ranges for each block, to be attended separately and merged.

Rotary positions are the caller's business and are absolute over the rollout.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

# A range of buffer rows per block: (start, length) int32 device tensors.
Range = Tuple[torch.Tensor, torch.Tensor]


class CyclicKVCache:
    """A rollout's K/V in one cyclic buffer per layer.

    Args:
        num_layers: attention layers that persist K/V (the generator tower).
        num_kv_heads: K/V heads this rank holds, i.e. the head count the attention backend
            it serves was built with. Head sharding (tensor parallel, Ulysses) is settled by
            the model before K/V reach the cache.
        head_dim, dtype: K/V geometry.
        pin_tokens: the first tokens committed are the fixed region: always visible, never
            overwritten.
        window_tokens: history each causal block sees before its first token, in tokens
            (``(window_frames - 1) * tokens_per_frame``).
        max_staged_tokens: most tokens a forward may write before a commit
            (``chunk_frames * tokens_per_frame``).
        device: where the buffers live; the current CUDA device by default.

    Device tensors that forwards read are allocated once and rewritten in place by ``commit``
    and ``reset``, so a CUDA graph captured around a forward keeps reading current values.
    """

    @staticmethod
    def capacity_tokens(*, pin_tokens: int, window_tokens: int, max_staged_tokens: int) -> int:
        """Tokens one layer's buffer holds: the fixed region and a ring for the window and
        the staged tokens."""
        _check_geometry(pin_tokens, window_tokens, max_staged_tokens)
        return pin_tokens + window_tokens + max_staged_tokens

    @staticmethod
    def pool_bytes_for(
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
    ) -> int:
        """Device bytes the buffers of this geometry take: K and V of every token in every
        layer."""
        tokens = CyclicKVCache.capacity_tokens(
            pin_tokens=pin_tokens,
            window_tokens=window_tokens,
            max_staged_tokens=max_staged_tokens,
        )
        element_size = torch.tensor([], dtype=dtype).element_size()
        return num_layers * 2 * tokens * num_kv_heads * head_dim * element_size

    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        pin_tokens: int,
        window_tokens: int,
        max_staged_tokens: int,
        device: Optional[torch.device] = None,
    ) -> None:
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"CyclicKVCache supports bf16/fp16 K/V, got {dtype}")
        self.capacity = self.capacity_tokens(
            pin_tokens=pin_tokens,
            window_tokens=window_tokens,
            max_staged_tokens=max_staged_tokens,
        )
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.pin_tokens = pin_tokens
        self.window_tokens = window_tokens
        self.max_staged_tokens = max_staged_tokens
        self.ring_tokens = window_tokens + max_staged_tokens
        # Zeros, so that a kernel reading past the end of a range, which it then masks, reads
        # finite values.
        self._buffer = torch.zeros(
            num_layers, 2, self.capacity, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        self._committed_tokens = 0
        self._fixed_tokens = 0
        # Buffer row of each token the next forward may stage, in staging order.
        self._staged_rows = torch.zeros(
            pin_tokens + max_staged_tokens, dtype=torch.int64, device=device
        )
        # Key ranges per (num_blocks, block_size), made on first use.
        self._ranges: Dict[Tuple[int, int], Tuple[Range, Range]] = {}
        self._refresh()

    # ------------------------------------------------------------------ the rollout

    def reset(self) -> None:
        """Start a new rollout: the sequence is empty and ``pin_tokens + max_staged_tokens``
        tokens may be staged."""
        self._committed_tokens = 0
        self._fixed_tokens = 0
        self._refresh()

    @property
    def fixed_tokens(self) -> int:
        """Tokens every block sees for the whole rollout; logical ``[0, fixed_tokens)``."""
        return self._fixed_tokens

    @property
    def staging_offset(self) -> int:
        """Logical position of the first staged token: the tokens committed so far. Rotary
        positions are absolute over the rollout and are the caller's business."""
        return self._committed_tokens

    @property
    def staging_capacity(self) -> int:
        """Most tokens the next forward may stage. The fixed region may be staged in one go
        on top of ``max_staged_tokens``."""
        return self.max_staged_tokens + self.pin_tokens - self._fixed_tokens

    def commit(self, num_tokens: int) -> None:
        """The ``num_tokens`` staged tokens, written at ``staging_offset``, are final. Those
        that fall within the first ``pin_tokens`` become fixed; the window moves past the
        rest. The count is the caller's: a forward may stage fewer than ``max_staged_tokens``."""
        if not 0 < num_tokens <= self.staging_capacity:
            raise ValueError(
                f"commit of {num_tokens} tokens; the next forward stages 1..{self.staging_capacity}"
            )
        self._committed_tokens += num_tokens
        self._fixed_tokens = min(self.pin_tokens, self._committed_tokens)
        self._refresh()

    @staticmethod
    def causal_blocks(num_tokens: int, causal_block_size: Optional[int]) -> Tuple[int, int]:
        """The block size and block count a forward of ``num_tokens`` real tokens uses: one
        block of all of them when ``causal_block_size`` is None; otherwise the size must be
        positive and divide ``num_tokens``. Shared by the attention backends so they agree
        on the contract."""
        size = num_tokens if causal_block_size is None else int(causal_block_size)
        if size <= 0 or num_tokens % size:
            raise ValueError(
                f"{num_tokens} tokens do not split into causal blocks of {causal_block_size}"
            )
        return size, num_tokens // size

    # ------------------------------------------------------------------ the buffer

    def row(self, position: np.ndarray) -> np.ndarray:
        """Buffer row of each logical position: the fixed region in place, the rest in the
        ring right after it."""
        position = np.asarray(position, dtype=np.int64)
        fixed = self.pin_tokens
        return np.where(position < fixed, position, fixed + (position - fixed) % self.ring_tokens)

    def kv_buffers(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The layer's ``(K, V)``, each a contiguous ``[capacity, num_kv_heads, head_dim]``."""
        return self._buffer[layer_idx, 0], self._buffer[layer_idx, 1]

    def write_staged(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Stage ``T`` tokens of K/V, the forward's new tokens, at logical positions
        ``[staging_offset, staging_offset + T)``. ``k``/``v`` are ``[T, num_kv_heads,
        head_dim]``. Rewrites the same positions on every call until ``commit``."""
        num_tokens = k.shape[0]
        if v.shape != k.shape or k.shape[1:] != (self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"staged write of k {tuple(k.shape)}, v {tuple(v.shape)} does not fit "
                f"[T, {self.num_kv_heads}, {self.head_dim}]"
            )
        if not 0 < num_tokens <= self.staging_capacity:
            raise ValueError(
                f"staged write of {num_tokens} tokens; the next forward stages "
                f"1..{self.staging_capacity}"
            )
        if k.dtype != self.dtype or v.dtype != self.dtype:
            raise TypeError(f"K/V dtype {k.dtype} does not match the cache's {self.dtype}")
        rows = self._staged_rows[:num_tokens]
        k_buffer, v_buffer = self.kv_buffers(layer_idx)
        k_buffer.index_copy_(0, rows, k)
        v_buffer.index_copy_(0, rows, v)

    def kv_ranges(self, num_blocks: int, block_size: int) -> Tuple[Range, Range]:
        """The buffer rows each causal block of ``block_size`` new tokens attends to, as two
        ranges ``((start, length), (start, length))``, each a pair of int32 device tensors of
        ``num_blocks`` entries: block ``i`` attends to the rows ``[start[i], start[i] +
        length[i])`` of each.

        Together they are the fixed region, the ``window_tokens`` before the block, and
        everything up to the block's end; the second range is empty when one covers them.
        Persistent per ``(num_blocks, block_size)`` and refreshed in place by ``commit``.
        """
        key = (num_blocks, block_size)
        ranges = self._ranges.get(key)
        if ranges is None:
            device = self._staged_rows.device
            ranges = tuple(
                tuple(torch.zeros(num_blocks, dtype=torch.int32, device=device) for _ in range(2))
                for _ in range(2)
            )
            self._ranges[key] = ranges
            self._fill_ranges(key, ranges)
        return ranges

    def kv_ranges_host(self, num_blocks: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """``kv_ranges`` as host arrays ``(starts, lengths)``, each ``[2, num_blocks]`` int32:
        row ``j`` is range ``j``. Reflects the current commit; for callers that slice the
        buffer per block on the host."""
        return self._range_table(num_blocks, block_size)

    # ------------------------------------------------------------------ device state

    def _refresh(self) -> None:
        count = self.staging_capacity
        positions = self._committed_tokens + np.arange(count, dtype=np.int64)
        self._staged_rows[:count].copy_(torch.from_numpy(self.row(positions)))
        for key, ranges in self._ranges.items():
            self._fill_ranges(key, ranges)

    def _fill_ranges(self, key: Tuple[int, int], ranges: Tuple[Range, Range]) -> None:
        starts, lengths = self._range_table(*key)
        for j, (start, length) in enumerate(ranges):
            start.copy_(torch.from_numpy(starts[j]))
            length.copy_(torch.from_numpy(lengths[j]))

    def _range_table(self, num_blocks: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        fixed, ring = self.pin_tokens, self.ring_tokens
        starts = np.zeros((2, num_blocks), dtype=np.int32)
        lengths = np.zeros((2, num_blocks), dtype=np.int32)
        for i in range(num_blocks):
            first = self._committed_tokens + i * block_size
            end = first + block_size
            # The window part is everything past the fixed region from `window_tokens` before
            # the block; the fixed region itself is [0, fixed).
            low = max(first - self.window_tokens, fixed)
            length = max(end - low, 0)
            if length == 0:
                lengths[0, i] = min(fixed, end)
                continue
            offset = (low - fixed) % ring
            wrapped = max(offset + length - ring, 0)
            # The fixed region and the wrapped part are one range: the ring starts right
            # after the fixed region.
            first_end, second_start = fixed + wrapped, fixed + offset
            if first_end == second_start:
                lengths[0, i] = first_end + (length - wrapped)
            else:
                lengths[0, i] = first_end
                starts[1, i] = second_start
                lengths[1, i] = length - wrapped
        return starts, lengths


def _check_geometry(pin_tokens: int, window_tokens: int, max_staged_tokens: int) -> None:
    if window_tokens < 0 or pin_tokens < 0 or max_staged_tokens <= 0:
        raise ValueError(
            "window_tokens and pin_tokens must be non-negative and max_staged_tokens positive"
        )
