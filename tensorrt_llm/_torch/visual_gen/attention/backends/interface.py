# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Visual Generation Attention Backend Interface

Defines shared types, enums, and the abstract base class for attention backends.
"""

from abc import ABC, abstractmethod
from enum import Enum

import torch


class AttentionTensorLayout(str, Enum):
    """
    Tensor layout for attention backend input/output.

    Backends declare their preferred layout so the attention module
    can reshape tensors optimally before calling the backend.
    """

    NHD = "NHD"  # [B, S, H, D] - batch, seq, heads, dim
    HND = "HND"  # [B, H, S, D] - batch, heads, seq, dim


class AttentionBackend(ABC):
    """Contract for all visual-gen attention backends.

    Every backend must implement ``forward`` and declare a ``preferred_layout``.
    Backends pick the kwargs they need from the caller and ignore the rest
    via ``**kwargs``.
    """

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over ``q``, ``k``, ``v`` in the backend's ``preferred_layout``.

        Keyword contract shared by every backend (unknown keywords are ignored):

        * ``batch_size``, ``seq_len``, ``seq_len_kv``: the batch and the query and
          key sequence lengths.
        * ``attention_mask``, ``key_padding_mask``: the mask, if the backend takes one.
        * ``kv_start``, ``kv_len``: only for backends whose ``support_kv_window()`` is true.
          ``[batch_size]`` int32 device tensors. ``k`` and ``v`` are then one shared buffer
          ``[1, N, H_kv, D]`` and sequence ``b`` of ``q`` attends to the rows
          ``[kv_start[b], kv_start[b] + kv_len[b])`` of it. Ranges are free: they may overlap,
          repeat, or be empty, and an empty range gives zero output and an LSE of ``-inf``
          from ``forward_with_lse``.
        """

    @property
    @abstractmethod
    def preferred_layout(self) -> AttentionTensorLayout: ...

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support LSE output. "
            "Override forward_with_lse() or check support_lse() before calling."
        )

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        """Whether the backend supports returning the softmax log-sum-exp (LSE) of the attention weights."""
        return False

    def support_kv_window(self) -> bool:
        """Whether ``forward`` takes ``kv_start`` and ``kv_len``: every query sequence attends to
        its own range of one shared K/V buffer. A backend without support would silently drop
        the keywords through ``**kwargs``, so callers must check before passing them."""
        return False
