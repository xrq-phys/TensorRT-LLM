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
MXFP8 block-scaled quantization utilities shared by attention backends.

Both variants call ``torch.ops.trtllm.mxfp8_quantize`` with block size 32. The
operator always returns uint8 E8M0 scale factors for MXFP8; its boolean argument
selects the scale-factor layout.
"""

from typing import Tuple

import torch

_MXFP8_BLOCK = 32  # elements per scale-factor block along D


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_up(x: int, multiple: int) -> int:
    return _ceil_div(x, multiple) * multiple


def quantize_mxfp8_qk(
    x_bhsd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize MXFP8 SDPA Q/K tensors in HND layout.

    Backends with NHD tensors can transpose around this helper. No scalar scale
    is returned because the E8M0 scale factors carry the quantization range.

    Returns:
        x_q: [B, H, S, D] float8_e4m3fn.
        x_sf: [B, H, S_padded, D_scale_padded] uint8 E8M0 scale factors in
            F8_128x4 layout.
    """
    if x_bhsd.dim() != 4:
        raise ValueError(f"quantize_mxfp8_qk expects (B, H, S, D); got shape {x_bhsd.shape}")
    b, h, s, d = x_bhsd.shape
    if d % _MXFP8_BLOCK != 0:
        raise ValueError(f"head_dim={d} must be divisible by MXFP8 block size {_MXFP8_BLOCK}.")
    s_pad = _pad_up(s, 128)
    if s_pad != s:
        pad = x_bhsd.new_zeros(b, h, s_pad - s, d)
        x_padded = torch.cat([x_bhsd, pad], dim=2).contiguous()
    else:
        x_padded = x_bhsd.contiguous()
    x_2d = x_padded.reshape(b * h * s_pad, d)
    x_q_2d, x_sf_1d = torch.ops.trtllm.mxfp8_quantize(x_2d, True, alignment=_MXFP8_BLOCK)
    cols = _pad_up(d // _MXFP8_BLOCK, 4)
    x_q = x_q_2d.view(b, h, s_pad, d)[:, :, :s, :].contiguous()
    x_sf = x_sf_1d.view(b, h, s_pad, cols)
    return x_q, x_sf


def quantize_mxfp8_v(
    x_bhsd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize cuDNN MXFP8 SDPA V tensors.

    cuDNN scales V along the sequence dimension because BMM2 contracts over S.

    Returns:
        x_q: [B, H, S, D] float8_e4m3fn.
        x_sf: [B, H, S_scale_padded, D_padded] uint8 E8M0 scale factors in
            cuDNN F8_128x4 layout, with the S-scale dimension contiguous.
    """
    if x_bhsd.dim() != 4:
        raise ValueError(f"quantize_mxfp8_v expects (B, H, S, D); got shape {x_bhsd.shape}")
    b, h, s, d = x_bhsd.shape
    s_pad = _pad_up(s, 128)
    d_pad = _pad_up(d, 128)
    x_bhds = x_bhsd.transpose(2, 3)
    x_padded = x_bhds.new_zeros(b, h, d_pad, s_pad)
    x_padded[:, :, :d, :s] = x_bhds

    x_2d = x_padded.reshape(b * h * d_pad, s_pad)
    x_q_2d, x_sf_1d = torch.ops.trtllm.mxfp8_quantize(x_2d, True, alignment=_MXFP8_BLOCK)
    cols = s_pad // _MXFP8_BLOCK
    x_q = x_q_2d.view(b, h, d_pad, s_pad)[:, :, :d, :s].permute(0, 1, 3, 2).contiguous()
    x_sf = x_sf_1d.view(b, h, d_pad, cols).permute(0, 1, 3, 2)
    return x_q, x_sf
