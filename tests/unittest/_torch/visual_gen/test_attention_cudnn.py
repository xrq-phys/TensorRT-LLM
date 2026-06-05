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
"""Tests for CuDNNAttention; requires Blackwell GPU + nvidia-cudnn-frontend."""

import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.attention_backend.cudnn import (
    CuDNNAttention,
    _cudnn_import_error,
)
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

_QUANT_CFG = QuantAttentionConfig(
    qk_dtype="fp8",
    v_dtype="fp8",
    q_block_size=1,
    k_block_size=1,
    v_block_size=1,
    qk_sf_vec=32,
    v_sf_vec=32,
)


def _require_setup():
    if _cudnn_import_error is not None:
        pytest.skip(f"nvidia-cudnn-frontend not installed: {_cudnn_import_error}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    cap = torch.cuda.get_device_capability()
    if (cap[0], cap[1]) < (10, 0):
        pytest.skip(f"Blackwell GPU required (sm100/sm103); got sm_{cap[0]}{cap[1]}")


@pytest.mark.parametrize(
    "b,h,s,d,is_causal",
    [
        (1, 8, 128, 128, False),
        (1, 8, 256, 128, True),
        (2, 4, 512, 128, False),
    ],
)
def test_output_shape(b, h, s, d, is_causal):
    _require_setup()
    backend = CuDNNAttention(num_heads=h, head_dim=d, quant_attention_config=_QUANT_CFG)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    out = backend.forward(q, k, v, is_causal=is_causal)
    assert out.shape == (b, h, s, d)


def test_forward_with_lse_shapes():
    _require_setup()
    b, h, s, d = 1, 8, 256, 128
    backend = CuDNNAttention(num_heads=h, head_dim=d, quant_attention_config=_QUANT_CFG)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda")
    out, lse = backend.forward_with_lse(q, k, v, is_causal=False)
    assert out.shape == (b, h, s, d)
    assert lse.shape == (b, s, h)
    assert lse.dtype == torch.float32


def test_numerical_accuracy_vs_sdpa():
    _require_setup()
    torch.manual_seed(42)
    b, h, s, d = 1, 8, 256, 128
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda") * 0.1
    v = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda") * 0.1

    sm_scale = 1.0 / math.sqrt(d)
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=sm_scale)
    ref = ref.to(torch.bfloat16)

    backend = CuDNNAttention(num_heads=h, head_dim=d, quant_attention_config=_QUANT_CFG)
    out = backend.forward(q, k, v, is_causal=False)

    max_abs_err = (out.float() - ref.float()).abs().max().item()
    ref_max = ref.float().abs().max().item()
    msg = f"max_abs_err={max_abs_err:.4f}, ref_max={ref_max:.4f}"
    assert max_abs_err < 0.05 * ref_max + 1e-3, msg
    print(msg)
