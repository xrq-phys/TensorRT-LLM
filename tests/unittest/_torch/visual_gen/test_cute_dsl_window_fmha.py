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
"""Query sequences over free ranges of one shared K/V buffer, against a torch reference."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl.fmha import cute_dsl_fmha_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the CuTe DSL FMHA kernel needs Blackwell",
)


def reference(q, k, v, starts, lens, scale):
    """fp32 output and log-sum-exp of each sequence over its range; empty ranges give 0, -inf."""
    batch, seq_len, heads, _ = q.shape
    rep = heads // k.shape[2]
    out = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full((batch, seq_len, heads), float("-inf"), device=q.device)
    for i, (start, length) in enumerate(zip(starts, lens)):
        if length == 0:
            continue
        keys = k[0, start : start + length].repeat_interleave(rep, dim=1).float()
        values = v[0, start : start + length].repeat_interleave(rep, dim=1).float()
        scores = torch.einsum("shd,nhd->hsn", q[i].float(), keys) * scale
        lse[i] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        out[i] = torch.einsum("hsn,nhd->shd", scores.softmax(dim=-1), values)
    return out, lse


def run_case(starts, lens, seq_len, heads, heads_kv, head_dim, num_rows, dtype=torch.bfloat16):
    batch = len(starts)
    q = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device="cuda")
    k = torch.randn(1, num_rows, heads_kv, head_dim, dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    out = torch.zeros_like(q)
    lse = torch.full((batch, seq_len, heads), float("-inf"), device="cuda")
    scale = head_dim**-0.5
    cute_dsl_fmha_fwd(
        q,
        k,
        v,
        out,
        sm_scale=scale,
        lse=lse,
        kv_start=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        kv_len=torch.tensor(lens, dtype=torch.int32, device="cuda"),
    )
    want_out, want_lse = reference(q, k, v, starts, lens, scale)
    torch.testing.assert_close(out.float(), want_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, want_lse, rtol=1e-3, atol=1e-3)


CASES = [
    pytest.param([0], [384], 96, id="one-range"),
    pytest.param([0, 100, 37], [200, 200, 513], 96, id="free-starts"),
    pytest.param([50, 50], [300, 300], 128, id="shared-range"),
    pytest.param([0, 10], [0, 64], 96, id="empty-range-skipped"),
    pytest.param([5, 700], [1, 300], 64, id="tiny-range-and-end-of-buffer"),
    pytest.param([0, 0, 640, 3], [700, 129, 60, 128], 192, id="mixed"),
]


@pytest.mark.parametrize("starts,lens,seq_len", CASES)
def test_matches_reference(starts, lens, seq_len):
    run_case(starts, lens, seq_len, heads=4, heads_kv=2, head_dim=128, num_rows=1000)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("heads,heads_kv", [(8, 8), (8, 2)])
def test_heads_and_head_dims(head_dim, heads, heads_kv):
    run_case(
        [0, 321], [400, 333], 96, heads=heads, heads_kv=heads_kv, head_dim=head_dim, num_rows=800
    )
