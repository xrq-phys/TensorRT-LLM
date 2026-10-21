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
"""CyclicKVAttention: a causal video rollout, a denoising pass and a clean pass per chunk,
against a dense reference that knows nothing about the cache."""

import pytest
import torch
from utils.cyclic_kv_reference import ReferenceWindowAttention, dense_window_reference

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention.backends.cudnn import CuDNNAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cute_dsl import CuTeDSLAttention
from tensorrt_llm._torch.visual_gen.attention.backends.cyclic_kv import CyclicKVAttention
from tensorrt_llm._torch.visual_gen.attention.backends.flash_attn4 import FlashAttn4Attention
from tensorrt_llm._torch.visual_gen.attention.backends.flashinfer import FlashInferAttention
from tensorrt_llm._torch.visual_gen.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.visual_gen.cache import CyclicKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_LAYERS = 2
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 64
DTYPE = torch.bfloat16

# (pin_tokens, tokens_per_frame, window_frames, chunk_frames)
GEOMETRIES = [
    pytest.param(48, 96, 4, 4, id="pin48-frame96"),
    pytest.param(17, 40, 3, 2, id="pin17-frame40"),
    pytest.param(100, 200, 2, 3, id="pin100-frame200"),
    pytest.param(63, 96, 3, 2, id="pin63-frame96"),
]


_NEEDS_BLACKWELL = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs Blackwell",
)
# Backends without a K/V window, which the wrapper calls once per block and range.
PER_BLOCK_BACKENDS = {
    "fa4": FlashAttn4Attention,
    "flashinfer": FlashInferAttention,
    "cudnn": CuDNNAttention,
}


@pytest.fixture(
    params=[
        "reference",
        pytest.param("kernel", marks=_NEEDS_BLACKWELL),
        *(pytest.param(name, marks=_NEEDS_BLACKWELL) for name in PER_BLOCK_BACKENDS),
    ]
)
def inner_backend(request):
    """The backend the wrapper calls: a torch reference, which checks the wrapper and the
    cache alone; the CuTe DSL kernel, which attends to a K/V window; and backends that do
    not, which the wrapper calls per block."""

    def make(layer_idx):
        if request.param == "reference":
            return ReferenceWindowAttention(HEAD_DIM)
        cls = PER_BLOCK_BACKENDS.get(request.param, CuTeDSLAttention)
        return cls(
            layer_idx=layer_idx,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=NUM_KV_HEADS,
            dtype=DTYPE,
        )

    return make


def make_cache(pin, frame, window_frames, chunk_frames):
    return CyclicKVCache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        pin_tokens=pin,
        window_tokens=(window_frames - 1) * frame,
        max_staged_tokens=chunk_frames * frame,
    )


def rand(*shape):
    return torch.randn(*shape, dtype=DTYPE, device="cuda")


class Rollout:
    """Runs forwards over a cache and the dense reference side by side."""

    def __init__(self, cache, make_inner):
        self.cache = cache
        self.attentions = [CyclicKVAttention(make_inner(i), layer_idx=i) for i in range(NUM_LAYERS)]
        # Every committed token's K/V per layer, by logical position.
        empty = torch.empty(0, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
        self.keys = [empty] * NUM_LAYERS
        self.values = [empty] * NUM_LAYERS

    def forward(self, num_tokens, block_size, num_rows=None):
        """One forward of ``num_tokens`` new tokens in every layer; checks each layer's output
        against the reference and returns the K/V it staged."""
        cache = self.cache
        rows = num_tokens if num_rows is None else num_rows
        staged = []
        for layer, attention in enumerate(self.attentions):
            q = rand(1, rows, NUM_HEADS, HEAD_DIM)
            k, v = rand(1, rows, NUM_KV_HEADS, HEAD_DIM), rand(1, rows, NUM_KV_HEADS, HEAD_DIM)
            out = attention.forward(
                q,
                k,
                v,
                kv_cache=cache,
                seq_len=num_tokens,
                causal_block_size=block_size,
                batch_size=1,
                seq_len_kv=rows,
            )
            assert out.shape == (1, rows, NUM_HEADS, HEAD_DIM)
            want = dense_window_reference(
                q[0, :num_tokens],
                torch.cat([self.keys[layer], k[0, :num_tokens]]),
                torch.cat([self.values[layer], v[0, :num_tokens]]),
                start=cache.staging_offset,
                fixed=cache.pin_tokens,
                window=cache.window_tokens,
                block_size=block_size or num_tokens,
                sm_scale=attention.inner_backend.scale,
            )
            torch.testing.assert_close(out[0, :num_tokens], want, rtol=3e-2, atol=3e-2)
            assert not out[0, num_tokens:].any(), "padding rows must come back zero"
            staged.append((k[0, :num_tokens], v[0, :num_tokens]))
        return staged

    def commit(self, staged):
        for layer, (k, v) in enumerate(staged):
            self.keys[layer] = torch.cat([self.keys[layer], k])
            self.values[layer] = torch.cat([self.values[layer], v])
        self.cache.commit(staged[0][0].shape[0])


@pytest.mark.parametrize("pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_rollout_matches_dense_reference(inner_backend, pin, frame, window_frames, chunk_frames):
    cache = make_cache(pin, frame, window_frames, chunk_frames)
    rollout = Rollout(cache, inner_backend)
    if pin:
        rollout.commit(rollout.forward(pin, None))
    chunk = chunk_frames * frame
    # Long enough that the ring wraps many times.
    for _ in range(12):
        # Denoising steps restage the chunk as one block, the clean pass one frame per block,
        # causal across frames.
        for _ in range(2):
            rollout.forward(chunk, None)
        rollout.commit(rollout.forward(chunk, frame))


def test_padding_rows_are_not_cached_or_attended(inner_backend):
    cache = make_cache(48, 96, 4, 4)
    rollout = Rollout(cache, inner_backend)
    rollout.commit(rollout.forward(48, None, num_rows=64))
    for _ in range(3):
        rollout.commit(rollout.forward(384, 96, num_rows=384 + 16))
    # A later forward still matches the reference, which never saw the padding rows.
    rollout.commit(rollout.forward(384, 96))


def test_a_short_final_chunk(inner_backend):
    cache = make_cache(48, 96, 4, 4)
    rollout = Rollout(cache, inner_backend)
    rollout.commit(rollout.forward(48, None))
    for _ in range(4):
        rollout.commit(rollout.forward(384, 96))
    rollout.commit(rollout.forward(192, 96))


def test_without_a_cache_it_is_the_inner_backend(inner_backend):
    attention = CyclicKVAttention(inner_backend(0), layer_idx=0)
    assert attention.support_lse()
    if not attention.inner_backend.support_kv_window():
        pytest.skip("the passthrough is checked with kv_start/kv_len")
    q = rand(1, 4, NUM_HEADS, HEAD_DIM)
    kv = rand(1, 16, NUM_KV_HEADS, HEAD_DIM)
    start = torch.zeros(1, dtype=torch.int32, device="cuda")
    length = torch.full((1,), 16, dtype=torch.int32, device="cuda")
    want = attention.inner_backend.forward(q, kv, kv, kv_start=start, kv_len=length)
    torch.testing.assert_close(attention.forward(q, kv, kv, kv_start=start, kv_len=length), want)
    with pytest.raises(NotImplementedError, match="only implemented over a K/V cache"):
        attention.forward(q, kv, kv, causal_block_size=2)


def test_rejects_unsupported_calls(inner_backend):
    cache = make_cache(48, 96, 4, 4)
    attention = CyclicKVAttention(inner_backend(0), layer_idx=0)
    q, k, v = rand(1, 96, NUM_HEADS, HEAD_DIM), rand(1, 96, 2, HEAD_DIM), rand(1, 96, 2, HEAD_DIM)
    with pytest.raises(NotImplementedError, match="full attention"):
        attention.forward(q, k, v, kv_cache=cache, attention_mask=PredefinedAttentionMask.CAUSAL)
    with pytest.raises(ValueError, match="one video"):
        attention.forward(rand(2, 96, NUM_HEADS, HEAD_DIM), k, v, kv_cache=cache)
    with pytest.raises(ValueError, match="heads"):
        attention.forward(q, rand(1, 96, 4, HEAD_DIM), rand(1, 96, 4, HEAD_DIM), kv_cache=cache)
    with pytest.raises(ValueError, match="seq_len"):
        attention.forward(q, k, v, kv_cache=cache, seq_len=97)
    with pytest.raises(ValueError, match="do not split"):
        attention.forward(q, k, v, kv_cache=cache, causal_block_size=40)
    with pytest.raises(NotImplementedError, match="forward_with_lse"):
        attention.forward_with_lse(q, k, v, kv_cache=cache)


def test_needs_a_backend_with_lse():
    with pytest.raises(ValueError, match="does not return the LSE"):
        CyclicKVAttention(VanillaAttention(num_heads=NUM_HEADS, head_dim=HEAD_DIM), layer_idx=0)
