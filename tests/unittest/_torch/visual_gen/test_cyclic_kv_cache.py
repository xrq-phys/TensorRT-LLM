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
"""CyclicKVCache: the cyclic buffer, and the ranges a causal block attends to."""

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache import CyclicKVCache

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
DTYPE = torch.float16  # integer stamps up to 2048 stay exact

# (pin_tokens, tokens_per_frame, window_frames, chunk_frames): the window start falls inside,
# on and just before the end of the fixed region, and the ring is and is not a whole number
# of frames.
GEOMETRIES = [
    pytest.param(48, 96, 4, 4, id="pin48-frame96"),
    pytest.param(64, 96, 3, 2, id="pin64-frame96"),
    pytest.param(17, 40, 5, 2, id="pin17-frame40"),
    pytest.param(100, 200, 2, 3, id="pin100-frame200"),
    pytest.param(0, 96, 4, 4, id="no-pin"),
    pytest.param(63, 96, 3, 2, id="pin63-frame96"),
]


def make_cache(pin, frame, window_frames, chunk_frames):
    return CyclicKVCache(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        pin_tokens=pin,
        window_tokens=(window_frames - 1) * frame,
        max_staged_tokens=chunk_frames * frame,
        device=torch.device("cpu"),
    )


def stamp(layer, start, count, plane):
    """K (plane 0) or V (plane 1) of tokens ``[start, start + count)``: a function of layer,
    absolute position, head and channel alone."""
    pos = torch.arange(start, start + count) % 512 + 1
    head = torch.arange(1, NUM_KV_HEADS + 1)
    chan = torch.arange(HEAD_DIM)
    x = pos[:, None, None] * head[None, :, None] + chan[None, None, :] + 1000 * (layer + 1)
    return (x if plane == 0 else -x).to(DTYPE)


def stage(cache, count):
    for layer in range(NUM_LAYERS):
        cache.write_staged(
            layer,
            stamp(layer, cache.staging_offset, count, 0),
            stamp(layer, cache.staging_offset, count, 1),
        )


def rollout(cache, num_chunks):
    """Drive a rollout: the fixed region first, then ``num_chunks`` full chunks. Yields
    after every commit."""
    if cache.pin_tokens:
        stage(cache, cache.pin_tokens)
        cache.commit(cache.pin_tokens)
        yield
    for _ in range(num_chunks):
        stage(cache, cache.max_staged_tokens)
        cache.commit(cache.max_staged_tokens)
        yield


def visible_positions(cache, first, end):
    """Positions a block ``[first, end)`` attends to: the fixed region and everything from
    ``window_tokens`` before the block to its end."""
    fixed = cache.pin_tokens
    low = max(first - cache.window_tokens, fixed)
    return list(range(min(fixed, end))) + list(range(low, end))


def block_ranges(cache, num_blocks, block_size, i):
    """``[(start, length), (start, length)]`` of block ``i``, from ``kv_ranges``."""
    return [
        (start[i].item(), length[i].item())
        for start, length in cache.kv_ranges(num_blocks, block_size)
    ]


def block_rows(cache, num_blocks, block_size, i):
    """Rows of block ``i`` read from the two ranges ``kv_ranges`` returns."""
    rows = []
    for start, length in block_ranges(cache, num_blocks, block_size, i):
        rows += list(range(start, start + length))
    return rows


@pytest.mark.parametrize("pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_ranges_cover_exactly_what_a_block_attends_to(pin, frame, window_frames, chunk_frames):
    cache = make_cache(pin, frame, window_frames, chunk_frames)
    chunk = chunk_frames * frame
    for _ in rollout(cache, 30):
        # A denoising pass is one block of the chunk, the clean pass one block per frame.
        for size in (chunk, frame):
            blocks = chunk // size
            for i in range(blocks):
                first = cache.staging_offset + i * size
                want = cache.row(np.array(visible_positions(cache, first, first + size)))
                got = block_rows(cache, blocks, size, i)
                assert sorted(got) == sorted(want.tolist())
                assert len(set(got)) == len(got), "the two ranges overlap"


@pytest.mark.parametrize("pin,frame,window_frames,chunk_frames", GEOMETRIES)
def test_buffer_holds_the_visible_tokens_of_the_next_forward(
    pin, frame, window_frames, chunk_frames
):
    cache = make_cache(pin, frame, window_frames, chunk_frames)
    for _ in rollout(cache, 30):
        history = cache.staging_offset
        positions = visible_positions(cache, history, history)
        rows = cache.row(np.array(positions, dtype=np.int64))
        for layer in range(NUM_LAYERS):
            k, v = cache.kv_buffers(layer)
            for plane, buf in enumerate((k, v)):
                torch.testing.assert_close(
                    buf[torch.from_numpy(rows)],
                    torch.cat([stamp(layer, p, 1, plane) for p in positions]),
                    rtol=0,
                    atol=0,
                )


def test_a_block_that_wraps_the_ring_attends_to_two_ranges():
    cache = make_cache(48, 96, 4, 4)
    wrapped = 0
    for _ in rollout(cache, 20):
        for i in range(4):
            (start, length), (second_start, second_length) = block_ranges(cache, 4, 96, i)
            assert start == 0 and length > 0
            if second_length:
                # The second range is the rest of the ring, which the first one skips.
                assert second_start > length
                assert second_start + second_length <= cache.capacity
                wrapped += 1
    assert wrapped, "the rollout never wrapped"


def test_a_window_the_size_of_the_ring_is_the_whole_buffer():
    """In steady state a denoising block attends to everything the buffer holds."""
    cache = make_cache(48, 96, 4, 4)
    for _ in rollout(cache, 6):
        pass
    (start, length), (_, second_length) = block_ranges(cache, 1, 384, 0)
    assert (start, length, second_length) == (0, cache.capacity, 0)


def test_restaging_overwrites_the_same_rows():
    """A denoising step stages the chunk again over the previous step's tokens."""
    cache = make_cache(48, 96, 4, 4)
    for _ in rollout(cache, 3):
        pass
    count = cache.max_staged_tokens
    junk = torch.full((count, NUM_KV_HEADS, HEAD_DIM), 7.0, dtype=DTYPE)
    for layer in range(NUM_LAYERS):
        cache.write_staged(layer, junk, -junk)
    stage(cache, count)
    rows = torch.from_numpy(cache.row(cache.staging_offset + np.arange(count)))
    for layer in range(NUM_LAYERS):
        k, v = cache.kv_buffers(layer)
        torch.testing.assert_close(k[rows], stamp(layer, cache.staging_offset, count, 0))
        torch.testing.assert_close(v[rows], stamp(layer, cache.staging_offset, count, 1))


def test_fewer_tokens_than_a_full_chunk():
    cache = make_cache(48, 96, 4, 4)
    for _ in rollout(cache, 2):
        pass
    stage(cache, 100)
    cache.commit(100)
    assert cache.staging_offset == 48 + 2 * cache.max_staged_tokens + 100
    first = cache.staging_offset
    assert sorted(block_rows(cache, 1, 96, 0)) == sorted(
        cache.row(np.array(visible_positions(cache, first, first + 96))).tolist()
    )


def test_the_fixed_region_may_exceed_one_chunk_and_be_staged_with_the_first_chunk():
    cache = make_cache(700, 96, 3, 2)
    assert cache.staging_capacity == 700 + 2 * 96
    stage(cache, 700)
    cache.commit(700)
    assert cache.fixed_tokens == 700
    assert cache.staging_capacity == 2 * 96
    # The first forward may stage the fixed region and the first chunk together.
    cache.reset()
    stage(cache, 700 + 2 * 96)
    cache.commit(700 + 2 * 96)
    assert cache.staging_offset == 700 + 2 * 96


def test_the_fixed_region_is_its_own_block_before_it_is_committed():
    cache = make_cache(48, 96, 4, 4)
    (start, length), (_, second_length) = block_ranges(cache, 1, 48, 0)
    assert (start, length, second_length) == (0, 48, 0)


def test_reset_starts_a_new_rollout():
    cache = make_cache(48, 96, 4, 4)
    for _ in rollout(cache, 4):
        pass
    cache.reset()
    assert cache.staging_offset == 0 and cache.fixed_tokens == 0
    for _ in rollout(cache, 4):
        first = cache.staging_offset
        got = sorted(block_rows(cache, 1, 384, 0))
        want = sorted(cache.row(np.array(visible_positions(cache, first, first + 384))).tolist())
        assert got == want


def test_ranges_are_persistent_and_follow_commits():
    cache = make_cache(48, 96, 4, 4)
    (start, length), _ = cache.kv_ranges(4, 96)
    address = start.data_ptr()
    before = (start.tolist(), length.tolist())
    for _ in rollout(cache, 3):
        pass
    assert cache.kv_ranges(4, 96)[0][0].data_ptr() == address
    assert (start.tolist(), length.tolist()) != before


def test_rejects_bad_calls():
    cache = make_cache(48, 96, 4, 4)
    with pytest.raises(ValueError, match="stages 1"):
        cache.commit(0)
    with pytest.raises(ValueError, match="stages 1"):
        cache.commit(cache.staging_capacity + 1)
    k = torch.zeros(1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE)
    with pytest.raises(TypeError, match="dtype"):
        cache.write_staged(0, k.float(), k.float())
    with pytest.raises(ValueError, match="does not fit"):
        cache.write_staged(0, k, k.expand(2, -1, -1))
    too_many = k.expand(cache.staging_capacity + 1, -1, -1)
    with pytest.raises(ValueError, match="stages 1"):
        cache.write_staged(0, too_many, too_many)
    with pytest.raises(ValueError, match="does not fit"):
        cache.write_staged(0, k[:, :1], k[:, :1])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(window_tokens=-1), "non-negative"),
        (dict(max_staged_tokens=0), "positive"),
        (dict(dtype=torch.float32), "bf16/fp16"),
    ],
)
def test_rejects_bad_geometry(kwargs, match):
    args = dict(
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        dtype=DTYPE,
        pin_tokens=48,
        window_tokens=96,
        max_staged_tokens=96,
        device=torch.device("cpu"),
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        CyclicKVCache(**args)


def test_sizes():
    assert (
        CyclicKVCache.capacity_tokens(pin_tokens=48, window_tokens=288, max_staged_tokens=384)
        == 720
    )
    assert (
        CyclicKVCache.pool_bytes_for(
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=DTYPE,
            pin_tokens=48,
            window_tokens=288,
            max_staged_tokens=384,
        )
        == 2 * 2 * 720 * 2 * 8 * 2
    )


def test_causal_blocks():
    assert CyclicKVCache.causal_blocks(384, None) == (384, 1)
    assert CyclicKVCache.causal_blocks(384, 96) == (96, 4)
    with pytest.raises(ValueError, match="do not split"):
        CyclicKVCache.causal_blocks(384, 100)
