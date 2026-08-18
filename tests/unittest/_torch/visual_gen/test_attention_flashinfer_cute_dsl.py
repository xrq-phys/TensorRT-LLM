# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the FlashInfer CuTe DSL VisualGen attention adapter."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.visual_gen.attention_backend import (
    FlashInferCuTeDSLAttention,
    get_visual_gen_attention_backend,
)
from tensorrt_llm._torch.visual_gen.attention_backend import flashinfer_cute_dsl as backend_module


def test_factory_resolves_flashinfer_cute_dsl_backend():
    assert get_visual_gen_attention_backend("FLASHINFER_CUTEDSL") is FlashInferCuTeDSLAttention


def test_forward_builds_uniform_ragged_metadata(monkeypatch):
    calls = []

    def fake_fmha(**kwargs):
        calls.append(kwargs)
        kwargs["o"].fill_(1)
        kwargs["lse"].fill_(2)

    monkeypatch.setattr(backend_module, "cute_dsl_fmha_ragged_prefill", fake_fmha)
    attention = FlashInferCuTeDSLAttention(num_heads=4, num_kv_heads=2, head_dim=8)
    q = torch.randn(2, 3, 4, 8)
    k = torch.randn(2, 5, 2, 8)
    v = torch.randn(2, 5, 2, 8)

    output, lse = attention.forward_with_lse(q, k, v)

    assert output.shape == (2, 3, 4, 8)
    assert lse.shape == (2, 4, 3)
    assert torch.all(output == 1)
    assert torch.all(lse == 2)
    assert calls[0]["qo_indptr"].tolist() == [0, 3, 6]
    assert calls[0]["kv_indptr"].tolist() == [0, 5, 10]
    assert calls[0]["max_qo_len"] == 3
    assert calls[0]["max_kv_len"] == 5
    assert calls[0]["is_causal"] is False

    attention(q, k, v, attention_mask=PredefinedAttentionMask.CAUSAL)
    assert calls[1]["qo_indptr"] is calls[0]["qo_indptr"]
    assert calls[1]["kv_indptr"] is calls[0]["kv_indptr"]
    assert calls[1]["is_causal"] is True


def test_key_padding_mask_is_rejected():
    attention = FlashInferCuTeDSLAttention(num_heads=4, head_dim=8)
    qkv = torch.randn(2, 3, 4, 8)

    with pytest.raises(NotImplementedError, match="padded dense batches"):
        attention(qkv, qkv, qkv, key_padding_mask=torch.ones(2, 3, dtype=torch.bool))
