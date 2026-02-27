import math
import random

import numpy as np
import pytest
import torch
from torch import nn

from src.common.functions import (
    compute_log_probs,
    create_completion_mask,
    extract_answer_from_model_output,
    selective_log_softmax,
    set_random_seed,
)


class DummyOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 7):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, dtype=torch.float32)
        for b in range(batch_size):
            for t in range(seq_len):
                base = float(input_ids[b, t].item())
                logits[b, t] = torch.arange(self.vocab_size, dtype=torch.float32) * 0.25 + base
        return DummyOutput(logits=logits)


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "<reason>test</reason>\n<answer>[1,2,3]</answer>",
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        (
            "<ANSWER>[ 1.5 , 2.5 , 3.5 ]</ANSWER>",
            np.array([1.5, 2.5, 3.5], dtype=np.float32),
        ),
        (
            "prefix <answer>[1;2;3]</answer> suffix",
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        (
            "<answer>[1,2,3]</answer>",
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ),
        (
            "<answer>[1, 2, 3, 4, 5]</answer>",
            np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        ),
        (
            "<answer>[0]</answer>",
            np.array([0.0], dtype=np.float32),
        ),
    ],
)
def test_extract_answer_valid(text, expected):
    actual = extract_answer_from_model_output(text)
    assert actual.dtype == np.float32
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no tags at all",
        "<answer>no list</answer>",
        "<answer>[]</answer>",
        "<answer>[a,b,c]</answer>",
        "<reason>only reason</reason>",
        "<answer>[1,2,3]",
        "[1,2,3]</answer>",
        "<answer>[1,2,3]</ans>",
        "<answer>[1,2,three]</answer>",
    ],
)
def test_extract_answer_invalid(text):
    actual = extract_answer_from_model_output(text)
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float32
    assert actual.size == 0


def test_set_random_seed_reproducible():
    set_random_seed(2026)
    py_first = [random.random() for _ in range(3)]
    np_first = np.random.rand(3).astype(np.float32)
    torch_first = torch.rand(3, dtype=torch.float32)

    set_random_seed(2026)
    py_second = [random.random() for _ in range(3)]
    np_second = np.random.rand(3).astype(np.float32)
    torch_second = torch.rand(3, dtype=torch.float32)

    assert py_first == py_second
    np.testing.assert_allclose(np_first, np_second, rtol=0, atol=0)
    assert torch.allclose(torch_first, torch_second)


def test_selective_log_softmax_shape_and_values():
    logits = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
            [[-1.0, 0.0, 1.0], [2.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    input_ids = torch.tensor([[2, 1], [0, 2]], dtype=torch.long)

    out = selective_log_softmax(logits, input_ids)
    assert out.shape == (2, 2)

    expected = torch.empty(2, 2, dtype=torch.float32)
    for b in range(2):
        for t in range(2):
            row = logits[b, t]
            expected[b, t] = torch.log_softmax(row, dim=-1)[input_ids[b, t]]
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [3, 5, 8])
@pytest.mark.parametrize("vocab_size", [4, 9])
def test_selective_log_softmax_randomized(batch_size, seq_len, vocab_size):
    g = torch.Generator().manual_seed(100 + batch_size * 10 + seq_len + vocab_size)
    logits = torch.randn(batch_size, seq_len, vocab_size, generator=g, dtype=torch.float32)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g, dtype=torch.long)

    out = selective_log_softmax(logits, input_ids)
    manual = []
    for b in range(batch_size):
        row_vals = []
        for t in range(seq_len):
            row_vals.append(torch.log_softmax(logits[b, t], dim=-1)[input_ids[b, t]].item())
        manual.append(row_vals)
    manual = torch.tensor(manual, dtype=torch.float32)

    assert out.shape == (batch_size, seq_len)
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-6)


def test_compute_log_probs_matches_manual_slicing():
    model = DummyModel(vocab_size=7)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    logits_to_keep = 3
    out = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)

    full_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    sliced_ids = input_ids[:, -logits_to_keep:]
    sliced_logits = full_logits[:, -logits_to_keep:, :]
    expected = selective_log_softmax(sliced_logits, sliced_ids)

    assert out.shape == (2, logits_to_keep)
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("logits_to_keep", [1, 2, 4])
def test_compute_log_probs_various_lengths(logits_to_keep):
    model = DummyModel(vocab_size=6)
    seq_len = 6
    input_ids = torch.tensor(
        [
            [1, 1, 2, 3, 5, 0],
            [2, 3, 1, 1, 4, 2],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    out = compute_log_probs(model, input_ids, attention_mask, logits_to_keep)
    assert out.shape == (2, logits_to_keep)
    assert torch.isfinite(out).all()

    manual = []
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    for b in range(input_ids.shape[0]):
        vals = []
        for t in range(seq_len - logits_to_keep, seq_len):
            token = input_ids[b, t]
            vals.append(torch.log_softmax(logits[b, t - 1], dim=-1)[token].item())
        manual.append(vals)
    manual = torch.tensor(manual, dtype=torch.float32)
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "completion_ids,eos_token_id,expected",
    [
        (
            torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            3,
            torch.tensor([[1, 1, 1, 0]], dtype=torch.int32),
        ),
        (
            torch.tensor([[3, 2, 2, 2]], dtype=torch.long),
            3,
            torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
        ),
        (
            torch.tensor([[1, 2, 4, 5]], dtype=torch.long),
            3,
            torch.tensor([[1, 1, 1, 1]], dtype=torch.int32),
        ),
        (
            torch.tensor([[3, 3, 3, 3]], dtype=torch.long),
            3,
            torch.tensor([[1, 0, 0, 0]], dtype=torch.int32),
        ),
    ],
)
def test_create_completion_mask_basic_cases(completion_ids, eos_token_id, expected):
    mask = create_completion_mask(completion_ids, eos_token_id)
    assert mask.shape == completion_ids.shape
    assert torch.equal(mask.cpu().to(torch.int32), expected)


def test_create_completion_mask_batch_behavior():
    completion_ids = torch.tensor(
        [
            [1, 2, 9, 4, 5],  # first eos at index 2
            [1, 2, 3, 4, 5],  # no eos
            [9, 2, 3, 4, 5],  # eos at index 0
            [1, 9, 9, 4, 5],  # eos at index 1
        ],
        dtype=torch.long,
    )
    eos = 9
    mask = create_completion_mask(completion_ids, eos)
    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(mask.cpu().to(torch.int32), expected)


def test_create_completion_mask_large_random():
    g = torch.Generator().manual_seed(42)
    batch_size = 16
    seq_len = 32
    eos = 11

    completion_ids = torch.randint(0, 20, (batch_size, seq_len), generator=g, dtype=torch.long)
    mask = create_completion_mask(completion_ids, eos)

    assert mask.shape == (batch_size, seq_len)
    assert mask.dtype in (torch.int32, torch.int64)

    for b in range(batch_size):
        row = completion_ids[b].tolist()
        if eos in row:
            first_idx = row.index(eos)
            assert mask[b, : first_idx + 1].sum().item() == first_idx + 1
            assert mask[b, first_idx + 1 :].sum().item() == 0
        else:
            assert mask[b].sum().item() == seq_len


def test_log_prob_numerical_stability_with_large_logits():
    logits = torch.tensor([[[1000.0, 1001.0, 1002.0]]], dtype=torch.float32)
    input_ids = torch.tensor([[2]], dtype=torch.long)
    out = selective_log_softmax(logits, input_ids)
    assert out.shape == (1, 1)
    assert torch.isfinite(out).all()
    assert math.isclose(out.item(), torch.log_softmax(logits[0, 0], dim=-1)[2].item(), rel_tol=1e-6, abs_tol=1e-6)
