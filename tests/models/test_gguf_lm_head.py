"""H0 regression for the untied GGUF LM-head prefill contract.

The semantic reference is FreeToken PR 131 commit ``b2f84751826cb380156fad4fd36e613bfb454625``.
This test exercises the downstream implementation and copies no donor source.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# ruff: noqa: E402
torch = pytest.importorskip("torch")

from freetoken.gguf_types import GGML_F32
from freetoken.layers.gguf import GGUFLMHead

HIDDEN = 3
VOCAB = 5
WEIGHT = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
        [0.1, 0.1, 0.1],
    ],
    dtype=torch.float32,
)
BIAS = torch.zeros(VOCAB, dtype=torch.float32)


def _head() -> GGUFLMHead:
    head = GGUFLMHead(HIDDEN, VOCAB, GGML_F32, has_bias=True)
    head.qweight.view(torch.float32).copy_(WEIGHT)
    head.bias.copy_(BIAS)
    return head


def _batch(phase: str, last_indices: list[int]):
    indices = torch.tensor(last_indices, dtype=torch.int64)
    metadata = SimpleNamespace(get_last_indices=lambda bs: indices[:bs])
    return SimpleNamespace(
        is_prefill=phase == "prefill",
        size=len(last_indices),
        attn_metadata=metadata,
    )


def _forward(head, batch, hidden, monkeypatch):
    import freetoken.core as core

    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(batch=batch))
    return head.forward(hidden)


def test_prefill_projects_only_each_request_last_position(monkeypatch):
    """The first sampled token must use one final hidden row per prompt."""
    hidden = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [-1.0, -1.0, -1.0]],
    )
    logits = _forward(_head(), _batch("prefill", [1, 3]), hidden, monkeypatch)

    expected = hidden[[1, 3]] @ WEIGHT.T + BIAS
    assert logits.shape == (2, VOCAB)
    torch.testing.assert_close(logits, expected)
    assert torch.equal(torch.argmax(logits, dim=-1), torch.tensor([1, 3]))


def test_decode_projects_every_input_row(monkeypatch):
    hidden = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    logits = _forward(_head(), _batch("decode", [0, 1]), hidden, monkeypatch)

    expected = hidden @ WEIGHT.T + BIAS
    assert logits.shape == (2, VOCAB)
    torch.testing.assert_close(logits, expected)


def test_empty_prefill_returns_empty_vocab_logits(monkeypatch):
    hidden = torch.empty((0, HIDDEN), dtype=torch.float32)
    logits = _forward(_head(), _batch("prefill", []), hidden, monkeypatch)

    assert logits.shape == (0, VOCAB)


def test_prefill_rejects_out_of_range_last_position(monkeypatch):
    hidden = torch.zeros((2, HIDDEN), dtype=torch.float32)

    with pytest.raises(IndexError, match="out of bounds"):
        _forward(_head(), _batch("prefill", [2]), hidden, monkeypatch)


def test_qwen4_gguf_converter_attaches_the_specialized_head(monkeypatch):
    from freetoken.models.qwen4_exp import gguf as qwen4_gguf

    class _Linear:
        def __init__(self):
            self.weight = torch.empty((2, 2))
            self.bias = None

    model = SimpleNamespace(
        model=SimpleNamespace(
            hyper_connection_mixer=SimpleNamespace(
                input_mix_weight_down=_Linear(),
                input_mix_weight_up=_Linear(),
            ),
        ),
    )
    config = SimpleNamespace(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        linear_attention_group=lambda: None,
    )
    quant_types = {
        (-1, "token_embd.weight"): GGML_F32,
        (-1, "output.weight"): GGML_F32,
        (-1, "output_hc_down.weight"): GGML_F32,
        (-1, "output_hc_up.weight"): GGML_F32,
    }
    monkeypatch.setattr(qwen4_gguf, "_require_tp1", lambda what: None)
    monkeypatch.setattr(qwen4_gguf, "_quant_map", lambda path: quant_types)

    with pytest.raises(ValueError, match="no linear-attention group"):
        qwen4_gguf.convert_qwen4_to_gguf(model, config, model_path="unused")

    assert isinstance(model.lm_head, GGUFLMHead)
