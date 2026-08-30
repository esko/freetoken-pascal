import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import (
    GemmaPlusOneRMSNorm,
    VocabParallelEmbedding,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
    swigluoai_and_mul,
)
from freetoken.layers.rotary import RotaryEmbedding


@pytest.fixture(autouse=True)
def _single_rank():
    info = try_get_tp_info()
    if info is None:
        set_tp_info(rank=0, size=1)
    elif info.size != 1:
        pytest.skip("CPU reference seam tests require tensor parallel size 1")


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (silu_and_mul, lambda x: F.silu(x[..., :4]) * x[..., 4:]),
        (gelu_and_mul, lambda x: F.gelu(x[..., :4]) * x[..., 4:]),
        (gelu_tanh_and_mul, lambda x: F.gelu(x[..., :4], approximate="tanh") * x[..., 4:]),
        (
            swigluoai_and_mul,
            lambda x: torch.clamp(x[..., :4], max=1.5)
            * torch.sigmoid(1.2 * x[..., :4])
            * (torch.clamp(x[..., 4:], -1.5, 1.5) + 1),
        ),
    ],
)
def test_cpu_activation_seams_match_reference(operation, expected):
    torch.manual_seed(31)
    values = torch.randn(3, 8)
    if operation is swigluoai_and_mul:
        actual = operation(values, alpha=1.2, limit=1.5)
    else:
        actual = operation(values)
    torch.testing.assert_close(actual, expected(values))


def test_cpu_activation_seam_honors_output_buffer():
    values = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    output = torch.empty(1, 4)
    result = silu_and_mul(values, out=output)
    assert result is output
    torch.testing.assert_close(result, F.silu(values[..., :4]) * values[..., 4:])


def test_cpu_embedding_seam_matches_torch_embedding():
    embedding = VocabParallelEmbedding(7, 3, embed_scale=2.0)
    embedding.weight.copy_(torch.arange(21, dtype=torch.float32).view(7, 3))
    ids = torch.tensor([6, 0, 4], dtype=torch.int64)
    torch.testing.assert_close(embedding.forward(ids), F.embedding(ids, embedding.weight) * 2.0)


def test_cpu_gemma_plus_one_norm_matches_fp32_equation():
    norm = GemmaPlusOneRMSNorm(4, eps=1e-6)
    norm.weight.copy_(torch.tensor([0.0, 0.1, -0.2, 0.3]))
    values = torch.tensor([[1.0, -2.0, 0.5, 4.0], [0.0, 1.0, 2.0, -3.0]])
    variance = values.square().mean(dim=-1, keepdim=True)
    expected = values * torch.rsqrt(variance + 1e-6) * (1 + norm.weight)
    torch.testing.assert_close(norm.forward(values), expected)


def test_cpu_rope_seam_is_deterministic_and_preserves_tail_dimensions():
    rope = RotaryEmbedding(8, 4, 16, 100.0)
    query = torch.arange(16, dtype=torch.float32).view(2, 8)
    key = torch.arange(8, dtype=torch.float32).view(1, 8).repeat(2, 1)
    original_tail = query[:, 4:].clone()
    first_q, first_k = rope.forward(torch.tensor([0, 1]), query.clone(), key.clone())
    second_q, second_k = rope.forward(torch.tensor([0, 1]), query.clone(), key.clone())
    torch.testing.assert_close(first_q, second_q)
    torch.testing.assert_close(first_k, second_k)
    torch.testing.assert_close(first_q[:, 4:], original_tail)


def test_cpu_rope_rejects_odd_head_geometry():
    with pytest.raises(ValueError, match="positive even"):
        RotaryEmbedding(7, 6, 16, 100.0)
