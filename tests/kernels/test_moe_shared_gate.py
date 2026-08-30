import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid


def test_shared_gate_cpu_matches_reference_equations():
    torch.manual_seed(23)
    hidden = torch.randn(5, 8)
    weight = torch.randn(8)
    routed = torch.randn(5, 8)
    shared = torch.randn(5, 8)

    gate = shared_gate_sigmoid(hidden, weight)
    expected_gate = torch.sigmoid(torch.sum(hidden.float() * weight.float(), dim=-1))
    torch.testing.assert_close(gate, expected_gate)
    torch.testing.assert_close(
        shared_gate_mul_add(routed, shared, gate),
        routed + expected_gate[:, None].to(shared.dtype) * shared,
    )
