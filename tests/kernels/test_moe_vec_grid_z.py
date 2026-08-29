"""Fused GGUF MoE past CUDA's gridDim.z ceiling.

``moe_vec.cuh`` carries the routed-pair index (token * top_k + slot) in ``gridDim.z``,
which CUDA caps at 65535 on every architecture.  The launcher now issues token-aligned
chunks.  The equality assertion catches a wrong per-chunk pointer offset, which a launch
success check alone would miss.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOP_K = 8
SLOTS, H, INTERMEDIATE = 16, 512, 256
CEIL = 65535 // TOP_K


def _pack_q4_0(S: int, OUT: int, K: int, dev) -> torch.Tensor:
    """Pack finite Q4_0 rows so failures are attributable to launch geometry."""
    nb = K // 32
    nib = torch.randint(0, 256, (S, OUT, nb, 16), dtype=torch.uint8)
    scale = (0.02 + 0.03 * torch.rand(S, OUT, nb)).to(torch.float16)
    sb = scale.view(torch.uint8).reshape(S, OUT, nb, 2)
    return torch.cat([sb, nib], dim=-1).reshape(S, OUT, nb * 18).contiguous().to(dev)


@pytest.fixture(scope="module")
def banks():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    max_t = 2 * CEIL + 16
    return {
        "gate_up": _pack_q4_0(SLOTS, 2 * INTERMEDIATE, H, dev),
        "down": _pack_q4_0(SLOTS, H, INTERMEDIATE, dev),
        "ids": torch.randint(0, SLOTS, (max_t, TOP_K), dtype=torch.int32, device=dev),
        "w": torch.rand(max_t, TOP_K, device=dev, dtype=torch.float32),
        "x": (torch.randn(max_t, H, device=dev, dtype=torch.bfloat16) * 0.5).contiguous(),
    }


def _run(b, n: int, start: int = 0) -> torch.Tensor:
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    return fused_experts_gguf_q4_0(
        b["x"][start : start + n].contiguous(),
        b["gate_up"],
        b["down"],
        b["w"][start : start + n].contiguous(),
        b["ids"][start : start + n].contiguous(),
        "silu",
    )


@pytest.mark.parametrize("n", [CEIL - 1, CEIL, CEIL + 1, 2 * CEIL + 7])
def test_moe_vec_launches_past_grid_z_ceiling(banks, n):
    out = _run(banks, n)
    torch.cuda.synchronize()
    assert out.shape == (n, H)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("n", [CEIL - 1, CEIL, CEIL + 1, 2 * CEIL + 7])
def test_moe_vec_chunking_preserves_rows(banks, n):
    got = _run(banks, n)
    torch.cuda.synchronize()
    window = 5
    boundaries = (CEIL, 2 * CEIL)
    for boundary in boundaries:
        if boundary + 2 >= n:
            continue
        start = boundary - 2
        ref = _run(banks, window, start=start)
        torch.cuda.synchronize()
        assert torch.equal(got[start : start + window], ref)


@pytest.mark.parametrize(
    ("top_k", "message"),
    [
        (0, "top_k must be positive"),
        (65536, "top_k must be <= 65535"),
    ],
)
def test_moe_vec_rejects_invalid_top_k(banks, top_k, message):
    from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

    ids = torch.zeros((1, top_k), dtype=torch.int32, device="cuda")
    weights = torch.ones((1, top_k), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match=message):
        fused_experts_gguf_q4_0(
            banks["x"][:1],
            banks["gate_up"],
            banks["down"],
            weights,
            ids,
            "silu",
        )
