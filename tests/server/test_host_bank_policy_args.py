from __future__ import annotations

from unittest.mock import patch

import pytest
from freetoken.server.args import parse_args


class _Config:
    architectures = ["LlamaForCausalLM"]

    def to_dict(self) -> dict:
        return {"architectures": self.architectures, "torch_dtype": "bfloat16"}


def test_host_bank_cli_constructs_explicit_pageable_policy():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _ = parse_args(
            [
                "--model",
                "/models/anon",
                "--host-bank-strategy",
                "pageable",
                "--host-bank-numa-policy",
                "bind",
                "--host-bank-numa-node",
                "1",
            ]
        )

    assert args.host_bank_policy.strategy.value == "pageable"
    assert args.host_bank_policy.numa_policy.value == "bind"
    assert args.host_bank_policy.numa_node == 1


def test_host_bank_cli_requires_finite_limit_for_pinned_policy():
    with pytest.raises(SystemExit):
        parse_args(["--model", "/models/anon", "--host-bank-strategy", "pinned"])


def test_host_bank_cli_constructs_no_swap_guard():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _ = parse_args(
            [
                "--model",
                "/models/anon",
                "--host-bank-strategy",
                "pageable",
                "--host-bank-require-no-swap",
            ]
        )

    assert args.host_bank_policy.require_no_swap is True


def test_host_bank_cli_constructs_opt_in_numa_placement():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        args, _ = parse_args(
            [
                "--model",
                "/models/anon",
                "--host-bank-strategy",
                "pageable",
                "--host-bank-enforce-numa-placement",
                "--host-bank-numa-sample-residency",
            ]
        )

    assert args.host_bank_policy.enforce_numa_placement is True
    assert args.host_bank_policy.sample_numa_residency is True


def test_host_bank_cli_requires_strategy_for_numa_placement():
    with pytest.raises(SystemExit):
        parse_args(["--model", "/models/anon", "--host-bank-enforce-numa-placement"])


def test_host_bank_cli_requires_strategy_for_no_swap_guard():
    with pytest.raises(SystemExit):
        parse_args(["--model", "/models/anon", "--host-bank-require-no-swap"])


def test_host_bank_cli_rejects_policy_options_without_strategy():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "/models/anon",
                "--host-bank-max-pinned-bytes",
                "4096",
            ]
        )


def test_ple_planner_cli_preserves_default_and_accepts_opt_in_adaptive_mode():
    with patch("freetoken.utils.cached_load_hf_config", lambda _path: _Config()):
        default, _ = parse_args(["--model", "/models/anon"])
        adaptive, _ = parse_args(
            [
                "--model",
                "/models/anon",
                "--ple-planner-mode",
                "adaptive",
                "--ple-planner-direct-threshold",
                "3",
            ]
        )

    assert default.ple_planner_mode == "vectorized"
    assert default.ple_planner_direct_threshold == 8
    assert adaptive.ple_planner_mode == "adaptive"
    assert adaptive.ple_planner_direct_threshold == 3


def test_ple_planner_cli_rejects_non_positive_threshold():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                "/models/anon",
                "--ple-planner-direct-threshold",
                "0",
            ]
        )
