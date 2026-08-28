"""Deterministic host-only expert-cache simulation for routing traces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CacheSimulationResult:
    policy: str
    capacity: int
    requests: int
    hits: int
    misses: int
    evictions: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    def to_dict(self) -> dict[str, int | float | str]:
        return {**asdict(self), "hit_rate": self.hit_rate}


def validate_routing_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict) or trace.get("schema_version") != 1:
        raise ValueError("routing trace must be an object with schema_version 1")
    num_experts = trace.get("num_experts")
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError("num_experts must be a positive integer")
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty array")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{index}] must be an object")
        layer = step.get("layer")
        experts = step.get("experts")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError(f"steps[{index}].layer must be a non-negative integer")
        if not isinstance(experts, list) or not experts:
            raise ValueError(f"steps[{index}].experts must be a non-empty array")
        if len(experts) != len(set(experts)):
            raise ValueError(f"steps[{index}].experts contains duplicates")
        if any(
            not isinstance(expert, int) or isinstance(expert, bool) or not 0 <= expert < num_experts
            for expert in experts
        ):
            raise ValueError(f"steps[{index}].experts contains an out-of-range id")
    return trace


def _requests(trace: dict[str, Any]) -> Iterable[tuple[int, int]]:
    for step in trace["steps"]:
        yield from ((step["layer"], expert) for expert in step["experts"])


def simulate_lru(trace: Any, capacity: int) -> CacheSimulationResult:
    validated = validate_routing_trace(trace)
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0:
        raise ValueError("capacity must be a non-negative integer")

    cache: dict[tuple[int, int], int] = {}
    hits = misses = evictions = 0
    for tick, key in enumerate(_requests(validated)):
        if key in cache:
            hits += 1
            cache[key] = tick
            continue
        misses += 1
        if capacity == 0:
            continue
        if len(cache) == capacity:
            victim = min(cache, key=lambda candidate: (cache[candidate], candidate))
            del cache[victim]
            evictions += 1
        cache[key] = tick

    return CacheSimulationResult(
        policy="lru",
        capacity=capacity,
        requests=hits + misses,
        hits=hits,
        misses=misses,
        evictions=evictions,
    )


__all__ = ["CacheSimulationResult", "simulate_lru", "validate_routing_trace"]
