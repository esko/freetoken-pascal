#!/usr/bin/env python3
"""Run a bounded H0 host-bank/Q4 lifecycle observation.

The default harness requires the repository's Torch-backed ``HostBank`` and
drives the production ``Q4KExecutor`` over packed descriptors.  The arithmetic
is supplied by an embedded deterministic fake Q4/mixed-GEMV primitive so this
small test does not require a native extension or a model artifact.  A
test-only ``--allow-fallback`` flag permits anonymous mmap banks when Torch or
the repository package is unavailable; that run is explicitly labelled as a
non-production fallback.

This is not a benchmark or a model-load test.  RSS, swap, descriptors, and
thread observations are descriptive and never determine pass/fail.  Use an
outer supervisor such as ``timeout 60s python scripts/stress_host_resources.py``
for CI or host runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import mmap
import os
import platform
import random
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_Q4_BLOCK_BYTES = 144
_Q4_BLOCK_ELEMENTS = 256
_Q5_1_BLOCK_BYTES = 24
_EXPERTS = 2
_OUTPUT_DIM = 256
_HIDDEN_DIM = 256
_ROUTES = 4
_DEFAULT_TIMEOUT_SECONDS = 2.0
_MIN_CLEANUP_TIMEOUT_SECONDS = 0.5
_FALLBACK_BUFFERS: list[mmap.mmap] = []
_FAKE_SEAM_LOCK = threading.Lock()


class _StressState:
    """Per-run gate used by the embedded GEMV seam."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def block(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()

    def release(self) -> None:
        gate = self.gate
        if gate is not None:
            gate.set()

    def clear(self) -> None:
        self.release()
        self.gate = None
        self.entered = threading.Event()


class _StressQ4Primitive:
    """Deterministic Q4-shaped direct primitive used by production Q4KExecutor."""

    isa = "avx2"
    backend = "stress_fake_q4_avx2"
    fallback_reason = None

    def __init__(self, state: _StressState) -> None:
        self.state = state

    def decode(
        self,
        block: np.ndarray,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        del block
        result = np.zeros(_Q4_BLOCK_ELEMENTS, dtype=np.float32) if out is None else out
        result.fill(0.0)
        return result

    def gemv(
        self,
        rows: np.ndarray,
        input_dim: int,
        vector: np.ndarray,
        *,
        out: np.ndarray,
        scratch: np.ndarray | None = None,
    ) -> np.ndarray:
        del input_dim, scratch
        self._wait_if_blocked()
        np.multiply(rows[:, 0].astype(np.float32), np.float32(vector.sum()), out=out)
        return out

    def _wait_if_blocked(self) -> None:
        gate = self.state.gate
        if gate is None or gate.is_set():
            return
        self.state.entered.set()
        if not gate.wait(self.state.timeout):
            raise TimeoutError("fake Q4 GEMV gate timed out")


class _StressMixedPrimitive(_StressQ4Primitive):
    """Deterministic Q5_1 companion seam, ensuring mixed-GEMV is exercised."""

    backend = "stress_fake_mixed_gemv_avx2"

    def backend_for(self, quant_name: str) -> str:
        return f"stress_fake_{str(quant_name).lower()}_avx2"

    def gemv(
        self,
        rows: np.ndarray,
        input_dim: int,
        vector: np.ndarray,
        *,
        quant_name: str,
        out: np.ndarray,
    ) -> np.ndarray:
        del input_dim, quant_name
        self._wait_if_blocked()
        np.multiply(rows[:, 0].astype(np.float32), np.float32(vector.sum()), out=out)
        return out


class _PackedSource:
    """Bounded source view over one HostBank mapping."""

    def __init__(self, values: np.ndarray, address: int) -> None:
        self.values: np.ndarray | None = values
        self.range_offset = 0
        self.range_size = int(values.nbytes)
        self.source_address = address

    def expert_packed(self, expert: int) -> np.ndarray:
        if self.values is None:
            raise RuntimeError("stress source is closed")
        return self.values[expert]

    def release_view(self) -> None:
        self.values = None


class _HostFixture:
    """Three packed banks and a complete one-layer CPU ABI layout."""

    def __init__(
        self,
        *,
        banks: dict[str, Any],
        sources: dict[str, _PackedSource],
        policy: Any,
        backend: str,
        layout: Any,
    ) -> None:
        self.banks = banks
        self.sources = sources
        self.policy = policy
        self.backend = backend
        self.layout = layout
        self._closed = False

    @classmethod
    def open(cls, *, fill_seed: int, allow_fallback: bool) -> _HostFixture:
        try:
            import torch
            from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout
            from freetoken.moe.host_banks import HostBankPolicy, alloc_banks
        except ModuleNotFoundError as error:
            if not allow_fallback:
                raise RuntimeError(
                    "Torch and the repository HostBank path are required; "
                    "use --allow-fallback only for an explicit non-production test run"
                ) from error
            return cls._fallback(fill_seed=fill_seed)

        row_bytes = {
            "gate": _Q4_BLOCK_BYTES,
            "up": _Q4_BLOCK_BYTES,
            "down": _Q5_1_BLOCK_BYTES * (_HIDDEN_DIM // 32),
        }
        quant_names = {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"}
        quant_types = {"gate": 12, "up": 12, "down": 7}
        specs = {
            projection: (
                (_EXPERTS, _OUTPUT_DIM, row_bytes[projection]),
                torch.uint8,
            )
            for projection in ("gate", "up", "down")
        }
        policy = HostBankPolicy(strategy="pageable")
        banks = alloc_banks(specs, policy=policy)
        sources: dict[str, _PackedSource] = {}
        descriptors = []
        layout = None
        try:
            for projection in ("gate", "up", "down"):
                bank = banks[projection]
                tensor = bank.tensor
                tensor.zero_()
                values = tensor.numpy().reshape(_EXPERTS, _OUTPUT_DIM, row_bytes[projection])
                for expert in range(_EXPERTS):
                    for row in range(_OUTPUT_DIM):
                        values[expert, row, 0] = (
                            fill_seed + 17 * expert + 3 * row + len(projection)
                        ) & 0xFF
                source = _PackedSource(values, bank.addr)
                sources[projection] = source
                descriptors.append(
                    CpuExpertDescriptor(
                        layer_id=0,
                        projection=projection,
                        quant_type=quant_types[projection],
                        quant_name=quant_names[projection],
                        num_experts=_EXPERTS,
                        output_dim=_OUTPUT_DIM,
                        input_dim=_HIDDEN_DIM,
                        rows_per_expert=_OUTPUT_DIM,
                        row_stride_bytes=row_bytes[projection],
                        expert_stride_bytes=_OUTPUT_DIM * row_bytes[projection],
                        tensor_bytes=_EXPERTS * _OUTPUT_DIM * row_bytes[projection],
                        source=source,
                    )
                )
            # Explicit settlement proves the selected pageable strategy was
            # applied to the complete packed bank set before execution.
            policy.settle(banks)
            layout = CpuExpertLayout(tuple(descriptors), top_k=_ROUTES)
        except BaseException:
            # Drop transient views before closing partially built banks.  In
            # particular, the descriptor constructor may fail after the
            # current tensor/NumPy view has been created but before ``source``
            # is fully owned by the layout.
            tensor = None
            values = None
            source = None
            for source in sources.values():
                source.release_view()
            for bank in banks.values():
                bank.close()
            raise
        assert layout is not None
        return cls(
            banks=banks,
            sources=sources,
            policy=policy,
            backend="torch-hostbank",
            layout=layout,
        )

    @classmethod
    def _fallback(cls, *, fill_seed: int) -> _HostFixture:
        from freetoken.moe.cpu_abi import CpuExpertDescriptor, CpuExpertLayout

        row_bytes = {
            "gate": _Q4_BLOCK_BYTES,
            "up": _Q4_BLOCK_BYTES,
            "down": _Q5_1_BLOCK_BYTES * (_HIDDEN_DIM // 32),
        }
        quant_names = {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"}
        quant_types = {"gate": 12, "up": 12, "down": 7}
        banks: dict[str, mmap.mmap] = {}
        sources: dict[str, _PackedSource] = {}
        descriptors = []
        layout = None
        try:
            for projection in ("gate", "up", "down"):
                nbytes = _EXPERTS * _OUTPUT_DIM * row_bytes[projection]
                mapping = mmap.mmap(
                    -1,
                    nbytes,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
                _FALLBACK_BUFFERS.append(mapping)
                banks[projection] = mapping
                values = np.frombuffer(mapping, dtype=np.uint8, count=nbytes).reshape(
                    _EXPERTS, _OUTPUT_DIM, row_bytes[projection]
                )
                values.fill(0)
                for expert in range(_EXPERTS):
                    for row in range(_OUTPUT_DIM):
                        values[expert, row, 0] = (
                            fill_seed + 17 * expert + 3 * row + len(projection)
                        ) & 0xFF
                address = int(values.__array_interface__["data"][0])
                source = _PackedSource(values, address)
                sources[projection] = source
                descriptors.append(
                    CpuExpertDescriptor(
                        layer_id=0,
                        projection=projection,
                        quant_type=quant_types[projection],
                        quant_name=quant_names[projection],
                        num_experts=_EXPERTS,
                        output_dim=_OUTPUT_DIM,
                        input_dim=_HIDDEN_DIM,
                        rows_per_expert=_OUTPUT_DIM,
                        row_stride_bytes=row_bytes[projection],
                        expert_stride_bytes=_OUTPUT_DIM * row_bytes[projection],
                        tensor_bytes=nbytes,
                        source=source,
                    )
                )
            layout = CpuExpertLayout(tuple(descriptors), top_k=_ROUTES)
        except BaseException:
            # ``values`` may still export the mapping when a later descriptor
            # rejects its source.  Release that local alias before mmap close.
            values = None
            source = None
            for source in sources.values():
                source.release_view()
            for mapping in banks.values():
                mapping.close()
                try:
                    _FALLBACK_BUFFERS.remove(mapping)
                except ValueError:
                    pass
            raise
        assert layout is not None
        return cls(
            banks=banks,
            sources=sources,
            policy=None,
            backend="fallback-test-mmap",
            layout=layout,
        )

    def accounting(self) -> dict[str, object]:
        if self.policy is not None:
            return self.policy.accounting.as_dict()
        source_bytes = sum(source.range_size for source in self.sources.values())
        return {
            "strategy": "pageable",
            "source_bytes": source_bytes,
            "pinned_bytes": 0,
            "staging_bytes": 0,
            "applied_pinned_bytes": 0,
            "applied_staging_bytes": 0,
            "layer_bytes": [source_bytes],
            "layers": ["pageable"],
            "applied_layers": ["pageable"],
            "no_swap_observed": None,
        }

    def close(self) -> None:
        if self._closed:
            return
        for source in self.sources.values():
            source.release_view()
        first_error: BaseException | None = None
        for bank in self.banks.values():
            closed = False
            try:
                bank.close()
                closed = True
            except BaseException as error:
                first_error = first_error or error
            if self.policy is None and closed:
                try:
                    _FALLBACK_BUFFERS.remove(bank)
                except ValueError:
                    pass
        if first_error is None:
            self._closed = True
        if first_error is not None:
            # Keep failed mappings tracked and permit a caller to retry close;
            # removing a still-open mmap would make cleanup accounting lie.
            raise first_error


def _cleanup_timeout(timeout: float) -> float:
    """Give released lifecycle gates enough time to drain before mmap close."""
    return max(float(timeout), _MIN_CLEANUP_TIMEOUT_SECONDS)


def _join_lifecycle_thread(thread: threading.Thread, *, timeout: float, label: str) -> None:
    """Join a released stress owner with a floor independent of tiny test timeouts."""
    _join(thread, timeout=_cleanup_timeout(timeout), label=label)


def _live_buffer_ids() -> tuple[int, ...]:
    try:
        from freetoken.moe.host_banks import _LIVE_BUFFERS
    except ModuleNotFoundError:
        return tuple(id(buffer) for buffer in _FALLBACK_BUFFERS)
    return tuple(id(buffer) for buffer in (*_LIVE_BUFFERS, *_FALLBACK_BUFFERS))


def _fd_count() -> int | None:
    try:
        return len(os.listdir("/proc/self/fd"))
    except (FileNotFoundError, OSError):
        return None


def _rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                value, unit = line.split()[1:3]
                return int(value) * 1024 if unit == "kB" else None
    except (FileNotFoundError, OSError, ValueError):
        return None
    return None


def _thread_snapshot() -> dict[str, object]:
    threads = tuple(threading.enumerate())
    workers = tuple(
        {"ident": thread.ident, "name": thread.name}
        for thread in threads
        if thread.name.startswith("freetoken-mixed")
    )
    return {
        "total": len(threads),
        "freetoken_mixed": len(workers),
        "workers": workers,
    }


def _unexpected_workers(
    baseline_workers: Sequence[dict[str, object]],
    current_workers: Sequence[dict[str, object]],
) -> tuple[dict[str, object], ...]:
    """Return current mixed workers that were not present at baseline.

    An unrelated executor may be garbage-collected while the stress run is
    active, so baseline workers are allowed to disappear.  New workers still
    indicate a leaked executor and must fail cleanup.
    """
    return tuple(worker for worker in current_workers if worker not in baseline_workers)


def _swap_probe(*, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {"status": "not-requested", "source": None, "errors": []}
    try:
        from freetoken.moe.host_memory import probe_swap

        return probe_swap().as_dict()
    except Exception as error:
        return {
            "status": "unavailable",
            "source": "procfs",
            "errors": [f"probe failed: {error}"],
        }


def _git_sha() -> str:
    supplied = os.environ.get("FREETOKEN_STRESS_GIT_COMMIT", "").strip()
    if supplied:
        return supplied
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _host_identity() -> dict[str, str]:
    return {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _wait(event: threading.Event, *, timeout: float, label: str) -> None:
    if not event.wait(timeout):
        raise TimeoutError(f"stress wait timed out: {label}")


def _join(thread: threading.Thread, *, timeout: float, label: str) -> None:
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"stress thread timed out: {label}")


def _telemetry_observation(telemetry: Any) -> dict[str, object]:
    """Keep stable, decision-bearing fields from one production request."""
    return {
        "backend": telemetry.backend,
        "kernel_census": list(telemetry.kernel_census),
        "fallback_reason": telemetry.fallback_reason,
        "error": telemetry.error,
        "cancelled": telemetry.cancelled,
        "thread_count": telemetry.thread_count,
        "routes_executed": telemetry.routes_executed,
    }


def _inputs(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden = np.linspace(-0.5, 0.5, 2 * _HIDDEN_DIM, dtype=np.float32).reshape(2, _HIDDEN_DIM)
    ids = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.int32)
    weights = np.array(
        [0.2 + seed / 10000, -0.3, 0.4, 0.5, -0.25, 0.75, 0.0, 0.1],
        dtype=np.float32,
    ).reshape(2, _ROUTES)
    return hidden, ids, weights


def _capture(bucket: list[BaseException], function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        function(*args, **kwargs)
    except BaseException as error:
        bucket.append(error)


def _detach_tracebacks(error: BaseException) -> None:
    """Drop execution-frame references before closing mmap-backed sources."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


def _production_executor(q4_k: Any, layout: Any, *, threads: int) -> tuple[Any, Any]:
    serial = None
    threaded = None
    try:
        serial = q4_k.Q4KExecutor(layout, mode="avx2", num_threads=1, required_alignment=1)
        threaded = q4_k.Q4KExecutor(
            layout,
            mode="avx2",
            num_threads=threads,
            required_alignment=1,
        )
        serial.prepare(2, _ROUTES)
        threaded.prepare(2, _ROUTES)
    except BaseException:
        for executor in (threaded, serial):
            if executor is not None:
                try:
                    executor.close()
                except BaseException:
                    pass
        raise
    assert serial is not None and threaded is not None
    return serial, threaded


def _run_q4_iteration(
    *,
    q4_k: Any,
    layout: Any,
    threads: int,
    seed: int,
    timeout: float,
) -> dict[str, object]:
    """Exercise real Q4 admission, partition, cancellation, and teardown."""
    hidden, expert_ids, weights = _inputs(seed)
    state = _StressState(timeout)
    q4_primitive = _StressQ4Primitive(state)
    mixed_primitive = _StressMixedPrimitive(state)
    original_q4 = q4_k.select_q4_k_primitive
    original_mixed = q4_k.select_mixed_gemv_primitive
    q4_k.select_q4_k_primitive = lambda mode="auto": q4_primitive
    q4_k.select_mixed_gemv_primitive = lambda mode="auto": mixed_primitive
    serial = threaded = None
    close_executor = None
    try:
        serial, threaded = _production_executor(q4_k, layout, threads=threads)
        expected = serial.execute(0, hidden, expert_ids, weights).output.copy()
        actual = threaded.execute(0, hidden, expert_ids, weights)
        if not np.allclose(actual.output, expected, rtol=3e-5, atol=3e-5):
            raise AssertionError("threaded production Q4 output differs from serial output")
        if actual.telemetry.thread_count != threads:
            raise AssertionError(
                f"production Q4 telemetry thread_count={actual.telemetry.thread_count}, "
                f"expected {threads}"
            )
        if actual.telemetry.error is not None or actual.telemetry.cancelled:
            raise AssertionError("normal production Q4 request reported an error")
        if actual.telemetry.backend != "mixed_avx2":
            raise AssertionError(
                f"normal production Q4 backend was {actual.telemetry.backend!r}, "
                "expected mixed_avx2"
            )
        if actual.telemetry.kernel_census != (
            "stress_fake_q4_avx2",
            "stress_fake_q5_1_avx2",
        ):
            raise AssertionError(
                "normal production Q4 selected an unexpected kernel census: "
                f"{actual.telemetry.kernel_census!r}"
            )
        if actual.telemetry.fallback_reason is not None:
            raise AssertionError(
                "normal production Q4 unexpectedly reported fallback: "
                f"{actual.telemetry.fallback_reason!r}"
            )

        state.block()
        busy_errors: list[BaseException] = []
        owner_errors: list[BaseException] = []
        owner = threading.Thread(
            target=_capture,
            args=(owner_errors, threaded.execute, 0, hidden, expert_ids, weights),
            name="stress-q4-busy-owner",
        )
        owner.start()
        try:
            _wait(state.entered, timeout=timeout, label="production Busy owner entry")
            contender = threading.Thread(
                target=_capture,
                args=(busy_errors, threaded.execute, 0, hidden, expert_ids, weights),
                name="stress-q4-busy-contender",
            )
            contender.start()
            _join(contender, timeout=timeout, label="production Busy contender")
            if not busy_errors or not isinstance(busy_errors[0], q4_k.Busy):
                raise AssertionError("production concurrent request was not rejected as Busy")
            busy_error = busy_errors[0]
            if busy_error.telemetry is None:
                raise AssertionError("production Busy rejection did not expose telemetry")
            if busy_error.telemetry.error != "Busy":
                raise AssertionError(
                    f"production Busy telemetry error was {busy_error.telemetry.error!r}"
                )
            if busy_error.telemetry.backend != "mixed_avx2":
                raise AssertionError(
                    "production Busy telemetry lost the selected backend: "
                    f"{busy_error.telemetry.backend!r}"
                )
        finally:
            state.release()
            try:
                _join_lifecycle_thread(owner, timeout=timeout, label="production Busy owner")
            finally:
                state.clear()
        if owner_errors:
            raise owner_errors[0]

        state.block()
        cancellation = threading.Event()
        cancelled_output = np.full_like(hidden, 9.0)
        cancel_errors: list[BaseException] = []
        cancelled_owner = threading.Thread(
            target=_capture,
            args=(cancel_errors, threaded.execute, 0, hidden, expert_ids, weights),
            kwargs={"output": cancelled_output, "cancellation": cancellation.is_set},
            name="stress-q4-cancel-owner",
        )
        cancelled_owner.start()
        try:
            _wait(state.entered, timeout=timeout, label="production cancellation entry")
            cancellation.set()
        finally:
            # A rendezvous timeout must cancel the owner before its gate is
            # released, so the owner cannot continue against teardown.
            cancellation.set()
            state.release()
            try:
                _join_lifecycle_thread(
                    cancelled_owner,
                    timeout=timeout,
                    label="production cancellation owner",
                )
            finally:
                state.clear()
        if not cancel_errors or not isinstance(cancel_errors[0], q4_k.Cancelled):
            raise AssertionError("production cancellation did not raise Cancelled")
        cancellation_error = cancel_errors[0]
        if cancellation_error.telemetry is None or not cancellation_error.telemetry.cancelled:
            raise AssertionError("production cancellation telemetry was not marked cancelled")
        if not np.all(cancelled_output == 0):
            raise AssertionError("production cancellation did not zero/rollback output")

        recovered_result = threaded.execute(0, hidden, expert_ids, weights)
        if not np.allclose(recovered_result.output, expected, rtol=3e-5, atol=3e-5):
            raise AssertionError("production Q4 did not recover after cancellation")

        close_executor = q4_k.Q4KExecutor(
            layout,
            mode="avx2",
            num_threads=threads,
            required_alignment=1,
        )
        close_executor.prepare(2, _ROUTES)
        state.block()
        close_errors: list[BaseException] = []
        close_owner = threading.Thread(
            target=_capture,
            args=(close_errors, close_executor.execute, 0, hidden, expert_ids, weights),
            name="stress-q4-close-owner",
        )
        close_owner.start()
        closer_started = threading.Event()
        closer_errors: list[BaseException] = []

        def close_in_thread() -> None:
            closer_started.set()
            _capture(closer_errors, close_executor.close)

        closer = threading.Thread(target=close_in_thread, name="stress-q4-close")
        closer_started_flag = False
        startup_error: BaseException | None = None
        try:
            _wait(state.entered, timeout=timeout, label="production close-race entry")
            closer.start()
            closer_started_flag = True
            _wait(closer_started, timeout=timeout, label="production close-race closer start")
            if not closer.is_alive():
                raise AssertionError("production close returned before gated execute completed")
        except BaseException as error:
            startup_error = error
        finally:
            state.release()
        try:
            _join_lifecycle_thread(
                close_owner,
                timeout=timeout,
                label="production close-race owner",
            )
            if closer_started_flag:
                _join_lifecycle_thread(
                    closer,
                    timeout=timeout,
                    label="production close-race closer",
                )
            else:
                close_executor.close()
        finally:
            state.clear()
        if startup_error is not None:
            raise startup_error
        if close_errors or closer_errors:
            raise (close_errors or closer_errors)[0]
        close_executor.close()
        try:
            close_executor.execute(0, hidden, expert_ids, weights)
        except q4_k.InvalidRequest:
            post_close_rejected = 1
        else:
            raise AssertionError("production post-close Q4 request was accepted")
        return {
            "serial_thread_parity": 1,
            "busy": 1,
            "cancelled": 1,
            "recovered": 1,
            "close_race": 1,
            "post_close_rejected": post_close_rejected,
            "telemetry": _telemetry_observation(actual.telemetry),
            "busy_telemetry": _telemetry_observation(busy_error.telemetry),
        }
    finally:
        state.release()
        cleanup_errors: list[BaseException] = []
        for executor in (close_executor, threaded, serial):
            if executor is not None:
                try:
                    executor.close()
                except BaseException as error:
                    cleanup_errors.append(error)
        # Restore both module globals even when one close path fails.  A
        # caller's next executor must never inherit this stress seam.
        q4_k.select_q4_k_primitive = original_q4
        q4_k.select_mixed_gemv_primitive = original_mixed
        if cleanup_errors and sys.exc_info()[0] is None:
            raise cleanup_errors[0]


def _aggregate_policy(
    actual_samples: Sequence[dict[str, object]],
    *,
    fallback: bool,
) -> dict[str, object]:
    if not actual_samples:
        raise ValueError("stress requires at least one policy accounting sample")
    current = dict(actual_samples[-1])
    current.update({"allocation_scope": "one_iteration", "allocation_state": "closed"})
    cumulative = {
        "allocation_scope": "all_iterations",
        "allocations": len(actual_samples),
        "closed_allocations": len(actual_samples),
        "source_bytes": sum(int(item.get("source_bytes", 0)) for item in actual_samples),
        "pinned_bytes": sum(int(item.get("pinned_bytes", 0)) for item in actual_samples),
        "staging_bytes": sum(int(item.get("staging_bytes", 0)) for item in actual_samples),
        "applied_pinned_bytes": sum(
            int(item.get("applied_pinned_bytes", 0)) for item in actual_samples
        ),
        "applied_staging_bytes": sum(
            int(item.get("applied_staging_bytes", 0)) for item in actual_samples
        ),
        "strategy": "pageable",
        "all_pageable": True,
    }
    return {
        "current": current,
        "cumulative": cumulative,
        "production_policy": not fallback,
    }


def _cleanup_check(
    baseline_live: tuple[int, ...],
    baseline_workers: tuple[dict[str, object], ...],
    baseline_fds: int | None,
) -> None:
    errors: list[str] = []
    if _live_buffer_ids() != baseline_live:
        errors.append("live host buffers were not restored")
    current_workers = _thread_snapshot()["workers"]
    assert isinstance(current_workers, tuple)
    unexpected_workers = _unexpected_workers(baseline_workers, current_workers)
    if unexpected_workers:
        errors.append(f"new freetoken-mixed workers leaked: {unexpected_workers!r}")
    current_fds = _fd_count()
    if baseline_fds is not None and current_fds is not None and current_fds != baseline_fds:
        errors.append(f"file-descriptor count changed: {baseline_fds} -> {current_fds}")
    if errors:
        raise RuntimeError("; ".join(errors))


def run_stress(
    *,
    seed: int = 18,
    iterations: int = 8,
    threads: int = 2,
    backend: str = "production",
    per_iteration_timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    include_resource_probes: bool = True,
    allow_fallback: bool = False,
) -> dict[str, object]:
    """Run production Q4/HostBank stress and return a JSON-serializable report."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _validate_positive_int(iterations, "iterations")
    _validate_positive_int(threads, "threads")
    if threads < 2:
        raise ValueError("threads must be at least 2 for the threaded Q4 stress")
    if threads > _ROUTES:
        raise ValueError(f"threads must not exceed {_ROUTES} routes, got {threads}")
    if backend != "production":
        raise ValueError("backend must be production; use allow_fallback for test-only banks")
    if (
        isinstance(per_iteration_timeout, bool)
        or not isinstance(per_iteration_timeout, int | float)
        or per_iteration_timeout <= 0
    ):
        raise ValueError("per_iteration_timeout must be positive")

    try:
        from freetoken.moe import q4_k
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "repository Python package is required for production Q4 stress"
        ) from error

    baseline_live = _live_buffer_ids()
    baseline_threads = _thread_snapshot()
    baseline_workers = tuple(baseline_threads["workers"])
    baseline_fds = _fd_count()
    rss_before = _rss_bytes()
    swap_before = _swap_probe(enabled=include_resource_probes)
    rng = random.Random(seed)
    actual_samples: list[dict[str, object]] = []
    scenario_counts = {
        "serial_thread_parity": 0,
        "busy": 0,
        "cancelled": 0,
        "recovered": 0,
        "close_race": 0,
    }
    accepted = busy = cancelled = recovered = 0
    host_backends: set[str] = set()
    normal_telemetry: dict[str, object] | None = None
    busy_telemetry: dict[str, object] | None = None
    rss_samples: list[int] = []
    start = time.monotonic()
    try:
        with _FAKE_SEAM_LOCK:
            for iteration in range(iterations):
                fixture = _HostFixture.open(
                    fill_seed=rng.randrange(0, 65536),
                    allow_fallback=allow_fallback,
                )
                host_backends.add(fixture.backend)
                try:
                    actual_samples.append(fixture.accounting())
                    try:
                        result = _run_q4_iteration(
                            q4_k=q4_k,
                            layout=fixture.layout,
                            threads=threads,
                            seed=seed + iteration,
                            timeout=float(per_iteration_timeout),
                        )
                    except BaseException as error:
                        # Q4 may retain packed ndarray locals in an exception
                        # traceback.  Detach them before closing HostBank mmaps.
                        _detach_tracebacks(error)
                        gc.collect()
                        raise
                    for key, value in result.items():
                        if key in scenario_counts:
                            scenario_counts[key] += value
                    accepted += 5
                    busy += result["busy"]
                    cancelled += result["cancelled"]
                    recovered += result["recovered"]
                    if normal_telemetry is None:
                        normal_telemetry = result["telemetry"]
                    if busy_telemetry is None:
                        busy_telemetry = result["busy_telemetry"]
                finally:
                    fixture.close()
                sample_rss = _rss_bytes()
                if sample_rss is not None:
                    rss_samples.append(sample_rss)
                if time.monotonic() - start > iterations * float(per_iteration_timeout) * 8:
                    raise TimeoutError("stress outer budget exceeded")
    except BaseException as error:
        try:
            _cleanup_check(baseline_live, baseline_workers, baseline_fds)
        except RuntimeError as cleanup_error:
            raise cleanup_error from error
        raise

    swap_after = _swap_probe(enabled=include_resource_probes)
    rss_after = _rss_bytes()
    final_live = _live_buffer_ids()
    final_threads = _thread_snapshot()
    final_fds = _fd_count()
    _cleanup_check(baseline_live, baseline_workers, baseline_fds)
    if normal_telemetry is None or busy_telemetry is None:
        raise RuntimeError("stress did not produce normal and Busy telemetry")
    command = " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv))
    sampled_max_after_iteration = max(rss_samples, default=rss_before)
    selected_kernel_census = normal_telemetry["kernel_census"]
    if not isinstance(selected_kernel_census, list):
        raise RuntimeError("normal telemetry kernel census was not a list")
    return {
        "schema_version": 2,
        "claim_status": "observation_only",
        "seed": seed,
        "iterations": iterations,
        "threads": threads,
        "backend": f"Q4KExecutor+{normal_telemetry['backend']}",
        "host_bank_backend": sorted(host_backends),
        "production_path": not allow_fallback or "torch-hostbank" in host_backends,
        "kernel_census": {
            "executor": "Q4KExecutor",
            "backend": normal_telemetry["backend"],
            "selected": selected_kernel_census,
            "partition": "production partition_q4_k_routes",
            "fallback_reason": normal_telemetry["fallback_reason"],
            "primitive_seam": "embedded-deterministic-fake",
        },
        "telemetry": normal_telemetry,
        "busy_telemetry": busy_telemetry,
        "accepted": accepted,
        "busy": busy,
        "cancelled": cancelled,
        "recovered": recovered,
        "scenario_counts": scenario_counts,
        "policy_accounting": _aggregate_policy(
            actual_samples, fallback=allow_fallback and "torch-hostbank" not in host_backends
        ),
        "live_buffers": {
            "before": len(baseline_live),
            "after": len(final_live),
            "restored": final_live == baseline_live,
        },
        "fd_counts": {
            "before": baseline_fds,
            "after": final_fds,
            "delta": None
            if baseline_fds is None or final_fds is None
            else final_fds - baseline_fds,
            "restored": (
                None if baseline_fds is None or final_fds is None else baseline_fds == final_fds
            ),
        },
        "thread_counts": {
            "before": baseline_threads,
            "after": final_threads,
            "restored": not _unexpected_workers(baseline_workers, tuple(final_threads["workers"])),
        },
        "rss_bytes": {
            "before": rss_before,
            "after_iteration_samples": rss_samples,
            "sampled_max_after_iteration": sampled_max_after_iteration,
            "after": rss_after,
        },
        "swap_probe": {"before": swap_before, "after": swap_after},
        "host": _host_identity(),
        "git_commit": _git_sha(),
        "command": command,
    }


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--per-iteration-timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_SECONDS,
        help="seconds allowed for each lifecycle rendezvous (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="test-only mmap fallback when Torch/HostBank is unavailable",
    )
    parser.add_argument(
        "--no-resource-probes",
        action="store_true",
        help="omit procfs swap/RSS/descriptor observations",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = run_stress(
            seed=args.seed,
            iterations=args.iterations,
            threads=args.threads,
            per_iteration_timeout=args.per_iteration_timeout,
            include_resource_probes=not args.no_resource_probes,
            allow_fallback=args.allow_fallback,
        )
    except (AssertionError, TimeoutError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
