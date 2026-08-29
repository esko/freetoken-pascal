"""Explicit borrowed-bundle attachment for the Qwen4-Exp model graph.

The helper deliberately knows only the model's small structural/configuration
surface. It does not import or construct a Qwen model, Engine, or CUDA operator,
which keeps fake-layer lifecycle tests independent from model construction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from freetoken.moe.gguf_cpu import QwenGGUFCpuExpertBundle
from freetoken.moe.gguf_layer import QwenGGUFCpuMoELayer
from freetoken.moe.gguf_transfer import EagerTransferSeam, GGUFCpuEagerBridge


@dataclass(frozen=True)
class GGUFCpuExpertAttachment:
    """Reversible state for one borrowed Qwen GGUF expert attachment."""

    bundle: QwenGGUFCpuExpertBundle
    replacements: tuple[tuple[object, object, object, object], ...]
    mode: str = "cpu"
    owned_bridges: tuple[object, ...] = ()


def validate_gguf_cpu_attachment(model: Any, bundle: QwenGGUFCpuExpertBundle) -> tuple[object, ...]:
    """Validate all model/bundle metadata before constructing any adapter."""
    if not isinstance(bundle, QwenGGUFCpuExpertBundle):
        raise TypeError("Qwen4-Exp model attachment requires a QwenGGUFCpuExpertBundle")
    if bundle.closed:
        raise RuntimeError("Qwen GGUF CPU expert bundle is closed")

    config = model._config
    layers = tuple(model.layers.op_list)
    expected_ids = tuple(range(config.num_layers))
    try:
        model_ids = tuple(layer._layer_id for layer in layers)
    except AttributeError as error:
        raise ValueError("Qwen4-Exp model layers must expose _layer_id") from error
    if model_ids != expected_ids:
        raise ValueError(
            f"Qwen4-Exp model layer IDs {model_ids} do not match expected layer IDs {expected_ids}"
        )
    if tuple(bundle.layout.layers) != expected_ids:
        raise ValueError(
            f"Qwen GGUF CPU bundle layer IDs {tuple(bundle.layout.layers)} do not match "
            f"model layer IDs {expected_ids}"
        )
    if bundle.layout.top_k != config.num_experts_per_tok:
        raise ValueError(
            f"Qwen GGUF CPU bundle top_k={bundle.layout.top_k} does not match "
            f"model top_k={config.num_experts_per_tok}"
        )

    for layer_id in expected_ids:
        try:
            gate = bundle.layout.descriptor(layer_id, "gate")
            up = bundle.layout.descriptor(layer_id, "up")
            down = bundle.layout.descriptor(layer_id, "down")
        except Exception as error:
            raise ValueError(
                f"Qwen GGUF CPU bundle is missing projections for layer {layer_id}"
            ) from error
        for descriptor in (gate, up, down):
            if descriptor.num_experts != config.num_experts:
                raise ValueError(
                    f"Qwen GGUF CPU bundle layer {layer_id} {descriptor.projection} "
                    f"num_experts={descriptor.num_experts} does not match model "
                    f"num_experts={config.num_experts}"
                )
        if (gate.input_dim, gate.output_dim) != (
            config.hidden_size,
            config.moe_intermediate_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} gate geometry "
                f"({gate.input_dim}, {gate.output_dim}) does not match model geometry "
                f"({config.hidden_size}, {config.moe_intermediate_size})"
            )
        if (up.input_dim, up.output_dim) != (
            config.hidden_size,
            config.moe_intermediate_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} up geometry "
                f"({up.input_dim}, {up.output_dim}) does not match model geometry "
                f"({config.hidden_size}, {config.moe_intermediate_size})"
            )
        if (down.input_dim, down.output_dim) != (
            config.moe_intermediate_size,
            config.hidden_size,
        ):
            raise ValueError(
                f"Qwen GGUF CPU bundle layer {layer_id} down geometry "
                f"({down.input_dim}, {down.output_dim}) does not match model geometry "
                f"({config.moe_intermediate_size}, {config.hidden_size})"
            )

    try:
        from freetoken.distributed.info import try_get_tp_info

        tp_info = try_get_tp_info()
    except ImportError:  # pragma: no cover - distributed package ships with runtime
        tp_info = None
    if tp_info is not None and tp_info.size != 1:
        raise ValueError(
            f"Qwen GGUF CPU model attachment requires TP=1, got tensor-parallel size {tp_info.size}"
        )

    executor = bundle.executor
    executor_reference = getattr(executor, "_reference", None)
    activation = getattr(
        executor,
        "activation",
        getattr(executor_reference, "activation", None),
    )
    if activation != config.hidden_act:
        raise ValueError(
            f"Qwen GGUF CPU bundle activation={activation!r} does not match "
            f"model activation={config.hidden_act!r}"
        )
    router_weight = getattr(
        executor,
        "apply_router_weight_on_input",
        getattr(executor_reference, "apply_router_weight_on_input", None),
    )
    if router_weight is not False:
        raise ValueError(
            "Qwen GGUF CPU model attachment router-weight policy requires "
            "apply_router_weight_on_input=False"
        )
    return layers


def _attachment_lock(model: Any) -> threading.RLock:
    lock = getattr(model, "_gguf_attachment_lock", None)
    if lock is None:
        lock = threading.RLock()
        model._gguf_attachment_lock = lock
    return lock


def _close_bridges(bridges: tuple[object, ...]) -> None:
    for bridge in reversed(bridges):
        close = getattr(bridge, "close", None)
        if close is not None:
            close()


def _freeze_bridges(bridges: tuple[object, ...]) -> tuple[object, ...]:
    frozen: list[object] = []
    try:
        for bridge in bridges:
            freeze = getattr(bridge, "freeze_admission", None)
            if freeze is None:
                raise RuntimeError("eager bridge does not support transactional admission freeze")
            freeze()
            frozen.append(bridge)
    except BaseException:
        _unfreeze_bridges(tuple(frozen))
        raise
    return tuple(frozen)


def _unfreeze_bridges(bridges: tuple[object, ...]) -> None:
    for bridge in reversed(bridges):
        unfreeze = getattr(bridge, "unfreeze_admission", None)
        if unfreeze is not None:
            unfreeze()


def _rollback_closed_bridges(bridges: tuple[object, ...]) -> None:
    for bridge in reversed(bridges):
        rollback = getattr(bridge, "rollback_close", None)
        if rollback is None:
            raise RuntimeError("eager bridge cannot roll back a transactional close")
        rollback()


def _build_cpu_adapter(model: Any, bundle: QwenGGUFCpuExpertBundle, layer_id: int) -> object:
    config = model._config
    return QwenGGUFCpuMoELayer(
        bundle,
        layer_id=layer_id,
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        renormalize=bool(config.norm_topk_prob),
        activation=config.hidden_act,
        apply_router_weight_on_input=False,
    )


def _attach_gguf_cpu_experts(
    model: Any,
    bundle: QwenGGUFCpuExpertBundle,
    *,
    eager_bridge: bool,
    transfer: EagerTransferSeam | None,
) -> None:
    """Build every replacement before atomically changing the model graph."""
    with _attachment_lock(model):
        if getattr(model, "_gguf_cpu_attachment", None) is not None:
            raise RuntimeError("Qwen GGUF CPU expert bundle is already attached")
        layers = validate_gguf_cpu_attachment(model, bundle)
        layer_records: list[tuple[object, object, object]] = []
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "experts"):
                raise ValueError(
                    f"Qwen4-Exp layer {getattr(layer, '_layer_id', '?')} has no sparse experts"
                )
            layer_records.append((layer, mlp, mlp.experts))

        replacements: list[tuple[object, object, object, object]] = []
        owned_bridges: list[object] = []
        try:
            for layer, mlp, original in layer_records:
                adapter = _build_cpu_adapter(model, bundle, layer._layer_id)
                installed = adapter
                if eager_bridge:
                    installed = GGUFCpuEagerBridge(
                        adapter,
                        transfer=transfer,
                        cache_size=0,
                        tp_size=1,
                    )
                    owned_bridges.append(installed)
                replacements.append((layer, mlp, original, installed))
        except BaseException:
            _close_bridges(tuple(owned_bridges))
            raise

        attachment = GGUFCpuExpertAttachment(
            bundle,
            tuple(replacements),
            mode="eager" if eager_bridge else "cpu",
            owned_bridges=tuple(owned_bridges),
        )
        try:
            for _layer, mlp, _original, installed in replacements:
                mlp.experts = installed
            model._gguf_cpu_attachment = attachment
        except BaseException:
            for _layer, mlp, original, _installed in reversed(replacements):
                mlp.experts = original
            _close_bridges(tuple(owned_bridges))
            raise


def attach_gguf_cpu_expert_bundle(model: Any, bundle: QwenGGUFCpuExpertBundle) -> None:
    """Validate, construct, and atomically install all CPU layer adapters."""
    _attach_gguf_cpu_experts(model, bundle, eager_bridge=False, transfer=None)


def attach_gguf_cpu_eager_bridge(
    model: Any,
    bundle: QwenGGUFCpuExpertBundle,
    *,
    transfer: EagerTransferSeam | None = None,
) -> None:
    """Attach explicit blocking device bridges around every CPU layer adapter."""
    _attach_gguf_cpu_experts(model, bundle, eager_bridge=True, transfer=transfer)


def detach_gguf_cpu_expert_bundle(model: Any) -> None:
    """Restore exact originals, quiesce owned bridges, and never close the bundle."""
    with _attachment_lock(model):
        attachment = getattr(model, "_gguf_cpu_attachment", None)
        if attachment is None:
            return

        for layer, mlp, _original, installed in attachment.replacements:
            if (
                getattr(layer, "mlp", None) is not mlp
                or getattr(mlp, "experts", None) is not installed
            ):
                raise RuntimeError(
                    "Qwen GGUF CPU expert attachment was modified outside its lifecycle"
                )

        if not attachment.owned_bridges:
            frozen = ()
            closed = ()
        else:
            frozen = _freeze_bridges(attachment.owned_bridges)
            closed: list[object] = []
            try:
                for bridge in attachment.owned_bridges:
                    close = getattr(bridge, "close", None)
                    if close is None:
                        raise RuntimeError("eager bridge cannot close during detach")
                    close()
                    closed.append(bridge)
            except BaseException:
                # A failed close leaves the model graph installed. Re-open any bridge
                # that did close, then release the admission freeze for a retry.
                failed = bridge
                if bool(getattr(failed, "closed", False)) and failed not in closed:
                    closed.append(failed)
                _rollback_closed_bridges(tuple(closed))
                _unfreeze_bridges(frozen)
                raise
            closed = tuple(closed)

        try:
            for _layer, mlp, original, _installed in attachment.replacements:
                mlp.experts = original
            model._gguf_cpu_attachment = None
        except BaseException:
            for _layer, mlp, _original, installed in reversed(attachment.replacements):
                mlp.experts = installed
            if closed:
                _rollback_closed_bridges(closed)
                _unfreeze_bridges(frozen)
            raise


def append_original_expert_state(
    model: Any,
    result: dict[str, Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Keep runtime adapter replacement out of the model state-dict surface."""
    attachment = getattr(model, "_gguf_cpu_attachment", None)
    if attachment is None:
        return result
    for index, (_layer, _mlp, original, _adapter) in enumerate(attachment.replacements):
        original_state_dict = getattr(original, "state_dict", None)
        if original_state_dict is None:
            continue
        expert_prefix = (
            f"{prefix}.layers.{index}.mlp.experts" if prefix else f"layers.{index}.mlp.experts"
        )
        original_state_dict(prefix=expert_prefix, result=result)
    return result


def gguf_cpu_expert_telemetry(model: Any) -> dict[int, dict[str, object]]:
    """Return per-layer telemetry for the currently attached expert wrappers."""
    attachment = getattr(model, "_gguf_cpu_attachment", None)
    if attachment is None:
        return {}
    result: dict[int, dict[str, object]] = {}
    for layer, _mlp, _original, installed in attachment.replacements:
        telemetry = getattr(installed, "host_weight_telemetry", None)
        if callable(telemetry):
            value = telemetry()
        elif isinstance(telemetry, dict):
            value = dict(telemetry)
        else:
            value = {}
        result[int(layer._layer_id)] = value
    return result


__all__ = [
    "GGUFCpuExpertAttachment",
    "append_original_expert_state",
    "attach_gguf_cpu_eager_bridge",
    "attach_gguf_cpu_expert_bundle",
    "detach_gguf_cpu_expert_bundle",
    "gguf_cpu_expert_telemetry",
    "validate_gguf_cpu_attachment",
]
