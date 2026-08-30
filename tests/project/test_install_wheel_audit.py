from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts" / "verify_pascal_wheel_bundle.py"
INSTALL = ROOT / "install.sh"


def _write_wheel(
    path: Path,
    *,
    role: str,
    cuda: str = "12.6",
    architectures: tuple[str, ...] = ("6.1",),
    metadata: object | None = None,
    package_metadata: bool = True,
) -> Path:
    package = "freetoken" if role == "runtime" else "freetoken_kernel_cache"
    distribution = "freetoken_pascal" if role == "runtime" else "freetoken_kernel_cache"
    dist_info = f"{distribution}-0.1.0.dist-info"
    if metadata is None:
        metadata = {
            "schema": "freetoken-pascal-bundle-v1",
            "profile": "pascal",
            "role": role,
            "cuda": cuda,
            "architectures": list(architectures),
            "version": "0.1.0",
            "runtime_version": "0.1.0",
        }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        wheel.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Name: {'freetoken-pascal' if role == 'runtime' else 'freetoken-kernel-cache'}\n"
            "Version: 0.1.0\n",
        )
        wheel.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: py3-none-any\n",
        )
        if package_metadata:
            wheel.writestr(f"{package}/_pascal_build_meta.json", json.dumps(metadata))
    return path


def _audit(runtime: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDIT), "--runtime", str(runtime), "--kernel-cache", str(cache)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_pascal_wheel_audit_accepts_cu126_sm61_bundle(tmp_path: Path) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime")
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")

    result = _audit(runtime, cache)

    assert result.returncode == 0, result.stderr
    assert "CUDA 12.6" in result.stdout
    assert "6.1" in result.stdout


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cuda", "13.0", "CUDA 12.6"),
        ("architectures", ["8.0"], "6.1"),
    ],
)
def test_pascal_wheel_audit_rejects_unsupported_bundle_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime", **{field: value})
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")

    result = _audit(runtime, cache)

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metadata": "not json"},
        {"package_metadata": False},
    ],
)
def test_pascal_wheel_audit_rejects_malformed_or_missing_metadata(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime", **kwargs)
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")

    result = _audit(runtime, cache)

    assert result.returncode != 0
    assert "metadata" in result.stderr.lower()


def test_pascal_wheel_audit_rejects_url_bundle(tmp_path: Path) -> None:
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")

    result = _audit(Path("https://example.invalid/runtime.whl"), cache)

    assert result.returncode != 0
    assert "local" in result.stderr.lower()


def test_pascal_wheel_audit_rejects_mismatched_runtime_identity(tmp_path: Path) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime")
    cache = _write_wheel(
        tmp_path / "cache.whl",
        role="kernel-cache",
        metadata={
            "schema": "freetoken-pascal-bundle-v1",
            "profile": "pascal",
            "role": "kernel-cache",
            "cuda": "12.6",
            "architectures": ["6.1"],
            "version": "0.1.0",
            "runtime_version": "0.2.0",
        },
    )

    result = _audit(runtime, cache)

    assert result.returncode != 0
    assert "runtime_version" in result.stderr


def test_installer_rejects_before_uv_or_filesystem_side_effects(tmp_path: Path) -> None:
    marker = tmp_path / "side-effect-marker"
    uv = tmp_path / "mock-bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f"#!/bin/sh\nprintf side-effect > {marker}\nexit 99\n", encoding="utf-8")
    uv.chmod(0o755)
    home = tmp_path / "install-root"
    env_dir = tmp_path / "environment.d"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{uv.parent}:{env['PATH']}",
            "FREETOKEN_HOME": str(home),
            "FREETOKEN_ENV_DIR": str(env_dir),
            "FREETOKEN_BIN_DIR": str(tmp_path / "bin"),
        }
    )
    env.pop("FREETOKEN_WHEEL", None)
    env.pop("FREETOKEN_KERNEL_CACHE_WHEEL", None)

    result = subprocess.run(
        ["bash", str(INSTALL)], check=False, capture_output=True, text=True, env=env
    )

    assert result.returncode != 0
    assert "local runtime wheel" in result.stderr
    assert not marker.exists()
    assert not home.exists()
    assert not env_dir.exists()


def test_installer_rejects_bad_explicit_bundle_before_uv(tmp_path: Path) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime", cuda="13.0")
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")
    marker = tmp_path / "side-effect-marker"
    uv = tmp_path / "mock-bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f"#!/bin/sh\nprintf side-effect > {marker}\nexit 99\n", encoding="utf-8")
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{uv.parent}:{env['PATH']}",
            "FREETOKEN_WHEEL": str(runtime),
            "FREETOKEN_KERNEL_CACHE_WHEEL": str(cache),
            "FREETOKEN_HOME": str(tmp_path / "install-root"),
            "FREETOKEN_ENV_DIR": str(tmp_path / "environment.d"),
            "FREETOKEN_BIN_DIR": str(tmp_path / "bin"),
        }
    )

    result = subprocess.run(
        ["bash", str(INSTALL)], check=False, capture_output=True, text=True, env=env
    )

    assert result.returncode != 0
    assert "CUDA 12.6" in result.stderr
    assert not marker.exists()


def test_installer_pascal_profile_uses_only_core_wheels_and_cu126(tmp_path: Path) -> None:
    runtime = _write_wheel(tmp_path / "runtime.whl", role="runtime")
    cache = _write_wheel(tmp_path / "cache.whl", role="kernel-cache")
    log = tmp_path / "uv.log"
    uv = tmp_path / "mock-bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {log}\n"
        "if [ \"$1\" = --version ]; then printf 'uv 0.0.0\\n'; exit 0; fi\n"
        'if [ "$1" = venv ]; then\n'
        '  mkdir -p "$2/bin"\n'
        "  printf '#!/bin/sh\\nexit 0\\n' > \"$2/bin/ft\"\n"
        '  chmod +x "$2/bin/ft"\n'
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{uv.parent}:{env['PATH']}",
            "FREETOKEN_WHEEL": str(runtime),
            "FREETOKEN_KERNEL_CACHE_WHEEL": str(cache),
            "FREETOKEN_HOME": str(tmp_path / "install-root"),
            "FREETOKEN_ENV_DIR": str(tmp_path / "environment.d"),
            "FREETOKEN_BIN_DIR": str(tmp_path / "bin"),
        }
    )

    result = subprocess.run(
        ["bash", str(INSTALL), "--yes"], check=False, capture_output=True, text=True, env=env
    )

    assert result.returncode == 0, result.stderr
    commands = log.read_text(encoding="utf-8")
    install = next(line for line in commands.splitlines() if line.startswith("pip install "))
    assert "cu126" in install
    assert str(runtime) in install
    assert str(cache) in install
    assert all(token not in install for token in ("cu130", "[accel]", "flashinfer", "sglang"))
