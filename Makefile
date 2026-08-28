.PHONY: check dev-cpu dev-cuda126 docs-check env-clean env-cpu env-cuda126 hosted-tests python-check toolchain-check

check: docs-check python-check hosted-tests
	pre-commit run --all-files

docs-check:
	python scripts/validate_docs.py
	python scripts/check_upstream_manifest.py
	python scripts/check_toolchain.py
	python scripts/check_lint_baseline.py
	python scripts/validate_evidence.py

toolchain-check:
	python scripts/check_toolchain.py

env-cpu:
	docker build --file containers/cpu.Dockerfile --tag freetoken-pascal:cpu .

env-cuda126:
	docker build --file containers/cuda126.Dockerfile --tag freetoken-pascal:cuda126 .

dev-cpu: env-cpu
	docker run --rm -it --volume "$(PWD):/workspace/freetoken-pascal" freetoken-pascal:cpu

dev-cuda126: env-cuda126
	docker run --rm -it --volume "$(PWD):/workspace/freetoken-pascal" freetoken-pascal:cuda126

env-clean:
	docker image rm freetoken-pascal:cpu freetoken-pascal:cuda126 2>/dev/null || true

python-check:
	python -m compileall -q python tests scripts
	ruff check scripts python/freetoken/moe/cpu_abi.py python/freetoken/moe/ggml_reference.py python/freetoken/moe/q4_k.py python/freetoken/moe/mixed_gemv.py tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py
	ruff format --check scripts python/freetoken/moe/cpu_abi.py python/freetoken/moe/ggml_reference.py python/freetoken/moe/q4_k.py python/freetoken/moe/mixed_gemv.py tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py

hosted-tests:
	PYTHONPATH=python pytest -q tests/project tests/daemon tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py
