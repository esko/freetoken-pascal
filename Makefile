.PHONY: check dev-cpu dev-cuda126 docs-check env-clean env-cpu env-cuda126 expert-probe target-cpu-native target-cpu-expert-benchmark hosted-tests python-check toolchain-check

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

expert-probe:
	PYTHONPATH=python python scripts/probe_qwen38_expert.py $(PROBE_ARGS)

target-cpu-native:
	python scripts/build_target_cpu_native.py $(NATIVE_ARGS)

target-cpu-expert-benchmark:
	PYTHONPATH=python python benchmarks/bench_qwen38_real_expert.py $(BENCH_ARGS)

env-clean:
	docker image rm freetoken-pascal:cpu freetoken-pascal:cuda126 2>/dev/null || true

python-check:
	python -m compileall -q python tests scripts benchmarks
	ruff check scripts benchmarks/bench_qwen38_real_expert.py python/freetoken/moe/cpu_abi.py python/freetoken/moe/ggml_reference.py python/freetoken/moe/q4_k.py python/freetoken/moe/mixed_gemv.py python/freetoken/moe/gguf_cpu.py python/freetoken/moe/real_artifact_probe.py python/freetoken/moe/real_artifact_benchmark.py tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py tests/moe/test_q4_k_threaded_mixed.py tests/moe/test_gguf_cpu_bridge.py tests/project/test_qwen38_real_expert_probe.py tests/project/test_qwen38_real_expert_benchmark.py
	ruff format --check scripts benchmarks/bench_qwen38_real_expert.py python/freetoken/moe/cpu_abi.py python/freetoken/moe/ggml_reference.py python/freetoken/moe/q4_k.py python/freetoken/moe/mixed_gemv.py python/freetoken/moe/gguf_cpu.py python/freetoken/moe/real_artifact_probe.py python/freetoken/moe/real_artifact_benchmark.py tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py tests/moe/test_q4_k_threaded_mixed.py tests/moe/test_gguf_cpu_bridge.py tests/project/test_qwen38_real_expert_probe.py tests/project/test_qwen38_real_expert_benchmark.py

hosted-tests:
	PYTHONPATH=python pytest -q tests/project tests/daemon tests/moe/test_cpu_abi.py tests/moe/test_ggml_reference.py tests/moe/test_q4_k_mixed_reference.py tests/moe/test_q4_k.py tests/moe/test_mixed_gemv.py tests/moe/test_q4_k_threaded_mixed.py tests/moe/test_gguf_cpu_bridge.py
