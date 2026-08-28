.PHONY: check dev-cpu dev-cuda126 docs-check env-clean env-cpu env-cuda126 hosted-tests python-check toolchain-check

check: docs-check python-check hosted-tests
	pre-commit run --all-files

docs-check:
	python scripts/validate_docs.py
	python scripts/check_upstream_manifest.py
	python scripts/check_toolchain.py

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
	ruff check scripts
	ruff format --check scripts

hosted-tests:
	PYTHONPATH=python pytest -q tests/project tests/daemon
