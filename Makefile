.PHONY: check docs-check python-check hosted-tests

check: docs-check python-check hosted-tests
	pre-commit run --all-files

docs-check:
	python scripts/validate_docs.py
	python scripts/check_upstream_manifest.py

python-check:
	python -m compileall -q python tests scripts
	ruff check scripts
	ruff format --check scripts

hosted-tests:
	PYTHONPATH=python pytest -q tests/daemon
