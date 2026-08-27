.PHONY: check docs-check python-check

check: docs-check python-check
	pre-commit run --all-files

docs-check:
	python scripts/validate_docs.py
	python scripts/check_upstream_manifest.py

python-check:
	@if find src tests scripts -type f -name '*.py' -print -quit 2>/dev/null | grep -q .; then \
		python -m compileall -q src tests scripts 2>/dev/null || true; \
		ruff check src tests scripts; \
		ruff format --check src tests scripts; \
	else \
		echo "No Python source directories yet"; \
	fi
