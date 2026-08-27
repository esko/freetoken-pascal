.PHONY: check docs-check python-check

check: docs-check python-check
	pre-commit run --all-files

docs-check:
	python scripts/validate_docs.py
	python scripts/check_upstream_manifest.py

python-check:
	@dirs=""; \
	for dir in src tests scripts; do \
		if [ -d "$$dir" ]; then dirs="$$dirs $$dir"; fi; \
	done; \
	if [ -n "$$dirs" ]; then \
		python -m compileall -q $$dirs; \
		ruff check $$dirs; \
		ruff format --check $$dirs; \
	else \
		echo "No Python source directories yet"; \
	fi
