.PHONY: runner

run: install
	cd src; poetry run python Runner.py

install: pyproject.toml
	poetry install --no-root

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .ruff_cache

check:
	poetry run ruff check src

runner: run clean