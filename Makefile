.PHONY: install format lint test sweep clean

install:
	uv venv .venv --python 3.13
	uv pip install -e ".[dev]"

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests

test:
	uv run pytest tests -v

sweep-synthetic:
	uv run sweep-synthetic --output results/synthetic.parquet

sweep-population:
	uv run sweep-population --output results/population.parquet

sweep: sweep-synthetic sweep-population

clean:
	rm -rf .venv build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
