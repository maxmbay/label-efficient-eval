.PHONY: data m1 m1-smoke m1-full figures test lint

PYTHON ?= uv run python
R_FULL ?= 1000
R_SMOKE ?= 5

data:
	$(PYTHON) -c "from lee.data import load_toxicchat; _, h, r = load_toxicchat(strict=True); print(r.to_json())"

m1-smoke:
	$(PYTHON) scripts/run_m1.py --smoke --R $(R_SMOKE)

m1-full:
	$(PYTHON) scripts/run_m1.py --R $(R_FULL)

# Default m1 target: smoke (R=5). Full R=1000: `make m1-full`.
m1: m1-smoke

figures:
	$(PYTHON) -m lee.figures.m1

test:
	$(PYTHON) -m pytest -q

lint:
	uv run ruff check src tests scripts
