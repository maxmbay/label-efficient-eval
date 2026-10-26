.PHONY: data data-criteo m3 m3-smoke m3-full m3-allocation m3-stopping m3-criteo figures figures-m3 test lint

PYTHON ?= uv run python
R_FULL ?= 500
R_SMOKE ?= 5

data:
	$(PYTHON) -c "from lee.data import load_toxicchat; _, h, r = load_toxicchat(strict=True); print(r.to_json())"

data-criteo:
	$(PYTHON) -c "from lee.data import load_criteo; _, r = load_criteo(); print(r.to_json())"

m3-allocation:
	$(PYTHON) scripts/run_m3_allocation.py --R $(R_FULL)

m3-stopping:
	$(PYTHON) scripts/run_m3_stopping.py --R $(R_FULL)

m3-criteo:
	$(PYTHON) scripts/run_m3_criteo.py

m3-smoke:
	$(PYTHON) scripts/run_m3.py --smoke --R $(R_SMOKE)

m3-full:
	$(PYTHON) scripts/run_m3.py --R $(R_FULL)

# Default m3 target: smoke. Full grids: `make m3-full`.
m3: m3-smoke

figures-m3:
	$(PYTHON) -m lee.figures.m3 --tag full

figures: figures-m3

test:
	$(PYTHON) -m pytest -q

lint:
	uv run ruff check src tests scripts
