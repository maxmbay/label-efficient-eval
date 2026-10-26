.PHONY: data judges judges-open-it judges-llama m1 m1-smoke m1-full figures test lint

PYTHON ?= uv run python
R_FULL ?= 1000
R_SMOKE ?= 5

data:
	$(PYTHON) -c "from lee.data import load_toxicchat; _, h, r = load_toxicchat(strict=True); print(r.to_json())"

# Ungated open-IT judges (local MPS/CPU). No HF license required.
judges-open-it:
	$(PYTHON) scripts/run_judges.py --judge open_it_toxicity --backend local
	$(PYTHON) scripts/run_judges.py --judge open_it_jailbreaking --backend local

# Gated: requires HF license accept + token; prefer --backend modal on personal Modal.
judges-llama:
	$(PYTHON) scripts/run_judges.py --judge llama_guard3_1b --backend modal

judges-shieldgemma:
	$(PYTHON) scripts/run_judges.py --judge shieldgemma_2b --backend local

judges: judges-open-it

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
