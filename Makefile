.PHONY: data data-m2 judges m1 m2 figures test test-unit

data:
	@echo "TODO: download ToxicChat / MT-Bench / Criteo via src/lee/data"

data-m2:
	python -c "from lee.data import load_mt_bench; h,g,c,r=load_mt_bench(); print(r.to_json())"

judges:
	@echo "TODO: run judge ladder; cache outputs under data/cache"

m1:
	@echo "TODO: run Milestone 1 simulation grid (after analysis_plan.md is locked)"

m2:
	@echo "TODO: run Milestone 2 simulation grid (after analysis_plan.md M2 is locked and load verifies)"

figures:
	@echo "TODO: regenerate milestone figures from cached sim outputs"

test-unit:
	pytest -q -m "not integration"

test:
	pytest -q
