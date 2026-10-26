.PHONY: data data-m2 judges m1 m2 m2-smoke m2-full m2-bt m2-judge-error figures figures-m2 test test-unit

data:
	@echo "TODO: download ToxicChat / MT-Bench / Criteo via src/lee/data"

data-m2:
	python -c "from lee.data import load_mt_bench; h,g,c,r=load_mt_bench(); print(r.to_json())"

judges:
	@echo "TODO: run judge ladder; cache outputs under data/cache"

m1:
	@echo "TODO: run Milestone 1 simulation grid (after analysis_plan.md is locked)"

m2-smoke:
	python scripts/run_m2.py --smoke --R 5

m2-full:
	python scripts/run_m2.py --R 1000

m2: m2-smoke

m2-bt:
	python scripts/run_m2_bt.py

m2-judge-error:
	python scripts/run_m2_judge_error.py --R 20 --warmup 100 --samples 100

figures-m2:
	python -m lee.figures.m2

figures: figures-m2

test-unit:
	pytest -q -m "not integration"

test:
	pytest -q
