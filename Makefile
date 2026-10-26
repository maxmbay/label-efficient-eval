.PHONY: data judges m1 figures test

data:
	@echo "TODO: download ToxicChat / MT-Bench / Criteo via src/lee/data"

judges:
	@echo "TODO: run judge ladder; cache outputs under data/cache"

m1:
	@echo "TODO: run Milestone 1 simulation grid (after analysis_plan.md is locked)"

figures:
	@echo "TODO: regenerate milestone figures from cached sim outputs"

test:
	pytest -q
