# Label-efficient evaluation

**Question.** How many human labels does a decision need when a model (LLM
judge or outcome predictor) scores every item?

Headline TBD after Milestone 1; analysis plan locked first.

See [`analysis_plan.md`](analysis_plan.md) for the locked Milestone 1
headline definition, the locked Milestone 2 design (MT-Bench model
comparison), and a stub for Milestone 3.

Milestone 2 scaffold: `src/lee/data/mt_bench.py` (loaders), pairwise PPI++
wrappers, and a NumPyro hierarchical judge-error skeleton in
`src/lee/models/judge_error.py`. No raw data in git; `make data-m2` downloads
and verifies.
