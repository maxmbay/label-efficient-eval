# Milestone 3 run notes

Branch: `m3/allocation-stopping-criteo`. Analysis plan locked in commit
before these runs. Numbers below are from the locked full grids on this
machine; they are not rounded up.

## Environment

- Self-hosted Mac; worktree `/Users/maxwellbay/projects/label-efficient-eval-m3`.
- Databricks Free Edition: **not configured** on this machine (`databricks`
  CLI absent). Criteo used the local pandas path with a 500,000-row reservoir
  subsample (seed `20261026`). Spark notebook source is in
  `notebooks/criteo_spark.py` for a cluster run later.
- ToxicChat: reused on-disk cache from the M1 worktree via
  `data/raw/toxicchat0124` (gitignored).

## Allocation (R = 500)

Headline status at $n_b = 200$, $\alpha = 0.05$: **coverage_fail**.

| method | coverage | coverage SE | mean width | mean n labeled |
| --- | ---: | ---: | ---: | ---: |
| uniform | 0.962 | 0.0086 | 0.0862 | 200.2 |
| active | 0.888 | 0.0141 | 0.0975 | 206.5 |

Active AIPW undercovered the nominal 95% interval (0.888 vs 0.95). Uniform
covered. Active mean width was also larger than uniform at this budget, so no
label-savings percentage is reported (per the locked plan). Full grid CSV:
`outputs/m3/allocation_full.csv`.

## Stopping (R = 500)

Truth toxicity prevalence in the human-annotated population: 0.1319.
Threshold $\tau = 0.10$ (clear-above). Cap $t_{\max} = 800$.

| metric | value |
| --- | ---: |
| median stopping time | 735 |
| mean stopping time | (see meta) |
| fraction hitting cap | 0.458 |
| empirical error rate (false clear-above) | 0.000 |

Many streams never clear the threshold by 800 labels; the betting CS is
conservative. Error rate for false clear-above claims was 0 under these
draws (truth lies above $\tau$). Meta: `outputs/m3/stopping_meta_full.json`.

## Criteo CUPAC bridge

- Downloaded from the Hugging Face `criteo/criteo-uplift` mirror (Criteo
  `go.criteo.net` redirect failed DNS / 404 on this host).
- **File row count: 13,979,592**, matching the TFDS `criteo` train split, **not**
  the Criteo AI Lab card figure of 25,309,483. Recorded in the load report.
- Used subsample: 500,000 rows, seed 20261026.
- Empirical rates on the subsample: treatment 0.8505, visit 0.0472,
  conversion 0.0030.

| estimator | ATE (visit) | SE | asymptotic var |
| --- | ---: | ---: | ---: |
| DIM | 0.010629 | (see JSON) | (see JSON) |
| CUPAC | 0.006984 | (see JSON) | (see JSON) |

- Variance ratio $\mathrm{Var}_{DIM}/\mathrm{Var}_{CUPAC}$ = **1.434**
  (equivalent sample-size factor ≈ 1.43×).
- CUPAC ≡ PPI numerical identity: **pass** (abs err 0).
- A/A on control (R = 200, up to 100k control rows, outcome model fit once):
  DIM FPR = **0.045**, CUPAC FPR = **0.065** (nominal 0.05).

Outputs: `outputs/m3/criteo_full.json`, `outputs/m3/figure_m3.png`.

## Commands

```bash
make m3-smoke
make m3-full
make figures-m3
make test
```
