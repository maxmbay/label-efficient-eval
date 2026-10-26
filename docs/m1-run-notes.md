# Milestone 1 run notes

## human_annotation verification

Dataset: `lmsys/toxic-chat`, subset `toxicchat0124`.

| Check | Result |
| --- | --- |
| Semantics | `human_annotation` is boolean; `True` marks human-labeled rows (matches paper + HF card) |
| Train / test | 5082 / 5083 |
| Pooled | 10165 |
| Human-annotated | 5654 (55.6%), matches HF card |
| Auto / non-human | 4511 (44.4%) |

Paper (arXiv 2310.17389) describes a **first-stage** process where about 61% of
prompts were never shown to humans (Perspective below ~0.037). The **released**
0124 set has more human labels than that first-stage fraction; HF card counts
are authoritative here. Population for estimation remains
`human_annotation == True`.

| Rate | Paper (context) | All rows (observed) | Human-annotated |
| --- | --- | --- | --- |
| Toxicity | 7.10% | 7.34% | 13.19% |
| Jailbreaking | 1.75% | 2.01% | 3.61% |

Observed all-rows rates match the HF card (7.33%, 2.01%). Load did **not**
stop: semantics and HF-card counts agree.

## Simulation

- Seeds: `SEED_ROOT = 20261026`; `seed = SEED_ROOT + 1_000_003 * n + r`
- Grid: $n \in \{50,100,200,400,800,1600\}$, $R = 1000$ (full) and $R = 5$ (smoke)
- Active judges: `openai_moderation_max`, `weak_keyword`
- Stubs (not scored): Llama Guard 3, open instruction-tuned

## Headline (locked definition, full $R=1000$)

At $n = 200$, PPI++ empirical 95% coverage was slightly below 95% and **outside
one Monte Carlo SE** for both active judges (toxicity and jailbreaking). Under
the locked rule, **no judge is eligible**; **no labels-saved percentage is
reported**. Widths and coverage curves are in `outputs/m1/`.

Example (toxicity, PPI++, $n=200$, $\alpha=0.05$):

- `openai_moderation_max`: coverage $0.940 \pm 0.0075$, mean width $0.0811$
- `weak_keyword`: coverage $0.935 \pm 0.0078$, mean width $0.0916$

Classical CLT at the same cell: coverage $0.943 \pm 0.0073$, mean width $0.0933$.

Reproduce:

```bash
make data && make test && make m1-full && make figures
uv run python -c "import pandas as pd; from lee.sim.headline import compute_headline; print(compute_headline(pd.read_csv('outputs/m1/metrics_full.csv')).message)"
```
