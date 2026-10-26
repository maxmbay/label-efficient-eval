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

## Judges

| Judge | Status |
| --- | --- |
| `openai_moderation_max` | Active (in-data) |
| `weak_keyword` | Active |
| `open_it_toxicity_crossfit` / `open_it_jailbreaking_crossfit` | Active; Qwen2.5-0.5B-Instruct local MPS; ToxicChat definitions; cross-fit Platt |
| `llama_guard3_1b` | **Blocked**: no HF token / license accept / Modal auth on this machine |
| `shieldgemma_2b` | **Blocked**: same |

See [`docs/judge-auth.md`](judge-auth.md) for the exact commands Max must run.

## Simulation

- Seeds: `SEED_ROOT = 20261026`; `seed = SEED_ROOT + 1_000_003 * n + r`
- Grid: $n \in \{50,100,200,400,800,1600\}$, $R = 1000$
- Outputs: `outputs/m1/metrics_full.csv`, `figure_m1.png`

## Headline (locked definition, full $R=1000$)

At $n = 200$, PPI++ empirical 95% coverage was still **outside one Monte Carlo
SE of 95%** for every active judge (toxicity and jailbreaking). Under the
locked rule, **no judge is eligible**; **no labels-saved percentage is
reported**.

Toxicity, PPI++, $n=200$, $\alpha=0.05$:

| Judge | Coverage | Mean width |
| --- | --- | --- |
| openai_moderation_max | $0.940 \pm 0.0075$ | 0.0811 |
| open_it_toxicity_crossfit | $0.942 \pm 0.0074$ | 0.0922 |
| weak_keyword | $0.935 \pm 0.0078$ | 0.0916 |

Open-IT is closest to the coverage gate but does not clear it. Llama Guard /
ShieldGemma remain the next rung once Max authenticates.

Reproduce:

```bash
make data && make judges-open-it && make test && make m1-full && make figures
# gated (after auth):
make judges-llama
```
