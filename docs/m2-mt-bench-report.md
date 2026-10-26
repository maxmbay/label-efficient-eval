# Milestone 2: MT-Bench model comparison (run notes)

Formal write-up length target is 1,000–1,500 words; this file records locked
choices, verification, and simulation results without inventing a headline
savings percentage.

## Question

Which of six chat models wins, by how much, and how many human votes does that
take when an LLM judge votes on everything?

## Data

- `lmsys/mt_bench_human_judgments`: human 3355, gpt4_pair 2400 (verified).
- Six models; 80 questions; categories from FastChat `question.jsonl`.
- **Inter-human agreement (ceiling on any judge):** pairwise percent agreement
  on multi-labeled units is **65.5%** three-way (including ties) and **82.7%**
  when both votes are non-tie. Zheng et al. report human–human agreement near
  80% on non-tie comparisons; GPT-4 reaches a similar level. No LLM judge can
  be expected to beat this ceiling in a meaningful way.

## Primary pairwise estimand

- Canonical pair **(gpt-3.5-turbo, gpt-4)**.
- Decisive majority units with GPT-4 overlap: **N = 105**.
- Truth: $P(\texttt{gpt-3.5-turbo} \succ \texttt{gpt-4} \mid \text{not tie})
  = 0.3238$.
- Plan headline budget n=200 is impossible on this pair; **amended locked
  budget n=50**.

## Full PPI++ grid (R = 1000)

Nominal 95% intervals (`outputs/m2/metrics_full.csv`):

| estimator | n | coverage | coverage SE | mean width | bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| classical_clt | 50 | 0.991 | 0.003 | 0.260 | -0.001 |
| classical_clustered | 50 | 0.990 | 0.003 | 0.262 | -0.001 |
| judge_only | 50 | 0.000 | 0.000 | 0.116 | -0.143 |
| ppi | 50 | 0.984 | 0.004 | 0.312 | 0.000 |
| ppi_pp | 50 | 0.989 | 0.003 | 0.252 | -0.003 |
| classical_clt | 100 | 1.000 | 0.000 | 0.184 | -0.001 |
| classical_clustered | 100 | 1.000 | 0.000 | 0.187 | -0.001 |
| judge_only | 100 | 0.000 | 0.000 | 0.116 | -0.143 |
| ppi | 100 | 0.794 | 0.013 | 0.472 | -0.005 |
| ppi_pp | 100 | 1.000 | 0.000 | 0.183 | -0.001 |

**Headline (locked definition, honest):** no judge meets nominal 95% coverage
within Monte Carlo SE at n=50 under PPI++ (ppi_pp coverage 0.989, SE 0.003).
**Do not report a labels-saved percentage.** Judge-only intervals are badly
biased (GPT-4 favors gpt-4). PPI++ is slightly narrower than classical at n=50
but overcovers; at n=100, lambda shrinks toward 0 and PPI++ tracks classical.

Figure: `docs/m2/figure_m2.png` (copy of regenerable `outputs/m2/figure_m2.png`).

## Bradley-Terry (human majority, non-tie)

Cluster-bootstrap SEs by `question_id` (B=500). Strengths mean-zero:

| model | strength | SE |
| --- | ---: | ---: |
| gpt-4 | +1.483 | 0.160 |
| claude-v1 | +1.085 | 0.144 |
| gpt-3.5-turbo | +0.900 | 0.105 |
| vicuna-13b-v1.2 | -0.144 | 0.108 |
| alpaca-13b | -1.209 | 0.163 |
| llama-13b | -2.116 | 0.183 |

Source: `docs/m2/bradley_terry.json`.

## Hierarchical judge-error model (NumPyro)

Dawid-Skene style with marginalized latent preference, hierarchical Dirichlet
confusion (two rater classes: pooled humans + GPT-4), category logits, and
LLM position bias. NUTS runs in tens of seconds at this size (SVI not
required). Per-expert human matrices are not fit on the single-pair slice
(too few votes per expert); that is a documented simplification of the plan.

**Full-data posterior** on decisive units (200 warmup / 200 samples):

- Truth (human majority): 0.3238
- Posterior mean of $P(A \mid A \vee B)$: 0.444
- 95% central interval: [0.027, 0.963] (covers truth; wide)
- Inter-human non-tie agreement ceiling: 82.7%

**Subsample eval** (same seeds as PPI++, n=50, R=20, alpha=0.05):

| estimator | coverage | mean width | bias |
| --- | ---: | ---: | ---: |
| ppi_pp (R=1000) | 0.989 | 0.252 | -0.003 |
| judge_error_nuts (R=20) | 1.000 | 0.929 | +0.174 |

Finding: the hierarchical arm covers more often here only by being much
wider and biased high; it does not beat PPI++ on width at the locked n.
Artifacts: `docs/m2/judge_error_full.json`,
`docs/m2/metrics_judge_error.csv`.

## Reproduce

```bash
make data-m2
make m2-full          # R=1000 PPI++ grid
make m2-bt            # Bradley-Terry
make m2-judge-error   # NumPyro fit + small-R subsample eval
make figures-m2
```
