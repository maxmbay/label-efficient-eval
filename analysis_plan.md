# Analysis plan

Committed before main simulation runs. Milestone 1 headline definition is
locked below. Do not change it after the main grid starts. Milestones 2 and 3
are stubs until their sections are locked the same way.

Working title: How many human labels does a decision need?

---

## Milestone 1: ToxicChat prevalence (LOCKED headline)

**Status: LOCKED for headline definition.** No main simulation has been run
yet. Start M1 with the in-data `openai_moderation` judge (no spend). Modal /
Hugging Face judges come later on the quality ladder.

### 1. Question

What share of user prompts is toxic, and what share are jailbreak attempts?
Estimate both from a cheap judge on every prompt plus human labels on a few.

### 2. Data and population

- Dataset: `lmsys/toxic-chat`, subset `toxicchat0124` (train and test; about
  5.08k rows each per the project brief; verify on load).
- Columns used: `conv_id`, `user_input`, `model_output`, `human_annotation`,
  `toxicity`, `jailbreaking`, `openai_moderation`.
- **Population (estimand support):** pool train and test, then restrict to rows
  with `human_annotation == True`. The estimand is prevalence **within this
  human-annotated subset**, not in all chatbot traffic.
- **Why.** Per Lin et al., ToxicChat (arXiv 2310.17389), about 61% of
  first-stage prompts were never seen by humans: a Perspective API score below
  about 0.037 auto-labeled them non-toxic. Only human-annotated rows carry
  human labels. The subset is enriched for Perspective-flagged prompts.
- **Do not** use Perspective API as a judge (it defined the selection).
- On load: verify `human_annotation` semantics and counts. If they differ from
  the paper's description, stop and report before going on.
- Paper overall rates (all rows, for context only): toxicity 7.10%, jailbreaking
  1.75%. Expect higher rates in the human-annotated subset. Report both the
  paper rates and the empirical rates in the analysis population.

### 3. Estimands

Let $Y^{\mathrm{tox}}, Y^{\mathrm{jb}} \in \{0,1\}$ be human labels on the
analysis population of size $N$.

- $\theta^{\mathrm{tox}} = \mathbb{E}[Y^{\mathrm{tox}}]$
- $\theta^{\mathrm{jb}} = \mathbb{E}[Y^{\mathrm{jb}}]$

Primary reporting: both estimands. Headline (section 6) uses the primary
estimand chosen after the load check: prefer toxicity if jailbreak prevalence
is too low for stable classical coverage at small $n$; otherwise report both
with the same headline definition applied to each.

### 4. Judges (quality ladder)

Run every judge on all $N$ prompts (cached). Scores used as $f(X)$ should be
soft where possible (P(unsafe) or calibrated score), not only hard labels.

1. **In-data OpenAI moderation (start here).** Use scores already in
   `openai_moderation`. Document the chosen summary: maximum category score,
   or a documented subset of categories. No Perspective. No API spend.
2. **Open-weight safety classifier on Modal.** Llama Guard 3 (1B or 8B) or
   ShieldGemma. Prefer P(unsafe) from token logits. Requires Hugging Face
   license accept for gated models.
3. **Zero-shot instruction-tuned open model.** Prompt with ToxicChat's own
   toxicity and jailbreak definitions; return a score or calibrated probability.
4. **Weak judge.** Keyword list or a tiny model, to show where PPI stops helping.

If any judge is calibrated or thresholded using labels, **cross-fit** (fit on
train, apply to test, and the reverse). PPI validity requires the predictor to
be independent of the labeled sample (cross-prediction-powered inference;
Zrnic and Candès).

### 5. Estimators

For a labeled subsample of size $n$ with labels $Y$ and predictions $f(X)$,
and unlabeled (judge-only) predictions $f(X_{\tilde{}})$ of size $N-n$:

```
theta_hat(lambda) = mean_n(Y) - lambda * ( mean_n(f(X)) - mean_N(f(X_tilde)) )
```

| Estimator | Role |
| --- | --- |
| Classical | Human labels only; CLT and Wilson intervals. Jailbreak is rare; report CLT coverage breaks. |
| Judge only | Mean of thresholded predictions; mean of calibrated scores. Shows bias and false confidence. |
| PPI | $\lambda = 1$ |
| PPI++ | Power-tuned $\lambda$ |
| Optional | PPI stratified by judge-score bins |

Reference: `ppi-python` (`ppi_mean_ci`). Implement estimators in-repo and
cross-check against `ppi_py` to numerical tolerance in a unit test.

Identity note for write-up: with $f$ a pre-period metric and
$\lambda = \mathrm{Cov}(Y,f)/\mathrm{Var}(f)$, this is CUPED; with $f$ an ML
prediction, CUPAC.

### 6. Headline definition (LOCKED)

**Pre-chosen label budget for the headline number:** $n = 200$.

**Definition.** Among judges whose empirical coverage of the nominal **95%**
interval is within Monte Carlo standard error of 95%, take the best judge by
mean interval width under PPI++ at $n = 200$. Let $w_{\mathrm{PPI++}}$ be that
mean width. Let $n_{\mathrm{classical}}$ be the smallest classical sample size
on the same grid whose mean width is $\le w_{\mathrm{PPI++}}$. Report

$$
\text{labels saved} = 1 - \frac{n}{n_{\mathrm{classical}}}
$$

at $n = 200$, and show the full curve of mean width vs $n$ for classical, PPI,
and PPI++ (and judge-only as a diagnostic). If no judge meets the coverage
criterion, report that failure as the result and do not invent a savings
percentage.

Also report empirical coverage at nominal 90% and 95% (with Monte Carlo SEs),
bias, and RMSE. Never round a result up.

### 7. Simulation grid

- Label budgets $n \in \{50, 100, 200, 400, 800, 1600\}$.
- For each $n$, draw $R = 1000$ random labeled subsets from the analysis
  population; remainder are judge-only.
- Fixed seeds (document in code and here when the run starts).
- Metrics: empirical coverage (90%, 95%) with Monte Carlo SE; mean interval
  width; bias; RMSE.

### 8. Figure

Mean width against $n$ (log $x$ axis), one line per estimator × judge; coverage
panel underneath with the 95% line and its Monte Carlo band.

### 9. Deliverables (after this plan is committed)

- This plan committed before the main run.
- The figure, regenerable from one Makefile target on a clean checkout.
- Write-up of about 1,000–1,500 words in `docs/`.
- README first screen with the locked headline.
- Resume line: fill $X$ only from the headline above (Max approves wording).

---

## Milestone 2: MT-Bench model comparison (STUB)

**Status: STUB.** Lock this section before the M2 grid.

- Data: `lmsys/mt_bench_human_judgments` (`human`, `gpt4_pair`).
- Unit: (question, turn, model pair). Pairwise win rates among non-tie votes,
  then Bradley-Terry across six models. Tie handling TBD in the lock.
- Inference: human-only (clustered SEs); PPI++ on win rates; hierarchical
  judge-error model in NumPyro (Dawid-Skene style with partial pooling and
  position bias).
- Same subsampling design as M1 for coverage/width comparison.
- Report inter-human agreement as the ceiling on any judge.

---

## Milestone 3: Allocation, stopping, Criteo CUPAC (STUB)

**Status: STUB.** Lock this section before M3 runs. Criteo is in scope; M2 and
M3 may proceed after M1.

1. **Allocation.** Uncertainty-weighted sampling with IPW (Zrnic and Candès,
   active statistical inference).
2. **Stopping.** Anytime-valid confidence sequences on the rectified mean;
   stopping-time distribution and error rate under repeated sampling.
3. **Criteo bridge.** ATE on `visit`; difference in means vs cross-fit ML
   outcome adjustment (CUPAC, not CUPED). Show identity with PPI form. A/A
   checks. Prefer PySpark on Databricks Free Edition; subsample and say so if
   limits bite.
