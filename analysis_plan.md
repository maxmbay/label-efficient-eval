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

## Milestone 3: Allocation, stopping, Criteo CUPAC (LOCKED)

**Status: LOCKED for design and reporting definitions.** No main M3 runs have
been executed yet. Commit this section before `make m3` / `make m3-full`.
Smoke paths (`make m3-smoke`) may validate wiring only; headline numbers come
from the locked full grids below.

M3 has three independent studies that share the PPI / control-variate form.
Allocation and stopping reuse the ToxicChat human-annotated toxicity population
and the in-data OpenAI moderation max-score judge from Milestone 1 (no new
judge spend). The Criteo bridge is a separate A/B dataset.

### 1. Shared estimator form

```
theta_hat(lambda) = mean_n(Y) - lambda * ( mean_n(f(X)) - mean_N(f(X_tilde)) )
```

With $f$ a pre-period metric and $\lambda = \mathrm{Cov}(Y,f)/\mathrm{Var}(f)$,
this is CUPED. With $f$ an ML prediction of the outcome (no pre-period), this is
CUPAC. With $f$ a judge score on unlabeled items, this is PPI / PPI++. M3 uses
the name **CUPAC** for the Criteo outcome model (no pre-period metric exists).

### 2. Allocation (active labeling)

**Question.** At a fixed human-label budget, does uncertainty-weighted sampling
with inverse-probability weighting shrink interval width relative to uniform
sampling, while keeping coverage?

**Population and judge.** Same as M1: ToxicChat `toxicchat0124` human-annotated
rows; estimand $\theta^{\mathrm{tox}} = \mathbb{E}[Y^{\mathrm{tox}}]$;
judge $f$ = OpenAI moderation max category score.

**Sampling rule (Zrnic and Candès, Active Statistical Inference, ICML 2024).**
For each item $i$ with judge score $f_i \in [0,1]$, define Bernoulli variance
proxy $u_i = f_i(1-f_i)$ (uncertainty). With target expected label count
$n_b$ out of $N$ items and floor propensity $\pi_{\min} = 0.01$:

$$
\pi_i \propto \max(u_i, \varepsilon_u),\qquad
\mathbb{E}[\pi(X)] = n_b / N,\qquad
\pi_i = \mathrm{clip}(\pi_i, \pi_{\min}, 1),
$$

then rescale so the mean propensity equals $n_b/N$ when feasible under the
clip. Draw $\xi_i \sim \mathrm{Bernoulli}(\pi_i)$ independently (fixed seed).
If the realized label count differs from $n_b$, do not post-hoc resample; report
the realized $n$ and compare estimators at equal expected budget $n_b$.

**Estimator (AIPW / active inference mean).**

$$
\hat\theta_\pi = \frac{1}{N}\sum_{i=1}^{N}
\left( f_i + \frac{\xi_i}{\pi_i}(Y_i - f_i) \right).
$$

Uniform baseline: $\pi_i = n_b/N$ for all $i$ (reduces to the PPI mean with
$\lambda=1$ when all labeled residuals are observed under that rule).

**Intervals.** Wald intervals from the empirical variance of the AIPW summands
(Zrnic and Candès), nominal $\alpha \in \{0.05, 0.10\}$.

**Simulation grid (LOCKED).**

- Budgets $n_b \in \{50, 100, 200, 400, 800\}$.
- Replicates $R = 500$ (full); smoke $R = 5$.
- Seeds: `SEED_ROOT = 20261026`. Replicate $r$ at budget $n_b$ uses
  `seed = SEED_ROOT + 3_000_007 * n_b + r`.
- Metrics: empirical coverage (90%, 95%) with MC SE; mean interval width; bias;
  RMSE; mean realized labeled count.

**Headline (allocation).** At $n_b = 200$ and nominal 95%: relative mean-width
reduction of active AIPW vs uniform AIPW,
$(w_{\mathrm{unif}} - w_{\mathrm{active}}) / w_{\mathrm{unif}}$, provided both
have coverage within MC SE of 95%. If either fails coverage, report that as
the result and do not invent a savings percentage.

### 3. Stopping (anytime-valid CS)

**Question.** If labels arrive sequentially and we stop when a confidence
sequence clears a decision threshold, what is the stopping-time distribution
and the error rate under repeated sampling?

**Literature (cite; do not claim novelty).** Fixed-time PPI is Angelopoulos et
al. Anytime-valid PPI confidence sequences: Kilian, Cortinovis, and Caron,
"Anytime-valid, Bayes-assisted, Prediction-Powered Inference" (arXiv
2505.18000; NeurIPS 2025), using Ville / method-of-mixtures asymptotic
confidence sequences. Related: Csillag et al., prediction-powered e-values
(ICML 2025). Betting confidence sequences for bounded means: Waudby-Smith and
Ramdas, JRSS-B 2024. This milestone applies a **betting CS to the PPI
rectifier stream** (labeled residuals after subtracting a fixed judge), which
is a practical stopping rule aligned with that literature; it is not a
reimplementation of the full asymptotic PPI CS of arXiv 2505.18000.

**Rectified observations.** Fix the full-population judge mean
$\bar f_N = N^{-1}\sum_j f_j$ (known under the M1 setup where every item has a
judge score). For a random permutation of the population (stream order), the
$t$-th labeled observation contributes the bounded proxy

$$
Z_t = Y_{\sigma(t)} - \lambda\bigl(f_{\sigma(t)} - \bar f_N\bigr),
$$

with $\lambda = 1$ (PPI rectifier). Clip $Z_t$ linearly into $[0,1]$ for the
betting CS using known bounds derived from $Y\in\{0,1\}$ and $f\in[0,1]$
(document the affine map). Run the hedged capital process of Waudby-Smith and
Ramdas; the CS at time $t$ is $\{m : K_t^{\pm}(m) < 1/\alpha\}$.

**Decision rule (LOCKED).** Estimand: toxicity prevalence. Threshold
$\tau = 0.10$ (one-sided clear-above: stop at the first $t$ with
$\mathrm{CS}_t.lower > \tau$). Cap at $t_{\max} = 800$. Alpha for the CS:
$\alpha = 0.05$. Also report a two-sided "clear away from $\tau$" variant
(interval excludes $\tau$) as secondary.

**Simulation grid (LOCKED).**

- $R = 500$ random stream orders (full); smoke $R = 5$.
- Seeds: `seed = SEED_ROOT + 4_000_009 * r`.
- Metrics: stopping time $T$ (or $t_{\max}$ if never clears); mean / median /
  quantiles of $T$; empirical error rate = fraction of runs that stop with a
  false decision relative to the true $\theta^{\mathrm{tox}}$ (stop claiming
  $\theta > \tau$ when truth $\le \tau$, and the symmetric miss if using
  clear-below); fraction that hit $t_{\max}$ without stopping.

**Headline (stopping).** Median stopping time under the clear-above rule, plus
the empirical error rate at $\alpha = 0.05$. Report honestly if the error rate
exceeds $\alpha$ (betting CS is conservative for optional stopping of the
interval, but the decision mapping to a one-sided claim can still err when
truth is near $\tau$).

### 4. Criteo bridge (CUPAC ≡ PPI)

**Question.** For ATE of treatment on `visit`, how much does cross-fit ML
outcome adjustment (CUPAC) reduce variance relative to difference-in-means, and
does CUPAC match the PPI algebra?

**Data.** Criteo Uplift v2.1 (`http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz`;
also TFDS `criteo`). Columns: `f0`–`f11`, `treatment`, `visit`, `conversion`,
`exposure`. TFDS lists 13,979,592 train rows while the Criteo card says
25,309,483; **verify on load** and record both the file row count and any TFDS
split size in the load report. Do not commit raw data.

**Estimand.** $\tau = \mathbb{E}[Y(1) - Y(0)]$ with $Y =$ `visit`, under the
randomized treatment in the release (report empirical treatment rate).

**Estimators.**

1. Difference in means (DIM).
2. CUPAC: cross-fit outcome model $\hat m(x)$ trained on features `f0`–`f11`
   (and not on treatment in the base model), predicting $Y$; adjusted outcome
   $Y - \hat m(X)$ (or the AIPW/interaction form documented in code); ATE =
   difference in means of adjusted outcomes. Default model: gradient boosting
   (`HistGradientBoostingClassifier` / regressor on $[0,1]$ scores) with
   $K = 5$ folds. Name: **CUPAC**, not CUPED (no pre-period).

**Algebraic identity (LOCKED statement to show in the write-up).** For a single
population mean, CUPAC with predictor $f=\hat m$ and
$\lambda = \mathrm{Cov}(Y,f)/\mathrm{Var}(f)$ is identical to the PPI++ mean
formula above. For ATE under randomization, applying the same adjustment
within treated and control (or to the contrast) is the two-arm analogue:
DIM on $Y - \lambda f(X)$ equals PPI-style rectification of the arm means.
Show this algebraically in `docs/m3-writeup.md` and numerically by checking
that the CUPAC point estimate matches the explicit PPI-form reconstruction
on the same fold predictions within tolerance $10^{-10}$ relative error (or
absolute $10^{-12}$).

**Compute path.** Prefer PySpark on Databricks Free Edition
(`notebooks/criteo_spark.py`). If Databricks is unavailable or limits bite,
run a **documented local subsample**: draw `N_sub = 500_000` rows with
`seed = SEED_ROOT` from the full CSV (or the full file if memory allows), and
state the subsample size in README and run notes. Optional local PySpark on
the same subsample. Full-file Spark remains the target when a cluster is
available.

**A/A check (LOCKED).** Restrict to control rows (`treatment == 0`). Randomly
split into fake A/A arms with equal probability (`seed = SEED_ROOT + 17`).
Test $H_0$: ATE $= 0$ at $\alpha = 0.05$ with DIM and with CUPAC. Repeat
$R_{\mathrm{AA}} = 200$ splits (smoke: 20). Report empirical false-positive
rate; expect near 5%.

**Metrics (main A/B on subsample or full).** Point ATE; SE; 95% CI; variance
ratio $\mathrm{Var}_{\mathrm{DIM}} / \mathrm{Var}_{\mathrm{CUPAC}}$; equivalent
sample-size factor (same ratio under asymptotic variance); CUPAC≡PPI numerical
check; A/A FPR.

**Headline (Criteo).** Variance-reduction factor (or equivalent $n$) of CUPAC
vs DIM for ATE on `visit`, plus A/A FPR for both estimators.

### 5. Seeds, deliverables, honesty

- Global `SEED_ROOT = 20261026` for all M3 studies.
- Deliverables: this locked plan; regenerable figure(s) under `outputs/m3/`;
  write-up `docs/m3-writeup.md` (~1,000–1,500 words); run notes with Databricks
  status and any subsample; Makefile targets `m3-smoke`, `m3-full`, `m3`;
  unit tests for AIPW unbiasedness on synthetic data, betting CS coverage on
  a known mean, CUPAC≡PPI identity, and A/A smoke.
- Report nulls and coverage breaks. Never round a result up. No raw Criteo or
  ToxicChat rows in git.
