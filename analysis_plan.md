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

## Milestone 2: MT-Bench model comparison (LOCKED plan; headline after load)

**Status: LOCKED for design.** Headline number is defined below but not yet
filled; commit this section before the main M2 simulation grid. No main M2
run has been started. Scaffold loaders and model stubs first; open-weight
re-judging and the full grid come after data verification.

### 1. Question

Which of six chat models wins, by how much, and how many human votes does that
take when an LLM judge votes on everything?

### 2. Data and population

- Dataset: `lmsys/mt_bench_human_judgments` (CC BY 4.0).
- Splits: `human` (about 3.3k expert pairwise votes) and `gpt4_pair` (about
  2.4k GPT-4 judge votes). Verify counts and columns on load; stop and report
  if they disagree with the dataset card by more than a documented tolerance.
- Columns expected: `question_id`, `model_a`, `model_b`, `winner`
  (`model_a`, `model_b`, `tie`), `judge`, `conversation_a`, `conversation_b`,
  `turn`.
- Six models: GPT-4, GPT-3.5-turbo, Claude-v1, Vicuna-13B, Alpaca-13B,
  LLaMA-13B. Eighty questions; several human judges per item.
- Question categories (writing, roleplay, reasoning, math, coding, extraction,
  STEM, humanities) from the MT-Bench question file in lm-sys/FastChat:
  `fastchat/llm_judge/data/mt_bench/question.jsonl`. Verify the path on fetch;
  cache a local copy under `data/cache/` (gitignored), never commit raw data.
- **Unit of analysis:** `(question_id, turn, model_a, model_b)` after
  canonicalizing pair order (lexicographic on model names; flip `winner`
  when swapping). Multiple human votes on the same unit are retained for
  agreement analysis and for the hierarchical model; PPI++ pairwise win-rate
  simulations first collapse to one label per unit via majority vote among
  non-tie human votes (ties in the majority: drop that unit from the primary
  estimand and report the drop count).

### 3. Tie handling (LOCKED)

1. **Primary pairwise estimand (non-tie conditional).** For a fixed ordered
   pair $(A,B)$, let $Y=1$ if the (majority) human winner is $A$, $Y=0$ if
   $B$, after discarding votes with `winner == tie`. The estimand is
   $P(A \succ B \mid \text{not tie})$ among analysis units that have a
   decisive majority. Report the human tie rate separately.
2. **Sensitivity.** Half-credit: treat each tie as $Y=0.5$ when forming a
   win rate (Bradley-Terry / soft labels). Report beside the primary
   estimand; do not use it for the locked headline unless primary coverage
   fails and the plan is explicitly amended.
3. **Hierarchical judge-error model.** Latent true preference
   $z \in \{A, B, \mathrm{tie}\}$ per unit; ties are a third class, not
   discarded inside that model.

Do not silently drop ties without reporting rates. Do not code ties as wins
for either side in the primary estimand.

### 4. Estimands

Let $\mathcal{P}$ be the set of well-covered unordered model pairs (both
directions observed enough after load; document the minimum vote threshold
chosen after the load report, suggested floor: at least 50 non-tie human
majority units).

- **Pairwise win rate (primary for PPI++):** for each ordered pair $(A,B)$ in
  $\mathcal{P}$,
  $\theta_{AB} = P(A \succ B \mid \text{not tie})$ on the majority-collapsed
  human labels.
- **Bradley-Terry strengths (secondary):** strengths $\beta_m$ for the six
  models fit on non-tie human majorities (and separately under half-credit
  sensitivity). Report win-rate contrasts implied by $\beta$, not only the
  latent scale.
- **Held-out human-majority target for hierarchical evaluation:** same
  $\theta_{AB}$ (or BT contrast) on units held out of the human subsample,
  so coverage is comparable to PPI++.

Headline uses one pre-chosen well-covered pair after the load check (see
section 8). Prefer a pair with GPT-4 as one side if coverage is adequate;
document the choice in the load report and do not change it after the main
grid starts.

### 5. Judges

1. **In-data GPT-4 pairwise votes** (`gpt4_pair` split). Align to human units
   by `(question_id, turn, canonical pair)`. Soft score for PPI: map
   `winner` to $\{0,1\}$ on non-ties; for units where GPT-4 ties, use $0.5$
   as $f(X)$ (documented). Measure position bias when both orders exist.
2. **Open-weight re-judges (later).** One or two open-weight judges scoring
   from `conversation_a` / `conversation_b` with FastChat's pairwise judge
   prompt, both orders. Cache outputs under `data/cache/` (gitignored).
3. **Humans as a rater class.** Individual human judges enter the hierarchical
   model; report inter-human agreement as the ceiling on any LLM judge
   (pairwise percent agreement and Cohen's $\kappa$ on overlapping units,
   non-tie and three-way).

If a judge is calibrated using labels, cross-fit. PPI requires the predictor
to be independent of the labeled sample used for the rectifier.

### 6. Estimators

| Estimator | Role |
| --- | --- |
| Human only | Mean win rate on the labeled subsample; SEs clustered by `question_id` and respecting paired structure (Anthropic, "Adding Error Bars to Evals", arXiv 2411.00640). |
| Judge only | Mean of GPT-4 (or open-weight) mapped scores on all units. Bias diagnostic. |
| PPI | $\lambda = 1$ on the binary (or half-credit) win indicator. |
| PPI++ | Power-tuned $\lambda$, same scalar-mean form as Milestone 1. |
| Hierarchical judge error | NumPyro model in `src/lee/models/judge_error.py` (section 7). |

PPI algebra is the Milestone 1 mean estimator with $Y$ the human win indicator
and $f$ the judge score on the same units. Reference: `ppi-python`;
implement in-repo and cross-check.

### 7. Hierarchical judge-error model (outline; implement after stubs)

Dawid-Skene style with partial pooling. Applied study, not a new method.

- **Latent preference** $z_i \in \{A, B, \mathrm{tie}\}$ per analysis unit $i$;
  enumerate (or marginalize) in the likelihood.
- **Confusion matrices.** Each judge $j$ has a category-specific confusion
  matrix over $\{A,B,\mathrm{tie}\}$, partially pooled toward that judge's
  overall matrix (logistic-normal or hierarchical Dirichlet). Categories from
  the MT-Bench question file.
- **Position bias.** Additive (on the appropriate scale) term for LLM judges
  when $A$ is presented first vs second.
- **Humans.** One rater class with individual deviations from the class mean
  confusion matrix.
- **Inference.** NUTS first at this size; SVI only if needed.
- **Evaluation.** Fit with $n$ human labels plus all judge labels; compare
  posterior intervals for $\theta_{AB}$ (or BT contrasts) to the held-out
  human-majority estimand on coverage and width under the same subsampling
  grid as PPI++. Beating PPI++ is not required; narrower intervals with worse
  coverage under misspecification is a valid finding if reported cleanly.

### 8. Headline definition (LOCKED form; pair chosen after load)

**Pre-chosen label budget:** $n = 200$ human-labeled units (majority-
collapsed), same spirit as Milestone 1.

**Definition.** For the pre-chosen pair $(A,B)$, among judges whose empirical
coverage of the nominal **95%** PPI++ interval is within Monte Carlo SE of
95%, take the best judge by mean interval width at $n = 200$. Let
$w_{\mathrm{PPI++}}$ be that width. Let $n_{\mathrm{classical}}$ be the
smallest classical (human-only) size on the grid with mean width
$\le w_{\mathrm{PPI++}}$. Report

$$
\text{labels saved} = 1 - \frac{n}{n_{\mathrm{classical}}}
$$

at $n = 200$, and the mean-width curve vs $n$. If no judge meets coverage,
report that failure and do not invent a savings percentage. Also report
inter-human agreement as an interpretive ceiling. Never round a result up.

Hierarchical-model intervals are reported beside PPI++ on the same figure or
a companion panel; they do not replace the PPI++ headline unless a later
locked amendment says so.

### 9. Simulation grid (mirrors Milestone 1)

- Label budgets $n \in \{50, 100, 200, 400, 800, 1600\}$ (cap at the number of
  decisive majority units for the chosen pair; skip or mark $n$ above that
  cap).
- For each $n$, draw $R = 1000$ random labeled subsets from the pair's
  analysis units; remainder are judge-only for PPI.
- Fixed seeds: `SEED_ROOT = 20261026`. Subset draw $r$ at budget $n$ uses
  `seed = SEED_ROOT + 2_000_003 * n + r` (M2 offset; distinct from M1).
- Smoke: $R = 5$. Full: $R = 1000$. Headline only from a full locked run.
- Metrics: empirical coverage (90%, 95%) with Monte Carlo SE; mean width;
  bias; RMSE. For hierarchical fits, same coverage/width on posterior
  intervals vs held-out human majority (compute budget may force a smaller
  $R$ for the Bayes arm; if so, document and still report Monte Carlo SEs).

### 10. Figure and deliverables

- Mean width vs $n$ (log $x$), classical / PPI / PPI++ / judge-only; coverage
  panel with 95% line and Monte Carlo band. Optional companion panel for
  hierarchical posterior width and coverage.
- This plan committed before the main M2 grid.
- Write-up about 1,000–1,500 words in `docs/`.
- README: add model-win-rate clause to the resume line only after M2 headline
  exists and Max approves.

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
