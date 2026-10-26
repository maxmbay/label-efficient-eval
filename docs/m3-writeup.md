# Milestone 3: allocation, stopping, and the A/B bridge

How should human labels be spent, when can labeling stop, and how does the same
control-variate idea appear in an A/B test? This milestone answers those three
questions with locked designs in `analysis_plan.md`. It is an applied study of
published methods, not a new estimator.

## Prior work

Prediction-powered inference (Angelopoulos et al., PPI and PPI++) estimates a
population mean from a small labeled sample and a large pool of machine
predictions. Active statistical inference (Zrnic and Candès, ICML 2024) chooses
which points to label using a model and corrects with inverse-propensity /
AIPW weights. Anytime-valid inference for means via betting is due to
Waudby-Smith and Ramdas (JRSS-B 2024). Anytime-valid PPI confidence sequences
appear in Kilian, Cortinovis, and Caron (arXiv 2505.18000; NeurIPS 2025);
Csillag et al. (ICML 2025) study prediction-powered e-values. CUPED uses a
pre-period covariate; when the covariate is an ML prediction of the outcome,
the same algebra is CUPAC.

## Shared form

```
theta_hat(λ) = mean_n(Y) - λ ( mean_n(f(X)) - mean_N(f(X_tilde)) )
```

With $f$ a pre-period metric this is CUPED. With $f$ an ML outcome model and no
pre-period this is CUPAC. With $f$ a judge score this is PPI. Milestone 3 uses
that identity in three settings.

## Allocation

On the ToxicChat human-annotated toxicity population, with the in-data OpenAI
moderation max score as $f$, we compare uniform labeling to uncertainty-weighted
propensities $\pi_i \propto f_i(1-f_i)$, clipped and rescaled to a target budget
$n_b$. The estimator is the AIPW mean of Zrnic and Candès. At equal expected
budget we report coverage and mean interval width. The locked headline is the
relative width reduction of active vs uniform at $n_b = 200$ when both cover
near 95%.

## Stopping

Labels arrive in a random order. Each step forms the PPI rectifier
$Z_t = Y_t - (f_t - \bar f_N)$ with $\lambda = 1$, mapped into $[0,1]$ by a known
affine transform, and feeds a hedged betting confidence sequence. Labeling
stops at the first $t$ whose interval lies entirely above $\tau = 0.10$, or at
$t_{\max} = 800$. We report the stopping-time distribution and the empirical
error rate for false clear-above decisions. This is a betting-CS stopping rule
on the rectifier stream; it cites, and does not reimplement, the asymptotic
anytime-valid PPI procedures of Kilian et al.

## Criteo CUPAC bridge

Criteo Uplift v2.1 provides randomized treatment, features `f0`–`f11`, and
outcomes `visit` / `conversion`. The estimand is the ATE on `visit`. There is
no pre-period metric, so outcome adjustment is CUPAC, not CUPED.

Difference in means is the baseline. CUPAC fits a cross-fit gradient boosting
model $\hat m(x)$ for `visit`, forms $Y - \lambda(\hat m(X) - \overline{\hat m})$,
and takes the arm difference. Algebraically, for a mean,

$\mathrm{mean}(Y) - \lambda(\mathrm{mean}(f_{\mathrm{lab}}) - \mathrm{mean}(f_{\mathrm{all}}))$

is exactly the PPI point estimate; the ATE case applies the same adjustment
inside each arm (or to the contrast). The code checks the mean identity to
numerical tolerance. We report the variance ratio (equivalent sample-size
factor) and A/A false-positive rates on control splits.

**Compute.** The preferred path is PySpark on Databricks Free Edition
(`notebooks/criteo_spark.py`). When Databricks is unavailable, the local path
downloads the Criteo gzip (not committed), reservoir-samples 500,000 rows with
seed `20261026`, and runs pandas / scikit-learn. Load reports record the file
row count against the Criteo card (25,309,483) and the TFDS train size
(13,979,592).

## What is new here

Nothing methodologically new: the contribution is a locked, reproducible
application that ties LLM-judge labeling, anytime-valid stopping, and an A/B
variance-reduction identity under one estimator form, with honest coverage and
A/A checks.

## Reproducibility

```bash
make m3-smoke    # wiring
make m3-full     # locked grids
make figures-m3
```

Figures and metrics are written under `outputs/m3/`. See `docs/m3-run-notes.md`
for the numbers from the run that produced the PR.
