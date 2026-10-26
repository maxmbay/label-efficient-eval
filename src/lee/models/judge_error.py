"""Hierarchical judge-error model for MT-Bench (NumPyro).

Dawid-Skene style applied model (not a new method). Matches analysis_plan.md
Milestone 2 §7, simplified to run at MT-Bench size with NUTS:

- Latent true preference z_i in {A, B, tie} per unit, marginalized in the
  likelihood (mixture over confusion rows).
- Each judge has a confusion matrix with rows partially pooled toward a
  shared profile via a hierarchical Dirichlet.
- LLM judges get a position-bias logit shift when model_a was presented
  second (``a_presented_first == 0``).
- Human judges share a class mean with light individual deviations
  (per-judge Dirichlet concentration toward the human-class profile).

Evaluation helper compares posterior intervals for the non-tie win rate
P(A | not tie) to the population human-majority estimand under the same
subsampling grid as PPI++.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Outcome / latent classes (fixed order for confusion matrices).
OUTCOME_CLASSES: tuple[str, ...] = ("model_a", "model_b", "tie")
N_OUTCOMES: int = len(OUTCOME_CLASSES)


@dataclass(frozen=True)
class JudgeErrorData:
    """Integer-coded arrays for the hierarchical judge-error model."""

    unit_index: np.ndarray
    judge_index: np.ndarray
    category_index: np.ndarray
    label_index: np.ndarray
    a_presented_first: np.ndarray
    is_human: np.ndarray
    n_units: int
    n_judges: int
    n_categories: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class JudgeErrorFitResult:
    """Posterior samples and light diagnostics."""

    samples: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    data: JudgeErrorData


def prepare_pair_judge_data(
    human: Any,
    gpt4: Any,
    *,
    model_lo: str,
    model_hi: str,
    categories: Any | None = None,
    unit_mask: np.ndarray | None = None,
) -> JudgeErrorData:
    """Build JudgeErrorData for one canonical pair from human + GPT-4 votes.

    Units are (question_id, turn) for this pair. Labels keep ties. Human and
    GPT-4 rows on the same unit both enter the likelihood.
    """
    import pandas as pd

    from lee.data.mt_bench import add_canonical_columns, unit_key_columns

    if "winner_canon" not in human.columns:
        human = add_canonical_columns(human)
    if "winner_canon" not in gpt4.columns:
        gpt4 = add_canonical_columns(gpt4)

    keys = list(unit_key_columns())
    h = human[
        (human["model_lo"] == model_lo) & (human["model_hi"] == model_hi)
    ].copy()
    g = gpt4[
        (gpt4["model_lo"] == model_lo) & (gpt4["model_hi"] == model_hi)
    ].copy()

    # Unit index from union of keys
    unit_frame = (
        pd.concat([h[keys], g[keys]], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unit_frame["unit_id"] = np.arange(len(unit_frame))
    if unit_mask is not None:
        if unit_mask.shape[0] != len(unit_frame):
            raise ValueError("unit_mask length must match n_units before subset")
        keep_ids = set(np.flatnonzero(unit_mask).tolist())
        unit_frame = unit_frame[unit_frame["unit_id"].isin(keep_ids)].copy()
        # reindex densely
        unit_frame = unit_frame.reset_index(drop=True)
        unit_frame["unit_id"] = np.arange(len(unit_frame))

    unit_key_to_id = {
        tuple(row[k] for k in keys): int(row["unit_id"])
        for _, row in unit_frame.iterrows()
    }

    if categories is not None:
        cat_map = {
            int(r.question_id): str(r.category) for _, r in categories.iterrows()
        }
        cat_names = sorted(set(cat_map.values()))
        cat_index = {c: i for i, c in enumerate(cat_names)}
    else:
        cat_map = {}
        cat_names = ["all"]
        cat_index = {"all": 0}

    # Two rater classes: human pool + GPT-4. Matches the plan's "humans as
    # one rater class"; individual human deviations are deferred (too few
    # votes per expert for stable per-judge matrices on a single pair).
    judges = ["human_class", "gpt4_pair"]
    judge_index = {j: i for i, j in enumerate(judges)}

    label_map = {
        "model_a": 0,
        "model_b": 1,
        "tie": 2,
        "tie (inconsistent)": 2,
    }

    rows: list[dict[str, Any]] = []
    for src, is_h in ((h, True), (g, False)):
        for _, row in src.iterrows():
            key = tuple(row[k] for k in keys)
            if key not in unit_key_to_id:
                continue
            lab = label_map.get(str(row["winner_canon"]))
            if lab is None:
                continue
            qid = int(row["question_id"])
            cat = cat_map.get(qid, "all")
            jname = "human_class" if is_h else "gpt4_pair"
            rows.append(
                {
                    "unit_index": unit_key_to_id[key],
                    "judge_index": judge_index[jname],
                    "category_index": cat_index[cat],
                    "label_index": lab,
                    "a_presented_first": 0 if bool(row["swapped"]) else 1,
                    "is_human": 1 if is_h else 0,
                }
            )

    if not rows:
        raise ValueError("No observations for judge-error data")

    obs = pd.DataFrame(rows)
    meta = {
        "model_lo": model_lo,
        "model_hi": model_hi,
        "judges": judges,
        "human_judges": ["human_class"],
        "categories": cat_names,
        "unit_keys": unit_frame[keys].to_dict(orient="list"),
        "label_map": label_map,
        "rater_pooling": "human_class+gpt4_pair",
    }
    return JudgeErrorData(
        unit_index=obs["unit_index"].to_numpy(dtype=np.int32),
        judge_index=obs["judge_index"].to_numpy(dtype=np.int32),
        category_index=obs["category_index"].to_numpy(dtype=np.int32),
        label_index=obs["label_index"].to_numpy(dtype=np.int32),
        a_presented_first=obs["a_presented_first"].to_numpy(dtype=np.int32),
        is_human=obs["is_human"].to_numpy(dtype=np.int32),
        n_units=int(unit_frame["unit_id"].nunique()),
        n_judges=len(judges),
        n_categories=len(cat_names),
        meta=meta,
    )


def build_model(data: JudgeErrorData):
    """Return a NumPyro model closed over ``data``."""
    import jax.nn as jnn
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    n_units = data.n_units
    n_judges = data.n_judges
    n_cat = data.n_categories
    unit_idx = jnp.asarray(data.unit_index)
    judge_idx = jnp.asarray(data.judge_index)
    cat_idx = jnp.asarray(data.category_index)
    labels = jnp.asarray(data.label_index)
    a_first = jnp.asarray(data.a_presented_first)
    is_human = jnp.asarray(data.is_human)

    def model() -> None:
        # Population mix over latent preferences
        pi = numpyro.sample("pi", dist.Dirichlet(jnp.ones(N_OUTCOMES)))

        # Shared confusion profile (3x3); per-judge rows pull toward it.
        base = numpyro.sample(
            "confusion_base",
            dist.Dirichlet(jnp.ones(N_OUTCOMES)).expand([N_OUTCOMES]).to_event(1),
        )  # (3, 3)
        conc = numpyro.sample("conc", dist.LogNormal(jnp.log(10.0), 0.5))

        with numpyro.plate("judges", n_judges):
            conf = numpyro.sample(
                "confusion",
                dist.Dirichlet(conc * base + 1e-3).to_event(1),
            )  # (J, 3, 3)

        with numpyro.plate("categories", n_cat):
            cat_logit = numpyro.sample(
                "cat_logit",
                dist.Normal(0.0, 0.3).expand([N_OUTCOMES]).to_event(1),
            )  # (C, 3)
        cat_pi = jnn.softmax(jnp.log(pi + 1e-8) + cat_logit, axis=-1)

        pos_bias = numpyro.sample("pos_bias", dist.Normal(0.0, 0.5))

        unit_cat = jnp.zeros(n_units, dtype=jnp.int32)
        unit_cat = unit_cat.at[unit_idx].set(cat_idx)
        pi_u = cat_pi[unit_cat]  # (U, 3)

        conf_j = conf[judge_idx]  # (N, 3, 3)
        conf_swap = conf_j[:, :, jnp.array([1, 0, 2])]
        use_swap = (a_first == 0) & (is_human == 0)
        conf_eff = jnp.where(use_swap[:, None, None], conf_swap, conf_j)

        tilt = jnp.exp(pos_bias)
        weights = jnp.array([tilt, 1.0 / tilt, 1.0])
        conf_llm = conf_eff * weights[None, None, :]
        conf_llm = conf_llm / conf_llm.sum(axis=-1, keepdims=True)
        conf_use = jnp.where((is_human == 0)[:, None, None], conf_llm, conf_eff)

        pi_obs = pi_u[unit_idx]  # (N, 3)
        mix = jnp.einsum("nz,nzy->ny", pi_obs, conf_use)
        mix = mix / mix.sum(axis=-1, keepdims=True)
        numpyro.sample("obs", dist.Categorical(probs=mix), obs=labels)

        numpyro.deterministic(
            "winrate_a_nontie",
            pi[0] / (pi[0] + pi[1] + 1e-12),
        )

    return model


def fit_nuts(
    data: JudgeErrorData,
    *,
    num_warmup: int = 300,
    num_samples: int = 300,
    seed: int = 20261026,
) -> JudgeErrorFitResult:
    """Fit with NUTS."""
    from jax import random
    from numpyro.infer import MCMC, NUTS

    model = build_model(data)
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed))
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    extras = {}
    try:
        extras["summary"] = str(mcmc.print_summary(exclude_deterministic=False))
    except Exception as exc:  # noqa: BLE001
        extras["summary_error"] = str(exc)
    return JudgeErrorFitResult(
        samples=samples,
        diagnostics={
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "seed": seed,
            "n_obs": int(data.label_index.size),
            "n_units": data.n_units,
            "n_judges": data.n_judges,
            **extras,
        },
        data=data,
    )


def fit_svi(
    data: JudgeErrorData,
    *,
    num_steps: int = 10_000,
    seed: int = 20261026,
) -> JudgeErrorFitResult:
    """Fit with SVI (fallback if NUTS is too slow)."""
    import numpyro
    import numpyro.distributions as dist
    from jax import random
    from numpyro.infer import SVI, Trace_ELBO, autoguide
    from numpyro.optim import Adam

    model = build_model(data)
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, Adam(step_size=1e-2), loss=Trace_ELBO())
    rng = random.PRNGKey(seed)
    state = svi.init(rng)
    losses = []
    for step in range(num_steps):
        state, loss = svi.update(state)
        if step % 500 == 0:
            losses.append(float(loss))
    params = svi.get_params(state)
    # predictive samples from guide
    predictive = guide.sample_posterior(rng, params, sample_shape=(500,))
    samples = {k: np.asarray(v) for k, v in predictive.items()}
    return JudgeErrorFitResult(
        samples=samples,
        diagnostics={
            "method": "svi",
            "num_steps": num_steps,
            "losses_every_500": losses,
            "n_obs": int(data.label_index.size),
        },
        data=data,
    )


def posterior_winrate_interval(
    result: JudgeErrorFitResult,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and central posterior interval for nontie P(A wins)."""
    if "winrate_a_nontie" in result.samples:
        s = np.asarray(result.samples["winrate_a_nontie"], dtype=float).ravel()
    else:
        pi = np.asarray(result.samples["pi"], dtype=float)
        s = pi[..., 0] / (pi[..., 0] + pi[..., 1] + 1e-12)
        s = s.ravel()
    lo = float(np.quantile(s, alpha / 2.0))
    hi = float(np.quantile(s, 1.0 - alpha / 2.0))
    return float(np.mean(s)), lo, hi


def _decisive_judge_data(
    human: Any,
    gpt4: Any,
    categories: Any,
    model_lo: str,
    model_hi: str,
) -> tuple[JudgeErrorData, Any, float]:
    """Judge-error arrays restricted to PPI++ decisive majority units."""
    from lee.data.mt_bench import build_pair_population

    pop = build_pair_population(
        human, gpt4, model_lo=model_lo, model_hi=model_hi
    )
    base = prepare_pair_judge_data(
        human,
        gpt4,
        model_lo=model_lo,
        model_hi=model_hi,
        categories=categories,
    )
    keys = list(
        zip(
            base.meta["unit_keys"]["question_id"],
            base.meta["unit_keys"]["turn"],
            base.meta["unit_keys"]["model_lo"],
            base.meta["unit_keys"]["model_hi"],
        )
    )
    pop_key_to_i = {
        (
            int(pop.loc[i, "question_id"]),
            int(pop.loc[i, "turn"]),
            pop.loc[i, "model_lo"],
            pop.loc[i, "model_hi"],
        ): i
        for i in range(len(pop))
    }
    keep = [
        o
        for o in range(base.label_index.size)
        if keys[int(base.unit_index[o])] in pop_key_to_i
    ]
    keep_arr = np.asarray(keep, dtype=int)
    old_units = base.unit_index[keep_arr]
    uniq, inv = np.unique(old_units, return_inverse=True)
    # Map new unit id -> population row
    unit_to_pop = np.zeros(uniq.size, dtype=int)
    for new_u, old_u in enumerate(uniq):
        unit_to_pop[new_u] = pop_key_to_i[keys[int(old_u)]]
    data = JudgeErrorData(
        unit_index=inv.astype(np.int32),
        judge_index=base.judge_index[keep_arr],
        category_index=base.category_index[keep_arr],
        label_index=base.label_index[keep_arr],
        a_presented_first=base.a_presented_first[keep_arr],
        is_human=base.is_human[keep_arr],
        n_units=int(uniq.size),
        n_judges=base.n_judges,
        n_categories=base.n_categories,
        meta={
            **base.meta,
            "decisive_only": True,
            "unit_to_pop": unit_to_pop.tolist(),
        },
    )
    return data, pop, float(pop["y"].mean())


def subsample_eval_vs_truth(
    *,
    human: Any,
    gpt4: Any,
    categories: Any,
    model_lo: str,
    model_hi: str,
    truth: float | None = None,
    n_grid: tuple[int, ...] = (30, 50),
    R: int = 20,
    alpha: float = 0.05,
    seed_root: int = 20261026,
    num_warmup: int = 150,
    num_samples: int = 150,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Fit on random human-labeled unit subsets; score coverage of ``truth``.

    Uses the same seed formula as M2 PPI++: ``seed_root + 2_000_003 * n + r``.
    Units are the decisive majority population (same as PPI++). Each fit uses
    human labels on the labeled subset plus all GPT-4 labels on the pair.
    Default R is small because each draw is a NUTS run.
    """
    from lee.sim.m2 import subset_seed

    log = progress or (lambda _m: None)
    base, pop, pop_truth = _decisive_judge_data(
        human, gpt4, categories, model_lo, model_hi
    )
    if truth is None:
        truth = pop_truth
    n_pop = len(pop)
    unit_to_pop = np.asarray(base.meta["unit_to_pop"], dtype=int)

    rows: list[dict[str, Any]] = []
    for n in n_grid:
        if n >= n_pop:
            continue
        cover: list[bool] = []
        widths: list[float] = []
        estimates: list[float] = []
        for r in range(R):
            rng = np.random.default_rng(subset_seed(n, r))
            labeled = set(rng.choice(n_pop, size=n, replace=False).tolist())
            keep = []
            for o in range(base.label_index.size):
                pop_i = int(unit_to_pop[int(base.unit_index[o])])
                if base.is_human[o] == 0:
                    keep.append(o)
                elif pop_i in labeled:
                    keep.append(o)
            keep_arr = np.asarray(keep, dtype=int)
            if keep_arr.size < 10:
                continue
            old_units = base.unit_index[keep_arr]
            uniq, inv = np.unique(old_units, return_inverse=True)
            # Drop unused judges to keep plates tight
            old_judges = base.judge_index[keep_arr]
            j_uniq, j_inv = np.unique(old_judges, return_inverse=True)
            sub = JudgeErrorData(
                unit_index=inv.astype(np.int32),
                judge_index=j_inv.astype(np.int32),
                category_index=base.category_index[keep_arr],
                label_index=base.label_index[keep_arr],
                a_presented_first=base.a_presented_first[keep_arr],
                is_human=base.is_human[keep_arr],
                n_units=int(uniq.size),
                n_judges=int(j_uniq.size),
                n_categories=base.n_categories,
                meta=base.meta,
            )
            try:
                fit = fit_nuts(
                    sub,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    seed=subset_seed(n, r),
                )
                est, lo, hi = posterior_winrate_interval(fit, alpha=alpha)
            except Exception as exc:  # noqa: BLE001
                log(f"fit failed n={n} r={r}: {exc}")
                continue
            cover.append(bool(lo <= truth <= hi))
            widths.append(float(hi - lo))
            estimates.append(float(est))
            log(f"judge_error n={n} r={r} est={est:.3f} width={hi - lo:.3f}")
        if not cover:
            continue
        cov = float(np.mean(cover))
        rows.append(
            {
                "estimator": "judge_error_nuts",
                "judge": "gpt4_pair+humans",
                "estimand": f"winrate:{model_lo}_vs_{model_hi}",
                "n": n,
                "R": len(cover),
                "alpha": alpha,
                "coverage": cov,
                "coverage_se": float(np.sqrt(cov * (1 - cov) / len(cover))),
                "mean_width": float(np.mean(widths)),
                "bias": float(np.mean(estimates) - truth),
                "rmse": float(
                    np.sqrt(np.mean((np.asarray(estimates) - truth) ** 2))
                ),
                "mean_lam": None,
            }
        )
    return rows
