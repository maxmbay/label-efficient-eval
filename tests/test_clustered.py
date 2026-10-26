"""Tests for cluster-robust human-only win-rate intervals."""

from __future__ import annotations

import numpy as np
import pytest

from lee.estimators.classical import classical_clt, classical_clustered, cluster_robust_se_mean
from lee.estimators.pairwise import human_only_winrate_clustered


def test_cluster_se_wider_than_iid_when_within_cluster_dep() -> None:
    # Two observations per question, identical within cluster -> clustering
    # should inflate SE relative to iid (which pretends n=20 independents).
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0] * 2)
    clusters = np.repeat(np.arange(10), 2)
    se_c = cluster_robust_se_mean(y, clusters)
    se_iid = float(y.std(ddof=1) / np.sqrt(y.size))
    assert se_c > se_iid


def test_cluster_se_matches_iid_when_one_obs_per_cluster() -> None:
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.4, size=40).astype(float)
    clusters = np.arange(40)
    se_c = cluster_robust_se_mean(y, clusters)
    se_iid = float(y.std(ddof=1) / np.sqrt(y.size))
    # CR1 with G=n is sqrt(n/(n-1)) * (sum (y_i-ybar)^2)/n^2 under the sqrt
    # = std(ddof=1)/sqrt(n). Exact match.
    assert se_c == pytest.approx(se_iid, rel=1e-12)


def test_human_only_clustered_interval() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    q = np.array([10, 10, 11, 11, 12, 13])
    est = human_only_winrate_clustered(y, question_ids=q, alpha=0.05)
    assert est.method == "classical_clustered"
    assert est.lower < est.estimate < est.upper
    # Width vs iid is not monotone in general; positive within-cluster
    # correlation widens (see test_cluster_se_wider_than_iid_when_within_cluster_dep).
    _ = classical_clt(y, alpha=0.05)


def test_single_cluster_fallback() -> None:
    y = np.array([1.0, 0.0, 1.0])
    q = np.array([1, 1, 1])
    est = classical_clustered(y, q, alpha=0.05)
    assert est.method == "classical_clustered_fallback_iid"
