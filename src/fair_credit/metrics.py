"""Fairness, performance and lender-view metrics.

Conventions: y and predictions use 1 = favourable (approve / repays),
s uses 1 = privileged. These match AIF360's definitions:
  DP_Diff = P(yhat=1 | s=0) - P(yhat=1 | s=1)        fair at 0
  DIR     = P(yhat=1 | s=0) / P(yhat=1 | s=1)        fair at 1 (0.8 to 1.25 rule)
  Consistency = 1 - mean_i | yhat_i - mean(yhat over i's k nearest neighbours) |
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def dp_diff(yhat, s) -> float:
    return float(yhat[s == 0].mean() - yhat[s == 1].mean())


def dir_ratio(yhat, s) -> float:
    priv = yhat[s == 1].mean()
    return float(yhat[s == 0].mean() / priv) if priv > 0 else float("nan")


def consistency(X, yhat, k: int = 5, max_rows: int = 20_000, seed: int = 0) -> float:
    """Individual fairness (Zemel et al. 2013). Features are standardised first so no
    single large-scale column dominates the distance. On large data a fixed random
    subsample is used; pass the same seed before and after mitigation."""
    X = np.asarray(X, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if len(yhat) > max_rows:
        idx = np.random.default_rng(seed).choice(len(yhat), max_rows, replace=False)
        X, yhat = X[idx], yhat[idx]
    Xs = StandardScaler().fit_transform(X)
    # k + 1 neighbours, then drop the point itself (AIF360 keeps it; difference is small)
    _, nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs)
    return float(1 - np.abs(yhat - yhat[nn[:, 1:]].mean(axis=1)).mean())


def group_rates(y, yhat, s) -> dict:
    out = {}
    for g, tag in [(1, "priv"), (0, "unpriv")]:
        m = s == g
        yt, yp = y[m], yhat[m]
        out[f"tpr_{tag}"] = float(yp[yt == 1].mean()) if (yt == 1).any() else np.nan
        out[f"fpr_{tag}"] = float(yp[yt == 0].mean()) if (yt == 0).any() else np.nan
        # lender view: who gets approved, and how many of the approved fail to repay
        out[f"approval_{tag}"] = float(yp.mean())
        approved = yp == 1
        out[f"default_among_approved_{tag}"] = float((yt[approved] == 0).mean()) if approved.any() else np.nan
    out["tpr_gap"] = out["tpr_unpriv"] - out["tpr_priv"]
    out["fpr_gap"] = out["fpr_unpriv"] - out["fpr_priv"]
    return out

def generalized_rates(y, scores, s) -> dict:
    """Generalised FNR and FPR per group, computed from scores instead of decisions."""
    out = {}
    for g, tag in [(1, "priv"), (0, "unpriv")]:
        m = s == g
        yt, sc = y[m], scores[m]
        out[f"gfnr_{tag}"] = float(np.mean(1 - sc[yt == 1]))   # doubt placed on repayers
        out[f"gfpr_{tag}"] = float(np.mean(sc[yt == 0]))       # trust given to defaulters
    out["gfnr_gap"] = out["gfnr_unpriv"] - out["gfnr_priv"]
    out["gfpr_gap"] = out["gfpr_unpriv"] - out["gfpr_priv"]
    return out


def all_metrics(X, y, yhat, s, scores=None, seed: int = 0) -> dict:
    out = {
        "dp_diff": dp_diff(yhat, s),
        "dir": dir_ratio(yhat, s),
        "consistency": consistency(X, yhat, seed=seed),
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
    }
    if scores is not None and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, scores))
        out.update(generalized_rates(y, scores, s))
    return out


def label_metrics(X, y, s, seed: int = 0) -> dict:
    """Task 2 step 2: fairness of the original labels, before any model."""
    return {"dp_diff": dp_diff(y, s), "dir": dir_ratio(y, s), "consistency": consistency(X, y, seed=seed)}
