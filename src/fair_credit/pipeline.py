"""Split, train, post-process with Calibrated Equalized Odds, evaluate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from .data import Dataset
from .metrics import all_metrics, consistency, generalized_rates

PRIV, UNPRIV = [{"s": 1}], [{"s": 0}]


def make_model(name: str, seed: int):
    if name == "lr":
        # Imputation is part of the model, so its medians are learned on the training split only.
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    if name == "gbm":
        # Boosted trees are often miscalibrated, and the method assumes calibrated scores,
        # so wrap them in isotonic calibration (cross-validated on the training set).
        # Gradient boosting handles missing values natively, so no imputer is needed.
        gbm = HistGradientBoostingClassifier(random_state=seed)
        return CalibratedClassifierCV(gbm, method="isotonic", cv=5)
    raise ValueError(f"Unknown model {name!r}. Choose 'lr' or 'gbm'.")


def split(ds: Dataset, seed: int, test_size=0.2, val_size=0.2):
    """Stratified train / validation / test split on label and group jointly.
    The validation set is only used to fit the post-processor."""
    strata = ds.y * 2 + ds.s
    idx = np.arange(len(ds.y))
    tr_val, te = train_test_split(idx, test_size=test_size, stratify=strata, random_state=seed)
    tr, va = train_test_split(tr_val, test_size=val_size / (1 - test_size),
                              stratify=strata[tr_val], random_state=seed)
    return tr, va, te


def _bld(y, s, scores=None) -> BinaryLabelDataset:
    """Minimal AIF360 dataset: the post-processor only needs labels, scores and group."""
    df = pd.DataFrame({"s": s.astype(float), "y": y.astype(float)})
    d = BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["s"],
                           favorable_label=1.0, unfavorable_label=0.0)
    if scores is not None:
        d.scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    return d

def choose_threshold(y, p):
    """Cutoff that maximises TPR - FPR (Youden's J) on the given labels and scores."""
    fpr, tpr, thresholds = roc_curve(y, p)
    best = np.argmax(tpr - fpr)
    return float(thresholds[best])

def feasibility(y, scores, s, constraint: str) -> dict:
    """Section 4 of Pleiss et al.: the group with the lower cost is degraded towards its
    constant base-rate classifier, the worst calibrated classifier it can have. The constraint
    is reachable only if the other group's cost is at most that trivial cost. Computed on the
    same (validation) data the post-processor is fitted on. Only for "fnr" and "fpr"."""
    if constraint not in ("fnr", "fpr"):
        return {}
    g = generalized_rates(y, scores, s)
    cost = {t: g[f"g{constraint}_{t}"] for t in ("priv", "unpriv")}
    mu = {"priv": float(y[s == 1].mean()), "unpriv": float(y[s == 0].mean())}
    trivial = {t: (1 - mu[t]) if constraint == "fnr" else mu[t] for t in mu}
    low = min(cost, key=cost.get); high = "unpriv" if low == "priv" else "priv"
    return {"degraded_is_priv": float(low == "priv"),
            "cost_degraded": cost[low], "trivial_cost_degraded": trivial[low],
            "cost_other": cost[high],
            "headroom": trivial[low] - cost[low],     # <= 0 breaks the calibration assumption
            "feasible": float(cost[high] <= trivial[low])}


def run_once(ds: Dataset, model: str, constraint: str, seed: int):
    tr, va, te = split(ds, seed)
    X = ds.X.to_numpy()
    clf = make_model(model, seed).fit(X[tr], ds.y[tr])
    p_va, p_te = clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]
    threshold = choose_threshold(ds.y[va], p_va)  

    yhat_base = (p_te >= threshold).astype(int)
    base = all_metrics(X[te], ds.y[te], yhat_base, ds.s[te], scores=p_te, seed=seed)

    ceo = CalibratedEqOddsPostprocessing(UNPRIV, PRIV, cost_constraint=constraint, seed=seed)
    ceo.fit(_bld(ds.y[va], ds.s[va]), _bld((p_va >= threshold).astype(int), ds.s[va], p_va))
    out = ceo.predict(_bld((p_te >= threshold).astype(int), ds.s[te], p_te), threshold=threshold)
    yhat_ceo, p_ceo = out.labels.ravel().astype(int), out.scores.ravel()
    mitig = all_metrics(X[te], ds.y[te], yhat_ceo, ds.s[te], scores=p_ceo, seed=seed)
    base["threshold"] = mitig["threshold"] = threshold

    # Diagnostics for the report
    lab_cons = consistency(X[te], ds.y[te], seed=seed)          # labels, same test rows
    base["consistency_labels_test"] = mitig["consistency_labels_test"] = lab_cons
    # Share of each group whose score is replaced by its base rate. AIF360 does not clip
    # this: above 1 means everyone is replaced, below 0 means nobody is.
    mitig["mix_rate_priv"] = float(ceo.priv_mix_rate)
    mitig["mix_rate_unpriv"] = float(ceo.unpriv_mix_rate)
    mitig.update({f"val_{k}": v for k, v in
                  feasibility(ds.y[va], p_va, ds.s[va], constraint).items()})

    curves = {"y": ds.y[te], "s": ds.s[te], "p_base": p_te, "p_ceo": p_ceo}
    return base, mitig, curves
