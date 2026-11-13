"""Split, train, post-process with Calibrated Equalized Odds, evaluate."""
from __future__ import annotations

import numpy as np
import pandas as pd
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Dataset
from .metrics import all_metrics

PRIV, UNPRIV = [{"s": 1}], [{"s": 0}]


def make_model(name: str, seed: int):
    if name == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "gbm":
        # Boosted trees are often miscalibrated, and the method assumes calibrated scores,
        # so wrap them in isotonic calibration (cross-validated on the training set).
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


def run_once(ds: Dataset, model: str, constraint: str, seed: int, threshold: float = 0.5):
    tr, va, te = split(ds, seed)
    X = ds.X.to_numpy()
    clf = make_model(model, seed).fit(X[tr], ds.y[tr])
    p_va, p_te = clf.predict_proba(X[va])[:, 1], clf.predict_proba(X[te])[:, 1]

    yhat_base = (p_te >= threshold).astype(int)
    base = all_metrics(X[te], ds.y[te], yhat_base, ds.s[te], scores=p_te, seed=seed)

    ceo = CalibratedEqOddsPostprocessing(UNPRIV, PRIV, cost_constraint=constraint, seed=seed)
    ceo.fit(_bld(ds.y[va], ds.s[va]), _bld((p_va >= threshold).astype(int), ds.s[va], p_va))
    out = ceo.predict(_bld((p_te >= threshold).astype(int), ds.s[te], p_te), threshold=threshold)
    yhat_ceo, p_ceo = out.labels.ravel().astype(int), out.scores.ravel()
    mitig = all_metrics(X[te], ds.y[te], yhat_ceo, ds.s[te], scores=p_ceo, seed=seed)

    curves = {"y": ds.y[te], "s": ds.s[te], "p_base": p_te, "p_ceo": p_ceo}
    return base, mitig, curves
