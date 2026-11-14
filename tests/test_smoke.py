import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fair_credit.data import load_synthetic  # noqa: E402
from fair_credit.metrics import dir_ratio, dp_diff, label_metrics  # noqa: E402
from fair_credit.pipeline import run_once  # noqa: E402


def test_metric_definitions():
    yhat = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    s = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert dp_diff(yhat, s) == 0.25 - 0.5
    assert dir_ratio(yhat, s) == 0.5


def test_pipeline_runs():
    ds = load_synthetic(n=3000)
    labels = label_metrics(ds.X.to_numpy(), ds.y, ds.s)
    assert 0 <= labels["consistency"] <= 1
    for model in ["lr", "gbm"]:
        base, mitig, curves = run_once(ds, model, "fnr", seed=0)
        assert 0 <= mitig["accuracy"] <= 1
        assert len(curves["p_ceo"]) == len(curves["y"])


def test_missing_values_are_handled_inside_the_model():
    ds = load_synthetic(n=2000)
    ds.X.iloc[::5, 0] = np.nan
    for model in ["lr", "gbm"]:
        base, mitig, _ = run_once(ds, model, "fnr", seed=1)
        assert np.isfinite(base["auc"]) and 0 <= mitig["consistency"] <= 1
        assert "mix_rate_priv" in mitig and "val_headroom" in mitig
