"""Dataset loaders.

Every loader returns a `Dataset` with:
  X : numeric feature DataFrame (protected attribute NOT included)
  y : 1 = favourable outcome (repays / good credit), 0 = unfavourable
  s : 1 = privileged group, 0 = unprivileged group

Encoding "1 = favourable" means a positive prediction is an approval, so
demographic parity compares approval rates between groups.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    s: np.ndarray
    protected: str
    privileged_desc: str
    unprivileged_desc: str


def _one_hot(df: pd.DataFrame, max_levels: int = 30) -> pd.DataFrame:
    text = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cats = [c for c in text if df[c].nunique() <= max_levels]
    drop = [c for c in text if c not in cats]
    return pd.get_dummies(df.drop(columns=drop), columns=cats, dtype=float).astype(float)


def load_german(path: Path | None = None) -> Dataset:
    """UCI Statlog German Credit (german.data). Protected: age, older than 25 = privileged."""
    path = path or RAW / "german.data"
    cols = [
        "status", "duration", "credit_history", "purpose", "credit_amount", "savings",
        "employment", "installment_rate", "personal_status", "other_debtors",
        "residence_since", "property", "age", "installment_plans", "housing",
        "number_of_credits", "skill_level", "people_liable", "telephone",
        "foreign_worker", "credit",
    ]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    y = (df.pop("credit") == 1).astype(int).to_numpy()  # 1 = good, 2 = bad
    s = (df["age"] > 25).astype(int).to_numpy()
    X = _one_hot(df.drop(columns=["age"]))
    return Dataset("german", X, y, s, "age", "age > 25", "age <= 25")


def load_taiwan(path: Path | None = None) -> Dataset:
    """UCI Default of Credit Card Clients. Accepts the UCI .xls or the Kaggle UCI_Credit_Card.csv.
    Protected: sex, male = privileged."""
    if path is None:
        xls, csv = RAW / "default of credit card clients.xls", RAW / "UCI_Credit_Card.csv"
        path = xls if xls.exists() else csv
    path = Path(path)
    df = pd.read_excel(path, header=1) if path.suffix == ".xls" else pd.read_csv(path)
    target = [c for c in df.columns if c.lower().replace(".", " ").startswith("default payment")][0]
    y = (df.pop(target) == 0).astype(int).to_numpy()  # 1 = no default
    s = (df["SEX"] == 1).astype(int).to_numpy()  # 1 = male, 2 = female
    X = df.drop(columns=["ID", "SEX"])
    X = pd.get_dummies(X, columns=["EDUCATION", "MARRIAGE"], dtype=float)
    return Dataset("taiwan", X.astype(float), y, s, "sex", "male", "female")


def load_homecredit(path: Path | None = None, max_missing: float = 0.4) -> Dataset:
    """Kaggle Home Credit Default Risk, application_train.csv only.
    Protected: CODE_GENDER, male = privileged. Rows with gender XNA are dropped."""
    path = path or RAW / "application_train.csv"
    df = pd.read_csv(path)
    df = df[df["CODE_GENDER"].isin(["M", "F"])].reset_index(drop=True)
    y = (df.pop("TARGET") == 0).astype(int).to_numpy()  # 1 = repaid
    s = (df.pop("CODE_GENDER") == "M").astype(int).to_numpy()
    df = df.drop(columns=["SK_ID_CURR"])
    df = df.loc[:, df.isna().mean() <= max_missing]
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)  # known sentinel value
    X = _one_hot(df)
    # Missing values stay NaN here; they are imputed inside the model pipeline,
    # with medians learned on the training split only (no leakage into test).
    return Dataset("homecredit", X.astype(float), y, s, "sex", "male", "female")


def load_synthetic(n: int = 4000, seed: int = 0) -> Dataset:
    """Small biased toy dataset, used by the tests and `--dataset synthetic`."""
    rng = np.random.default_rng(seed)
    s = rng.binomial(1, 0.6, n)
    income = rng.normal(0.5 * s, 1.0, n)
    debt = rng.normal(0, 1, n)
    proxy = s + rng.normal(0, 0.7, n)
    logit = 0.4 + 1.2 * income - 0.8 * debt
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    X = pd.DataFrame({"income": income, "debt": debt, "proxy": proxy, "noise": rng.normal(size=n)})
    return Dataset("synthetic", X, y, s, "group", "s = 1", "s = 0")


LOADERS = {
    "german": load_german,
    "taiwan": load_taiwan,
    "homecredit": load_homecredit,
    "synthetic": load_synthetic,
}


def load(name: str) -> Dataset:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(LOADERS)}.")
    return LOADERS[name]()
