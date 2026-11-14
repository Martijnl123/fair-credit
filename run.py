"""Run the full pipeline for one dataset.

Example:
    python run.py --dataset german --model lr --constraint fnr --seeds 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
logging.getLogger().setLevel(logging.ERROR) 

from fair_credit.data import LOADERS, load  
from fair_credit.metrics import label_metrics  
from fair_credit.pipeline import run_once 
from fair_credit.plots import reliability  

SHOW = ["threshold", "dp_diff", "dir", "consistency", "accuracy", "balanced_accuracy", "auc",
        "tpr_gap", "fpr_gap", "gfnr_gap", "gfpr_gap", "approval_priv", "approval_unpriv",
        "default_among_approved_priv", "default_among_approved_unpriv"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    ap.add_argument("--model", default="lr", choices=["lr", "gbm"])
    ap.add_argument("--constraint", default="fnr", choices=["fnr", "fpr", "weighted"],
                    help="which generalized error rate to equalise between groups")
    ap.add_argument("--seeds", type=int, default=5, help="number of random splits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ds = load(args.dataset)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"{args.dataset}_{args.model}_{args.constraint}"
    print(f"{ds.name}: {len(ds.y)} rows, {ds.X.shape[1]} features, protected = {ds.protected} "
          f"(privileged: {ds.privileged_desc})")

    labels = label_metrics(ds.X.to_numpy(), ds.y, ds.s)
    print("\nOriginal labels (before any model):")
    print(json.dumps({k: round(v, 4) for k, v in labels.items()}, indent=2))

    rows = []
    for seed in range(args.seeds):
        base, mitig, curves = run_once(ds, args.model, args.constraint, seed)
        rows += [{"seed": seed, "stage": "baseline", **base}, {"seed": seed, "stage": "ceo", **mitig}]
        if seed == 0:
            reliability(curves, f"{ds.name}, {args.model}, constraint = {args.constraint}",
                        out / f"{tag}_reliability.png")

    res = pd.DataFrame(rows)
    res.to_csv(out / f"{tag}_runs.csv", index=False)
    cols = [c for c in SHOW if c in res.columns]
    summary = res.groupby("stage")[cols].agg(["mean", "std", "count"]).T.unstack()
    summary.to_csv(out / f"{tag}_summary.csv")
    pd.Series(labels).to_csv(out / f"{tag}_labels.csv", header=["value"])

    print(f"\nTest set, mean over {args.seeds} seeds (std in brackets):")
    for c in cols:
        b, m = res[res.stage == "baseline"][c], res[res.stage == "ceo"][c]
        print(f"  {c:32s} baseline {b.mean():7.4f} ({b.std():.4f})   ceo {m.mean():7.4f} ({m.std():.4f}) n = {m.count()}")
    print(f"\nSaved to {out}/{tag}_*")


if __name__ == "__main__":
    main()
