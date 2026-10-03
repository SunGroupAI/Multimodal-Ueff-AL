#!/usr/bin/env python3
"""Rank unlabeled ABO3 candidates with uncertainty, diversity, and cost."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


TARGET_COL = "U_eV"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path).reset_index(drop=True)
    return pd.read_csv(path).reset_index(drop=True)


def sorted_prefixed(columns, prefix: str) -> list[str]:
    pat = re.compile(rf"{re.escape(prefix)}(\d+)$")
    return sorted([c for c in columns if pat.match(c)], key=lambda x: int(pat.match(x).group(1)))  # type: ignore[union-attr]


def norm01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    lo = np.nanmin(values[finite])
    hi = np.nanmax(values[finite])
    return (values - lo) / (hi - lo + 1e-12)


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    text_cols = sorted_prefixed(df.columns, "txt_")
    graph_cols = sorted_prefixed(df.columns, "graph_")
    non_features = {
        "material_id",
        "id",
        "formula",
        "formula_norm",
        "struct_path",
        "text_path",
        "text",
        TARGET_COL,
        "A_elem",
        "B_elem",
        "B_block",
    }
    tab_cols = [
        c
        for c in df.columns
        if c not in non_features
        and c not in text_cols
        and c not in graph_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]
    return tab_cols + graph_cols + text_cols


def diversity_to_labeled(X: np.ndarray, labeled: np.ndarray) -> np.ndarray:
    """Distance from each row to nearest labeled row in a standardized PCA space."""
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_imp)
    n_comp = int(min(20, X_scaled.shape[0], X_scaled.shape[1]))
    if n_comp >= 2:
        X_repr = PCA(n_components=n_comp, random_state=0).fit_transform(X_scaled)
    else:
        X_repr = X_scaled
    if labeled.sum() == 0:
        return np.ones(len(X_repr), dtype=float)
    dist = pairwise_distances(X_repr, X_repr[labeled], metric="euclidean")
    return dist.min(axis=1)


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=root / "processed/multimodal_dataset.csv")
    ap.add_argument("--predictions", type=Path, default=root / "results/predictions_with_uncertainty.csv")
    ap.add_argument("--out", type=Path, default=root / "results/active_learning_ranked_candidates.csv")
    ap.add_argument("--metadata", type=Path, default=root / "results/active_learning_selection_metadata.json")
    ap.add_argument("--w-uncertainty", type=float, default=1.0)
    ap.add_argument("--w-diversity", type=float, default=0.5)
    ap.add_argument("--w-cost", type=float, default=0.25)
    ap.add_argument("--cost-column", default="cost_estimate")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    df = read_table(args.dataset)
    pred = read_table(args.predictions)

    join_key = "material_id" if "material_id" in df.columns and "material_id" in pred.columns else None
    if join_key is None:
        join_key = "id" if "id" in df.columns and "id" in pred.columns else None
    if join_key is None:
        raise ValueError("Dataset and prediction table need a shared `material_id` or `id` column.")

    pred_cols = [
        join_key,
        "prediction_mean_eV",
        "epistemic_uncertainty_eV",
        "total_uncertainty_eV",
        "is_labeled",
    ]
    merged = df.merge(pred[[c for c in pred_cols if c in pred.columns]], on=join_key, how="left", suffixes=("", "_pred"))

    if "is_labeled" in merged.columns:
        labeled = merged["is_labeled"].astype(bool).to_numpy()
    elif TARGET_COL in merged.columns:
        labeled = merged[TARGET_COL].notna().to_numpy()
    else:
        labeled = np.zeros(len(merged), dtype=bool)

    feat_cols = infer_feature_columns(merged)
    if not feat_cols:
        raise ValueError("No numeric feature columns found for diversity scoring.")
    diversity_raw = diversity_to_labeled(merged[feat_cols].to_numpy(dtype=float), labeled)

    uncertainty_col = "total_uncertainty_eV" if "total_uncertainty_eV" in merged.columns else "epistemic_uncertainty_eV"
    if uncertainty_col not in merged.columns:
        raise ValueError("Prediction table lacks uncertainty columns.")
    uncertainty_raw = merged[uncertainty_col].to_numpy(dtype=float)

    if args.cost_column in merged.columns:
        cost_raw = merged[args.cost_column].to_numpy(dtype=float)
        cost_status = "ok"
    else:
        cost_raw = np.ones(len(merged), dtype=float)
        cost_status = f"missing `{args.cost_column}`; used uniform cost"

    merged["uncertainty_score"] = norm01(uncertainty_raw)
    merged["diversity_score"] = norm01(diversity_raw)
    merged["labeling_cost_score"] = norm01(cost_raw)
    merged["active_learning_score"] = (
        args.w_uncertainty * merged["uncertainty_score"]
        + args.w_diversity * merged["diversity_score"]
        - args.w_cost * merged["labeling_cost_score"]
    )

    candidates = merged.loc[~labeled].copy()
    candidates = candidates.sort_values("active_learning_score", ascending=False)
    if args.top_k > 0:
        candidates = candidates.head(args.top_k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.out, index=False)
    args.metadata.write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "predictions": str(args.predictions),
                "output": str(args.out),
                "weights": {
                    "uncertainty": args.w_uncertainty,
                    "diversity": args.w_diversity,
                    "cost": args.w_cost,
                },
                "cost_status": cost_status,
                "feature_columns_for_diversity": feat_cols,
                "formula": "score = w_uncertainty*uncertainty_score + w_diversity*diversity_score - w_cost*labeling_cost_score",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(candidates[[c for c in [join_key, "formula", "prediction_mean_eV", uncertainty_col, "active_learning_score"] if c in candidates.columns]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
