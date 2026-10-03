#!/usr/bin/env python
"""Strict multimodal ablation and active-learning benchmark for Ueff.

The script compares composition, structure, and electronic/physics modalities
under identical train/pool/test splits. It fits preprocessing only on the
currently labeled training subset at each active-learning round.

Example:
  python scripts/run_active_learning_ablation.py \
    --data data/processed/Ueff_clean_main.csv \
    --modalities comp struct electronic \
    --models rf extratrees xgb gpr ensemble \
    --strategies random uncertainty diversity hybrid \
    --initial_size 10 --batch_size 5 --rounds 20 \
    --seeds 0 1 2 3 4 --test_size 0.2 \
    --output results/active_learning_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, pairwise_distances, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


MODALITY_PREFIX = {"comp": "comp__", "struct": "struct__", "electronic": "elec__"}
MODALITY_COMBOS = [
    ("comp",),
    ("struct",),
    ("electronic",),
    ("comp", "struct"),
    ("comp", "electronic"),
    ("struct", "electronic"),
    ("comp", "struct", "electronic"),
]


@dataclass
class FitResult:
    pred: np.ndarray
    uncertainty: np.ndarray
    estimator: object
    feature_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/Ueff_clean_main.csv"))
    parser.add_argument("--modalities", nargs="+", default=["comp", "struct", "electronic"])
    parser.add_argument("--models", nargs="+", default=["rf", "extratrees", "ensemble"])
    parser.add_argument("--strategies", nargs="+", default=["random", "uncertainty", "diversity", "hybrid"])
    parser.add_argument("--initial_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("results/active_learning_ablation"))
    parser.add_argument("--fusion", nargs="+", choices=["early", "late"], default=["early"])
    parser.add_argument("--sensitivity", choices=["remove_abnormal", "include_all", "review_only"], default="remove_abnormal")
    parser.add_argument(
        "--expected_n",
        type=int,
        default=None,
        help="Fail before training unless the filtered dataset has exactly this many numeric labels.",
    )
    return parser.parse_args()


def stable_seed_offset(*parts: object, modulus: int = 100000) -> int:
    """Return a process-independent integer offset for policy RNG seeding."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % modulus


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, uncertainty: np.ndarray | None = None) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    spearman = np.nan
    if len(y_true) > 2 and np.std(y_pred) > 0 and np.std(y_true) > 0:
        spearman = float(spearmanr(y_true, y_pred).correlation)
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
        "R2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Spearman": spearman,
    }
    if uncertainty is not None and len(uncertainty) == len(y_true):
        abs_err = np.abs(y_true - y_pred)
        if np.std(abs_err) > 0 and np.std(uncertainty) > 0:
            out["uncertainty_error_spearman"] = float(spearmanr(abs_err, uncertainty).correlation)
        else:
            out["uncertainty_error_spearman"] = np.nan
        out["mean_uncertainty"] = float(np.nanmean(uncertainty))
    return out


def modality_columns(df: pd.DataFrame, modalities: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for modality in modalities:
        prefix = MODALITY_PREFIX[modality]
        cols.extend([c for c in df.columns if c.startswith(prefix)])
    # Drop all-empty or constant non-informative columns after numeric coercion.
    clean_cols = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        clean_cols.append(c)
    return clean_cols


def x_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def make_estimator(model_name: str, seed: int, n_train: int) -> object:
    if model_name == "rf":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=45, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1))])
    if model_name == "extratrees":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1))])
    if model_name == "gpr":
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, random_state=seed, n_restarts_optimizer=0))])
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError(f"xgboost unavailable: {exc}") from exc
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", XGBRegressor(n_estimators=90, max_depth=2, learning_rate=0.055, subsample=0.85, colsample_bytree=0.8, reg_lambda=3.0, random_state=seed, n_jobs=1, verbosity=0))])
    if model_name == "ensemble":
        return [
            make_estimator("rf", seed + 11, n_train),
            make_estimator("extratrees", seed + 23, n_train),
        ]
    raise ValueError(f"Unknown model {model_name}")


def tree_uncertainty(model: object, x: np.ndarray) -> np.ndarray:
    est = model.named_steps.get("model") if isinstance(model, Pipeline) else model
    if not hasattr(est, "estimators_"):
        return np.zeros(x.shape[0], dtype=float)
    xt = model[:-1].transform(x) if isinstance(model, Pipeline) else x
    preds = np.vstack([t.predict(xt) for t in est.estimators_])
    return np.std(preds, axis=0)


def fit_predict_early(model_name: str, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, seed: int) -> FitResult:
    if model_name == "ensemble":
        models = make_estimator("ensemble", seed, len(y_train))
        preds, uncs = [], []
        for model in models:
            model.fit(x_train, y_train)
            preds.append(model.predict(x_eval))
            uncs.append(tree_uncertainty(model, x_eval))
        pred_stack = np.vstack(preds)
        unc_stack = np.vstack(uncs)
        uncertainty = np.sqrt(np.var(pred_stack, axis=0) + np.mean(unc_stack ** 2, axis=0))
        return FitResult(np.mean(pred_stack, axis=0), uncertainty, models, [])
    model = make_estimator(model_name, seed, len(y_train))
    model.fit(x_train, y_train)
    if model_name == "gpr":
        pred, std = model.predict(x_eval, return_std=True)
        return FitResult(np.asarray(pred), np.asarray(std), model, [])
    pred = model.predict(x_eval)
    return FitResult(np.asarray(pred), tree_uncertainty(model, x_eval), model, [])


def oof_late_features(df: pd.DataFrame, combo: tuple[str, ...], train_idx: np.ndarray, eval_idx: np.ndarray, y: np.ndarray, seed: int, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    kfold = KFold(n_splits=min(5, len(train_idx)), shuffle=True, random_state=seed)
    oof_cols, eval_cols = [], []
    for i, modality in enumerate(combo):
        cols = modality_columns(df, (modality,))
        x_all = x_matrix(df, cols)
        oof = np.zeros(len(train_idx), dtype=float)
        eval_folds = []
        for fold, (tr_pos, va_pos) in enumerate(kfold.split(train_idx)):
            tr = train_idx[tr_pos]
            va = train_idx[va_pos]
            res = fit_predict_early(model_name if model_name != "gpr" else "extratrees", x_all[tr], y[tr], x_all[va], seed + i * 100 + fold)
            oof[va_pos] = res.pred
            eval_folds.append(fit_predict_early(model_name if model_name != "gpr" else "extratrees", x_all[tr], y[tr], x_all[eval_idx], seed + i * 100 + fold + 999).pred)
        oof_cols.append(oof)
        eval_cols.append(np.mean(np.vstack(eval_folds), axis=0))
    return np.vstack(oof_cols).T, np.vstack(eval_cols).T


def fit_predict_late(df: pd.DataFrame, combo: tuple[str, ...], train_idx: np.ndarray, eval_idx: np.ndarray, y: np.ndarray, seed: int, model_name: str) -> FitResult:
    oof, ev = oof_late_features(df, combo, train_idx, eval_idx, y, seed, model_name)
    meta = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 25)))])
    meta.fit(oof, y[train_idx])
    pred = meta.predict(ev)
    return FitResult(np.asarray(pred), np.std(ev, axis=1) if ev.shape[1] > 1 else np.zeros(ev.shape[0]), meta, [])


def fit_predict(df: pd.DataFrame, combo: tuple[str, ...], model_name: str, fusion: str, train_idx: np.ndarray, eval_idx: np.ndarray, y: np.ndarray, seed: int) -> FitResult:
    if fusion == "late" and len(combo) > 1:
        return fit_predict_late(df, combo, train_idx, eval_idx, y, seed, model_name)
    cols = modality_columns(df, combo)
    x = x_matrix(df, cols)
    res = fit_predict_early(model_name, x[train_idx], y[train_idx], x[eval_idx], seed)
    res.feature_names = cols
    return res


def split_indices(df: pd.DataFrame, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    y = pd.to_numeric(df["U_eV"], errors="coerce").to_numpy(float)
    bins = pd.qcut(pd.Series(y).rank(method="first"), q=min(5, len(df)), labels=False, duplicates="drop")
    return train_test_split(idx, test_size=test_size, random_state=seed, stratify=np.asarray(bins))


def initial_pool_split(train_pool: np.ndarray, y: np.ndarray, initial_size: int, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    if len(train_pool) <= initial_size:
        return list(train_pool), []
    bins = pd.qcut(pd.Series(y[train_pool]).rank(method="first"), q=min(4, len(train_pool)), labels=False, duplicates="drop")
    try:
        init, pool = train_test_split(train_pool, train_size=initial_size, random_state=seed, stratify=np.asarray(bins))
    except Exception:
        init = rng.choice(train_pool, size=initial_size, replace=False)
        pool = np.asarray([i for i in train_pool if i not in set(init)])
    return list(map(int, init)), list(map(int, pool))


def diversity_scores(df: pd.DataFrame, combo: tuple[str, ...], labeled: list[int], pool: list[int]) -> np.ndarray:
    cols = modality_columns(df, combo)
    x = x_matrix(df, cols)
    imp = SimpleImputer(strategy="median").fit(x[labeled])
    scaler = StandardScaler().fit(imp.transform(x[labeled]))
    xl = scaler.transform(imp.transform(x[labeled]))
    xp = scaler.transform(imp.transform(x[pool]))
    d = pairwise_distances(xp, xl)
    return np.min(d, axis=1)


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return v
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def select_batch(strategy: str, df: pd.DataFrame, combo: tuple[str, ...], labeled: list[int], pool: list[int], uncertainty: np.ndarray, batch_size: int, rng: np.random.Generator) -> tuple[list[int], np.ndarray]:
    if not pool:
        return [], np.array([])
    if strategy == "random":
        chosen = rng.choice(pool, size=min(batch_size, len(pool)), replace=False)
        return list(map(int, chosen)), np.zeros(len(pool))
    div = diversity_scores(df, combo, labeled, pool)
    if strategy == "uncertainty":
        score = minmax(uncertainty)
    elif strategy == "diversity":
        score = minmax(div)
    else:
        score = 0.65 * minmax(uncertainty) + 0.35 * minmax(div)
    order = np.argsort(score)[::-1][: min(batch_size, len(pool))]
    return [int(pool[i]) for i in order], score


def plot_outputs(outdir: Path, learning: pd.DataFrame, summary: pd.DataFrame) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    for metric, fname in [("MAE", "MAE_learning_curve.png"), ("R2", "R2_learning_curve.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        key_blocks = ["comp", "struct", "electronic", "comp+struct", "comp+electronic", "struct+electronic", "comp+struct+electronic"]
        sub = learning[(learning["model"].eq("ensemble")) & (learning["fusion"].eq("early")) & (learning["strategy"].eq("hybrid"))]
        for block in key_blocks:
            g = sub[sub["modality_block"].eq(block)].groupby("round")[metric].mean()
            if not g.empty:
                ax.plot(g.index, g.values, marker="o", linewidth=1.6, label=block)
        ax.set_xlabel("Active-learning round")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=180)
        plt.close(fig)

    best = summary[summary["round"].eq(summary["round"].max()) & summary["fusion"].eq("early")]
    best = best.sort_values("MAE_mean").drop_duplicates("modality_block")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(best["modality_block"], best["MAE_mean"], color="#4c78a8")
    ax.set_ylabel("Final-round MAE")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figdir / "tri_vs_bi_vs_single_bar.png", dpi=180)
    plt.close(fig)


def save_feature_importance(df: pd.DataFrame, outdir: Path, y: np.ndarray, seed: int) -> None:
    combo = ("comp", "struct", "electronic")
    cols = modality_columns(df, combo)
    if len(cols) == 0:
        return
    train_idx, test_idx = split_indices(df, seed, 0.2)
    x = x_matrix(df, cols)
    model = make_estimator("extratrees", seed, len(train_idx))
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    base = metric_dict(y[test_idx], pred)
    try:
        pi = permutation_importance(model, x[test_idx], y[test_idx], n_repeats=8, random_state=seed, scoring="neg_mean_absolute_error")
        imp = pd.DataFrame({"feature": cols, "importance_MAE_increase": pi.importances_mean, "importance_std": pi.importances_std}).sort_values("importance_MAE_increase", ascending=False)
    except Exception:
        est = model.named_steps["model"]
        imp = pd.DataFrame({"feature": cols, "importance_MAE_increase": getattr(est, "feature_importances_", np.zeros(len(cols))), "importance_std": 0.0}).sort_values("importance_MAE_increase", ascending=False)
    imp.to_csv(outdir / "feature_importance_tri_modal.csv", index=False)
    (outdir / "feature_importance_meta.json").write_text(json.dumps({"seed": seed, "test_metrics": base, "method": "permutation_importance_neg_mae_fallback_tree_importance"}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df["U_eV"] = pd.to_numeric(df["U_eV"], errors="coerce")
    if args.sensitivity == "review_only":
        df = df[df["U_eV"].le(8.0)].reset_index(drop=True)
    elif args.sensitivity == "remove_abnormal":
        df = df[(df["U_eV"].gt(0)) & (df["U_eV"].le(8.0))].reset_index(drop=True)
    else:
        df = df[df["U_eV"].notna()].reset_index(drop=True)
    if args.expected_n is not None and len(df) != args.expected_n:
        raise ValueError(
            f"Expected {args.expected_n} filtered numeric labels, found {len(df)} in {args.data}. "
            "Refusing to run with a mismatched dataset version."
        )
    y = df["U_eV"].to_numpy(float)

    combos = [c for c in MODALITY_COMBOS if all(m in args.modalities for m in c)]
    results, selected_rows = [], []
    skipped = []
    for seed in args.seeds:
        train_pool, test = split_indices(df, seed, args.test_size)
        initial, base_pool = initial_pool_split(train_pool, y, args.initial_size, seed)
        for combo in combos:
            block = "+".join(combo)
            for fusion in args.fusion:
                if fusion == "late" and len(combo) == 1:
                    continue
                for model_name in args.models:
                    for strategy in args.strategies:
                        labeled = list(initial)
                        pool = list(base_pool)
                        rng = np.random.default_rng(
                            seed + stable_seed_offset(block, model_name, strategy, fusion)
                        )
                        for round_i in range(args.rounds + 1):
                            eval_idx = np.asarray(list(test) + list(pool), dtype=int)
                            try:
                                eval_res = fit_predict(df, combo, model_name, fusion, np.asarray(labeled), eval_idx, y, seed + round_i)
                            except Exception as exc:
                                skipped.append({"seed": seed, "modality_block": block, "fusion": fusion, "model": model_name, "strategy": strategy, "round": round_i, "reason": str(exc)})
                                break
                            test_pred = eval_res.pred[: len(test)]
                            test_unc = eval_res.uncertainty[: len(test)]
                            pool_pred = eval_res.pred[len(test) :]
                            pool_unc = eval_res.uncertainty[len(test) :]
                            row = {
                                "seed": seed,
                                "modality_block": block,
                                "fusion": fusion,
                                "model": model_name,
                                "strategy": strategy,
                                "round": round_i,
                                "train_size": len(labeled),
                                "pool_size": len(pool),
                                "test_size": len(test),
                            }
                            row.update(metric_dict(y[test], test_pred, test_unc))
                            if pool:
                                row["pool_uncertainty_mean"] = float(np.mean(pool_unc))
                                row["pool_uncertainty_p90"] = float(np.percentile(pool_unc, 90))
                            else:
                                pool_unc = np.array([])
                                row["pool_uncertainty_mean"] = np.nan
                                row["pool_uncertainty_p90"] = np.nan
                            results.append(row)
                            if round_i == args.rounds or not pool:
                                continue
                            chosen, score = select_batch(strategy, df, combo, labeled, pool, pool_unc, args.batch_size, rng)
                            pool_pos = {idx: i for i, idx in enumerate(pool)}
                            for idx in chosen:
                                selected_rows.append(
                                    {
                                        "seed": seed,
                                        "modality_block": block,
                                        "fusion": fusion,
                                        "model": model_name,
                                        "strategy": strategy,
                                        "round": round_i + 1,
                                        "selected_index": idx,
                                        "material_id": df.loc[idx, "id"],
                                        "formula": df.loc[idx, "formula"],
                                        "target_element": df.loc[idx, "B_elem"],
                                        "U_eV_hidden_revealed_in_simulation": y[idx],
                                        "selection_score": float(score[pool_pos[idx]]) if len(score) and idx in pool_pos else np.nan,
                                        "selection_uncertainty": float(pool_unc[pool_pos[idx]]) if len(pool_unc) and idx in pool_pos else np.nan,
                                    }
                                )
                            labeled.extend(chosen)
                            chosen_set = set(chosen)
                            pool = [idx for idx in pool if idx not in chosen_set]

    results_df = pd.DataFrame(results)
    selected_df = pd.DataFrame(selected_rows)
    summary = (
        results_df.groupby(["modality_block", "fusion", "model", "strategy", "round"], as_index=False)
        .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"), Spearman_mean=("Spearman", "mean"), uncertainty_error_spearman_mean=("uncertainty_error_spearman", "mean"), n=("seed", "count"))
    )
    final_round = results_df["round"].max() if not results_df.empty else 0
    summary_by_modality = (
        results_df[results_df["round"].eq(final_round)]
        .groupby(["modality_block", "fusion", "model", "strategy"], as_index=False)
        .agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"), Spearman_mean=("Spearman", "mean"), n=("seed", "count"))
        .sort_values("MAE_mean")
    )
    results_df.to_csv(args.output / "results.csv", index=False)
    summary_by_modality.to_csv(args.output / "summary_by_modality.csv", index=False)
    summary.to_csv(args.output / "learning_curve.csv", index=False)
    selected_df.to_csv(args.output / "selected_samples.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.output / "skipped_runs.csv", index=False)
    plot_outputs(args.output, results_df, summary)
    save_feature_importance(df, args.output, y, args.seeds[0])

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "n_samples": int(len(df)),
        "modalities": args.modalities,
        "models_requested": args.models,
        "strategies": args.strategies,
        "fusion": args.fusion,
        "initial_size": args.initial_size,
        "batch_size": args.batch_size,
        "rounds": args.rounds,
        "seeds": args.seeds,
        "test_size": args.test_size,
        "sensitivity": args.sensitivity,
        "expected_n": args.expected_n,
        "policy_rng_seed": "seed + blake2b(block, model, strategy, fusion) modulo 100000",
        "leakage_controls": [
            "same seed-specific train/pool/test split for all modality/model/strategy comparisons",
            "initial labeled set fixed per seed before strategy simulation",
            "imputer/scaler/model fitted only on current labeled set each round",
            "test labels used only for evaluation",
        ],
        "outputs": ["results.csv", "summary_by_modality.csv", "learning_curve.csv", "selected_samples.csv", "figures/*.png"],
    }
    (args.output / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary_by_modality.head(20).to_string(index=False))
    print(f"Wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
