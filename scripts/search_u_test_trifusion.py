#!/usr/bin/env python
"""Search all-U_test element/graph/text fusion variants without test leakage."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "Ueff_all_U_test_enhanced_modalities.csv")
    parser.add_argument("--knowledge", type=Path, default=root / "processed" / "u_knowledge_cards.csv")
    parser.add_argument("--cgcnn_dir", type=Path, default=root / "experiments" / "cgcnn_u_test_all_structures_20260623")
    parser.add_argument("--split_dir_pattern", default=str(root / "processed" / "cgcnn_datasets" / "u_test_all_structures_seed{seed}"))
    parser.add_argument("--outdir", type=Path, default=root / "experiments" / "u_test_trifusion_search_20260623")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    sp = np.nan
    if len(y) > 2 and np.std(y) > 0 and np.std(pred) > 0:
        sp = float(spearmanr(y, pred).correlation)
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "Spearman": sp,
    }


def numeric_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)


def one_hot(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    parts = []
    for c in cols:
        if c in df:
            parts.append(pd.get_dummies(df[c].fillna("NEED_DATA").astype(str), prefix=f"{prefix}_{c}", dtype=float))
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)


def fit_text(train_text: pd.Series, test_text: pd.Series, seed: int, max_features: int, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    xtr = vec.fit_transform(train_text.fillna("").astype(str))
    xte = vec.transform(test_text.fillna("").astype(str))
    k = min(n_components, xtr.shape[0] - 1, xtr.shape[1] - 1)
    if k < 2:
        return xtr.toarray(), xte.toarray()
    svd = TruncatedSVD(n_components=k, random_state=seed)
    return svd.fit_transform(xtr), svd.transform(xte)


def load_cgcnn(df: pd.DataFrame, cgcnn_dir: Path, seed: int) -> pd.DataFrame:
    idx_path = cgcnn_dir / f"seed{seed}" / "cgcnn_embedding_index.csv"
    emb_path = cgcnn_dir / f"seed{seed}" / "cgcnn_embeddings.npy"
    if not idx_path.exists() or not emb_path.exists():
        return pd.DataFrame(index=df.index)
    idx = pd.read_csv(idx_path)
    emb = np.load(emb_path)
    emb_df = pd.DataFrame(emb, columns=[f"cgcnn__{i:03d}" for i in range(emb.shape[1])])
    emb_df["id"] = idx["material_id"].astype(str)
    return df[["id"]].merge(emb_df, on="id", how="left").drop(columns=["id"])


def model_templates(seed: int) -> dict[str, object]:
    return {
        "et_leaf1": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1, max_features=0.7, random_state=seed, n_jobs=-1))]),
        "et_leaf2": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=seed + 1, n_jobs=-1))]),
        "rf": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features="sqrt", random_state=seed + 2, n_jobs=-1))]),
        "hgb": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=220, learning_rate=0.035, l2_regularization=0.06, random_state=seed + 3))]),
        "ridge": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 25)))]),
    }


def expert_oof_test(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, seed: int, folds: int, model_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kf = KFold(n_splits=min(folds, len(ytr)), shuffle=True, random_state=seed)
    oof = pd.DataFrame(index=np.arange(len(ytr)))
    test = pd.DataFrame(index=np.arange(xte.shape[0]))
    templates = model_templates(seed)
    for name in model_names:
        tpl = templates[name]
        pred = np.zeros(len(ytr))
        tpreds = []
        for fold, (tr, va) in enumerate(kf.split(xtr)):
            m = clone(tpl)
            m.fit(xtr[tr], ytr[tr])
            pred[va] = m.predict(xtr[va])
            tpreds.append(m.predict(xte))
        final = clone(tpl)
        final.fit(xtr, ytr)
        tpreds.append(final.predict(xte))
        oof[name] = pred
        test[name] = np.mean(np.vstack(tpreds), axis=0)
    return oof, test


def add_meta_features(preds: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    out = preds.copy()
    for g, cols in groups.items():
        out[f"{g}_mean"] = preds[cols].mean(axis=1)
        out[f"{g}_std"] = preds[cols].std(axis=1).fillna(0.0)
    means = [f"{g}_mean" for g in groups]
    for i, a in enumerate(means):
        for b in means[i + 1 :]:
            out[f"absdiff_{a}_{b}"] = np.abs(out[a] - out[b])
    return out


def fit_meta(oof: pd.DataFrame, ytr: np.ndarray, test: pd.DataFrame, seed: int) -> tuple[np.ndarray, str]:
    x, xt = oof.to_numpy(float), test.to_numpy(float)
    best_name, best_mae, best_tpl = "", np.inf, None
    kf = KFold(n_splits=min(5, len(ytr)), shuffle=True, random_state=seed + 777)
    templates = {
        "meta_ridge": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4, 4, 25)))]),
        "meta_et": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=4, max_features="sqrt", random_state=seed, n_jobs=-1))]),
        "meta_hgb": Pipeline([("impute", SimpleImputer(strategy="median")), ("model", HistGradientBoostingRegressor(max_iter=120, learning_rate=0.035, l2_regularization=0.1, random_state=seed))]),
    }
    for name, tpl in templates.items():
        pred = np.zeros(len(ytr))
        for tr, va in kf.split(x):
            m = clone(tpl)
            m.fit(x[tr], ytr[tr])
            pred[va] = m.predict(x[va])
        mae = mean_absolute_error(ytr, pred)
        if mae < best_mae:
            best_name, best_mae, best_tpl = name, mae, tpl
    final = clone(best_tpl)
    final.fit(x, ytr)
    return final.predict(xt), best_name


def train_eval_blocks(blocks: dict[str, tuple[np.ndarray, np.ndarray]], ytr: np.ndarray, yte: np.ndarray, seed: int, folds: int, out_rows: list[dict], variant_name: str) -> None:
    all_oof, all_test, groups = [], [], {}
    for modality, (xtr, xte) in blocks.items():
        model_names = ["et_leaf1", "et_leaf2", "hgb", "ridge"] if xtr.shape[1] < 160 else ["et_leaf2", "ridge"]
        oof, te = expert_oof_test(xtr, ytr, xte, seed + len(all_oof) * 100, folds, model_names)
        oof = oof.add_prefix(modality + "__")
        te = te.add_prefix(modality + "__")
        groups[modality] = list(oof.columns)
        all_oof.append(oof)
        all_test.append(te)
        meta_oof = add_meta_features(oof, {modality: groups[modality]})
        meta_te = add_meta_features(te, {modality: groups[modality]})
        pred, meta = fit_meta(meta_oof, ytr, meta_te, seed)
        out_rows.append({"seed": seed, "variant": variant_name, "block": modality, "fusion": "single_stacked", "meta_model": meta, **metrics(yte, pred)})

    oof_all = pd.concat(all_oof, axis=1)
    test_all = pd.concat(all_test, axis=1)
    combos = [("element", "graph"), ("element", "text"), ("graph", "text"), ("element", "graph", "text")]
    for combo in combos:
        cols = [c for m in combo for c in groups[m]]
        meta_oof = add_meta_features(oof_all[cols], {m: groups[m] for m in combo})
        meta_te = add_meta_features(test_all[cols], {m: groups[m] for m in combo})
        pred, meta = fit_meta(meta_oof, ytr, meta_te, seed)
        out_rows.append({"seed": seed, "variant": variant_name, "block": "+".join(combo), "fusion": "late_oof_stacking", "meta_model": meta, **metrics(yte, pred)})

    # Residual tri-modal: text+element backbone plus graph residual, alpha selected only on train OOF.
    base_cols = groups["element"] + groups["text"]
    graph_cols = groups["graph"]
    base_oof = add_meta_features(oof_all[base_cols], {"element": groups["element"], "text": groups["text"]})
    base_te = add_meta_features(test_all[base_cols], {"element": groups["element"], "text": groups["text"]})
    graph_oof = add_meta_features(oof_all[graph_cols], {"graph": groups["graph"]})
    graph_te = add_meta_features(test_all[graph_cols], {"graph": groups["graph"]})
    base_train_pred, _ = fit_meta(base_oof, ytr, base_oof, seed + 1300)
    base_test_pred, base_meta = fit_meta(base_oof, ytr, base_te, seed + 1300)
    residual = ytr - base_train_pred
    residual_train_pred, _ = fit_meta(graph_oof, residual, graph_oof, seed + 1400)
    residual_test_pred, residual_meta = fit_meta(graph_oof, residual, graph_te, seed + 1400)
    best_alpha, best_mae = 0.0, mean_absolute_error(ytr, base_train_pred)
    for alpha in np.linspace(0, 1, 11):
        mae = mean_absolute_error(ytr, base_train_pred + alpha * residual_train_pred)
        if mae < best_mae:
            best_mae, best_alpha = mae, float(alpha)
    pred = base_test_pred + best_alpha * residual_test_pred
    out_rows.append({"seed": seed, "variant": variant_name, "block": "element+graph+text", "fusion": "graph_residual_on_element_text", "meta_model": f"base={base_meta};residual={residual_meta};alpha={best_alpha:.2f}", **metrics(yte, pred)})


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    df["id"] = df["id"].astype(str)
    y = pd.to_numeric(df["U_eV"], errors="coerce").to_numpy(float)
    knowledge = pd.read_csv(args.knowledge, encoding="utf-8-sig")
    ktext = knowledge[["material_id", "text_summary_for_embedding"]].rename(columns={"material_id": "id"})
    df = df.merge(ktext, on="id", how="left")
    df["text_all"] = (
        df.get("text_summary_for_embedding", "").fillna("").astype(str)
        + " "
        + df.get("raw_text", "").fillna("").astype(str)
        + " formula "
        + df["formula"].astype(str)
        + " B-site "
        + df["B_elem"].astype(str)
    )

    rows = []
    for seed in args.seeds:
        split_dir = Path(args.split_dir_pattern.format(seed=seed))
        if split_dir.exists():
            train_ids = set(pd.read_csv(split_dir / "train.csv")["material_id"].astype(str)) | set(pd.read_csv(split_dir / "val.csv")["material_id"].astype(str))
            test_ids = set(pd.read_csv(split_dir / "test.csv")["material_id"].astype(str))
            train_idx = df.index[df["id"].isin(train_ids)].to_numpy()
            test_idx = df.index[df["id"].isin(test_ids)].to_numpy()
        else:
            bins = pd.qcut(pd.Series(y).rank(method="first"), q=5, labels=False, duplicates="drop")
            train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=seed, stratify=np.asarray(bins))
        ytr, yte = y[train_idx], y[test_idx]

        comp_cols = [c for c in df.columns if c.startswith("comp__")]
        comp_onehot = one_hot(df, ["A_elem", "B_elem"], "elem")
        elem_basic = np.hstack([numeric_matrix(df, comp_cols), comp_onehot.to_numpy(float)])

        struct_cols = [c for c in df.columns if c.startswith("struct__") or c.startswith("graph_enh__")]
        graph_struct = numeric_matrix(df, struct_cols)
        cgcnn = load_cgcnn(df, args.cgcnn_dir, seed)
        graph_cgcnn = cgcnn.to_numpy(float) if not cgcnn.empty else graph_struct
        graph_hybrid = np.hstack([graph_struct, graph_cgcnn])

        elec_cols = [c for c in df.columns if c.startswith("elec__")]
        text_numeric = numeric_matrix(df, elec_cols)
        text_svd32_train, text_svd32_test = fit_text(df.loc[train_idx, "text_all"], df.loc[test_idx, "text_all"], seed, 768, 32)
        text_svd64_train, text_svd64_test = fit_text(df.loc[train_idx, "text_all"], df.loc[test_idx, "text_all"], seed, 1500, 64)

        variants = {
            "elem_basic__graph_struct__text_numeric": {
                "element": (elem_basic[train_idx], elem_basic[test_idx]),
                "graph": (graph_struct[train_idx], graph_struct[test_idx]),
                "text": (text_numeric[train_idx], text_numeric[test_idx]),
            },
            "elem_basic__graph_struct__text_svd32": {
                "element": (elem_basic[train_idx], elem_basic[test_idx]),
                "graph": (graph_struct[train_idx], graph_struct[test_idx]),
                "text": (text_svd32_train, text_svd32_test),
            },
            "elem_basic__graph_hybrid_cgcnn__text_svd32": {
                "element": (elem_basic[train_idx], elem_basic[test_idx]),
                "graph": (graph_hybrid[train_idx], graph_hybrid[test_idx]),
                "text": (text_svd32_train, text_svd32_test),
            },
            "elem_basic__graph_hybrid_cgcnn__text_numeric_plus_svd64": {
                "element": (elem_basic[train_idx], elem_basic[test_idx]),
                "graph": (graph_hybrid[train_idx], graph_hybrid[test_idx]),
                "text": (np.hstack([text_numeric[train_idx], text_svd64_train]), np.hstack([text_numeric[test_idx], text_svd64_test])),
            },
        }
        for name, blocks in variants.items():
            train_eval_blocks(blocks, ytr, yte, seed, args.folds, rows, name)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["variant", "block", "fusion"], as_index=False)
        .agg(n=("seed", "count"), MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"), R2_mean=("R2", "mean"), Spearman_mean=("Spearman", "mean"))
        .sort_values("MAE_mean")
    )
    detail.to_csv(args.outdir / "trifusion_search_by_seed.csv", index=False)
    summary.to_csv(args.outdir / "trifusion_search_summary.csv", index=False)
    tri = detail[detail["block"].eq("element+graph+text")]
    best_nontri = detail[~detail["block"].eq("element+graph+text")].groupby(["seed", "variant"])["MAE"].min().reset_index(name="best_nontri_MAE")
    cmp = tri.merge(best_nontri, on=["seed", "variant"], how="left")
    cmp["tri_beats_best_nontri"] = cmp["MAE"] < cmp["best_nontri_MAE"]
    cmp.to_csv(args.outdir / "tri_vs_best_nontri_by_seed.csv", index=False)
    meta = {
        "data": str(args.data),
        "knowledge": str(args.knowledge),
        "cgcnn_dir": str(args.cgcnn_dir),
        "seeds": args.seeds,
        "feature_construction_uses_test_labels": False,
        "text_vectorizers_fit_on_train_only": True,
        "fusion_selection": "Meta models and residual alpha are selected inside train OOF only.",
    }
    (args.outdir / "trifusion_search_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.head(30).to_string(index=False))
    print("Tri beats best non-tri:", int(cmp["tri_beats_best_nontri"].sum()), "/", len(cmp))
    print("Wrote outputs to", args.outdir)


if __name__ == "__main__":
    main()
