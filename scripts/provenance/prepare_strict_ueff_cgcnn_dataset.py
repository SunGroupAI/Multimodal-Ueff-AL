#!/usr/bin/env python
"""Prepare strict ABO3 Ueff labels as CGCNN-compatible CSV files.

Source files are read-only. Derived train/val/test/all CSVs and metadata are
written under ``processed/cgcnn_datasets`` by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def atom_init_path() -> Path:
    return repo_root() / "con-CDVAE" / "Con-CDVAE" / "cgcnn" / "data" / "mptest" / "atom_init.json"


def read_cif(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace")


def stratify_bins(y: pd.Series, n_bins: int = 5) -> pd.Series | None:
    try:
        bins = pd.qcut(y, q=min(n_bins, y.nunique()), labels=False, duplicates="drop")
    except ValueError:
        return None
    counts = bins.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    return bins


def split_labeled(
    df: pd.DataFrame,
    seed: int,
    val_fraction: float,
    test_fraction: float,
    stratify_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_bins = stratify_bins(df["U_eV"]) if stratify_target else None
    train_val, test = train_test_split(
        df,
        test_size=test_fraction,
        random_state=seed,
        stratify=y_bins,
    )
    y_bins_train_val = stratify_bins(train_val["U_eV"]) if stratify_target else None
    val_rel = val_fraction / max(1e-12, 1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=val_rel,
        random_state=seed + 17,
        stratify=y_bins_train_val,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_rows(feature_table: Path, structure_dir: Path, max_u: float | None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(feature_table)
    required = {"id", "formula", "U_eV"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise ValueError("Missing required columns in %s: %s" % (feature_table, missing_cols))

    rows = []
    missing_cifs = []
    excluded_high_u = []
    for _, row in df.iterrows():
        formula = str(row["formula"])
        if "struct_path" in df.columns and pd.notna(row.get("struct_path")):
            cif_path = Path(str(row["struct_path"]))
            if not cif_path.is_absolute():
                cif_path = repo_root() / cif_path
        else:
            cif_path = structure_dir / ("%s.cif" % formula)
        if not cif_path.exists():
            missing_cifs.append({"id": str(row["id"]), "formula": formula, "path": str(cif_path)})
            continue
        u_value = row["U_eV"]
        if pd.notna(u_value) and max_u is not None and float(u_value) > max_u:
            excluded_high_u.append({"id": str(row["id"]), "formula": formula, "U_eV": float(u_value)})
            continue
        rows.append(
            {
                "material_id": str(row["id"]),
                "formula": formula,
                "cif": read_cif(cif_path),
                "U_eV": np.nan if pd.isna(u_value) else float(u_value),
                "source_cif": str(cif_path),
            }
        )

    out = pd.DataFrame(rows)
    meta = {
        "input_feature_table": str(feature_table),
        "input_structure_dir": str(structure_dir),
        "max_u": max_u,
        "n_input_rows": int(len(df)),
        "n_output_rows": int(len(out)),
        "n_labeled_rows": int(out["U_eV"].notna().sum()) if "U_eV" in out else 0,
        "missing_cif_count": len(missing_cifs),
        "missing_cif_examples": missing_cifs[:10],
        "excluded_high_u_count": len(excluded_high_u),
        "excluded_high_u_examples": excluded_high_u[:10],
    }
    return out, meta


def write_split_files(df: pd.DataFrame, outdir: Path, seed: int, val_fraction: float, test_fraction: float, stratify_target: bool) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    labeled = df[df["U_eV"].notna()].reset_index(drop=True)
    unlabeled = df[df["U_eV"].isna()].reset_index(drop=True)
    train, val, test = split_labeled(labeled, seed, val_fraction, test_fraction, stratify_target)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part[["material_id", "cif", "U_eV"]].to_csv(outdir / ("%s.csv" % name), index=False)

    all_rows = df.copy()
    all_rows["U_eV"] = all_rows["U_eV"].fillna(0.0)
    all_rows[["material_id", "cif", "U_eV"]].to_csv(outdir / "all.csv", index=False)
    df[["material_id", "formula", "U_eV", "source_cif"]].to_csv(outdir / "metadata_rows.csv", index=False)
    unlabeled[["material_id", "formula", "source_cif"]].to_csv(outdir / "unlabeled_pool_rows.csv", index=False)

    atom_init = atom_init_path()
    if not atom_init.exists():
        raise FileNotFoundError("Missing atom_init.json: %s" % atom_init)
    shutil.copyfile(atom_init, outdir / "atom_init.json")
    return {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "stratify_target": stratify_target,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_unlabeled_pool": int(len(unlabeled)),
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_table",
        type=Path,
        default=root / "processed" / "ueff_multimodal_abo3_clean_20260622" / "features_ueff_multimodal_strict_abo3_only.parquet",
    )
    parser.add_argument("--structure_dir", type=Path, default=root / "structures_ABO3_only")
    parser.add_argument("--outdir", type=Path, default=root / "processed" / "cgcnn_datasets" / "ueff_strict_abo3_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument("--max_u", type=float, default=15.0)
    parser.add_argument("--no_stratify_target", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df, meta = load_rows(args.feature_table, args.structure_dir, args.max_u)
    split_meta = write_split_files(
        df=df,
        outdir=args.outdir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        stratify_target=not args.no_stratify_target,
    )
    meta.update(
        {
            "output_dir": str(args.outdir),
            "split": split_meta,
            "label_column": "U_eV",
            "feature_construction_uses_labels": False,
            "reproducibility_note": (
                "Run scripts/prepare_strict_ueff_cgcnn_dataset.py with the recorded arguments. "
                "It reads the strict ABO3 feature table and source CIFs read-only, then writes "
                "CGCNN-compatible derived CSVs plus metadata."
            ),
        }
    )
    (args.outdir / "cgcnn_strict_dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote strict CGCNN dataset to %s" % args.outdir)
    print("Rows: total=%d labeled=%d train=%d val=%d test=%d" % (
        meta["n_output_rows"],
        meta["n_labeled_rows"],
        split_meta["n_train"],
        split_meta["n_val"],
        split_meta["n_test"],
    ))


if __name__ == "__main__":
    main()
