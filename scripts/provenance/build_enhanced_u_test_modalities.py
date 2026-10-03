#!/usr/bin/env python
"""Build enhanced graph/structure modality features for all U_test labels.

This script uses only structure files and formula chemistry. It does not use
Ueff labels to construct features.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element, Structure


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "Ueff_all_U_test_modalities.csv")
    parser.add_argument("--strain_dir", type=Path, default=root / "ALIGNN" / "abo3_strain_con" / "cif_structures")
    parser.add_argument("--out", type=Path, default=root / "data" / "processed" / "Ueff_all_U_test_enhanced_modalities.csv")
    return parser.parse_args()


def safe_float(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def safe_el(symbol: object) -> Element | None:
    try:
        return Element(str(symbol))
    except Exception:
        return None


def ionic_radius(symbol: object) -> float:
    el = safe_el(symbol)
    if el is None:
        return np.nan
    for attr in ["average_ionic_radius", "atomic_radius"]:
        try:
            val = getattr(el, attr)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    return np.nan


def load_structure(path: object) -> Structure | None:
    try:
        p = Path(str(path))
        if not p.exists():
            return None
        return Structure.from_file(p)
    except Exception:
        return None


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den <= 1e-12:
        return np.nan
    cosv = np.clip(float(np.dot(v1, v2) / den), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def local_features(struct: Structure | None, a_elem: str, b_elem: str) -> dict[str, float]:
    out = {}
    keys = [
        "graph_enh__BO_mean",
        "graph_enh__BO_std",
        "graph_enh__BO_min",
        "graph_enh__BO_max",
        "graph_enh__BO_range",
        "graph_enh__B_coord_mean",
        "graph_enh__B_coord_std",
        "graph_enh__OBO_angle_mean",
        "graph_enh__OBO_angle_std",
        "graph_enh__OBO_angle_var90",
        "graph_enh__BOB_angle_mean",
        "graph_enh__BOB_angle_std",
        "graph_enh__oct_distortion_index",
        "graph_enh__quadratic_elongation",
        "graph_enh__bond_valence_sum",
        "graph_enh__goldschmidt_t",
        "graph_enh__octahedral_factor",
        "graph_enh__cell_a",
        "graph_enh__cell_b",
        "graph_enh__cell_c",
        "graph_enh__cell_alpha",
        "graph_enh__cell_beta",
        "graph_enh__cell_gamma",
    ]
    for k in keys:
        out[k] = np.nan
    if struct is None:
        return out
    out["graph_enh__cell_a"] = float(struct.lattice.a)
    out["graph_enh__cell_b"] = float(struct.lattice.b)
    out["graph_enh__cell_c"] = float(struct.lattice.c)
    out["graph_enh__cell_alpha"] = float(struct.lattice.alpha)
    out["graph_enh__cell_beta"] = float(struct.lattice.beta)
    out["graph_enh__cell_gamma"] = float(struct.lattice.gamma)

    b_dists, b_coords, obo_angles, bob_angles = [], [], [], []
    distortion, qelong, bvs = [], [], []
    b_sites = [site for site in struct if site.specie.symbol == b_elem]
    o_sites = [site for site in struct if site.specie.symbol == "O"]
    for b_site in b_sites:
        oxy = sorted([n for n in struct.get_neighbors(b_site, 3.3) if n.specie.symbol == "O"], key=lambda n: float(n.nn_distance))
        if not oxy:
            continue
        oxy = oxy[:8]
        ds = np.asarray([float(n.nn_distance) for n in oxy], dtype=float)
        b_dists.extend(ds.tolist())
        b_coords.append(float(len(ds)))
        mean_d = float(np.mean(ds))
        if mean_d > 0:
            distortion.append(float(np.mean(np.abs(ds - mean_d) / mean_d)))
            qelong.append(float(np.mean((ds / mean_d) ** 2)))
        for i in range(len(oxy)):
            for j in range(i + 1, len(oxy)):
                obo_angles.append(angle(oxy[i].coords, b_site.coords, oxy[j].coords))
        rb = ionic_radius(b_elem)
        r0 = (rb + 1.40) if np.isfinite(rb) else mean_d
        bvs.append(float(np.sum(np.exp((r0 - ds) / 0.37))))

    for o_site in o_sites:
        b_neigh = sorted([n for n in struct.get_neighbors(o_site, 3.5) if n.specie.symbol == b_elem], key=lambda n: float(n.nn_distance))[:4]
        for i in range(len(b_neigh)):
            for j in range(i + 1, len(b_neigh)):
                bob_angles.append(angle(b_neigh[i].coords, o_site.coords, b_neigh[j].coords))

    arr = np.asarray(b_dists, dtype=float)
    if arr.size:
        out["graph_enh__BO_mean"] = float(np.nanmean(arr))
        out["graph_enh__BO_std"] = float(np.nanstd(arr))
        out["graph_enh__BO_min"] = float(np.nanmin(arr))
        out["graph_enh__BO_max"] = float(np.nanmax(arr))
        out["graph_enh__BO_range"] = float(np.nanmax(arr) - np.nanmin(arr))
    for name, values in [
        ("graph_enh__B_coord", b_coords),
        ("graph_enh__OBO_angle", obo_angles),
        ("graph_enh__BOB_angle", bob_angles),
    ]:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[f"{name}_mean"] = float(np.mean(vals))
            out[f"{name}_std"] = float(np.std(vals))
    obo = np.asarray(obo_angles, dtype=float)
    obo = obo[np.isfinite(obo)]
    if obo.size:
        out["graph_enh__OBO_angle_var90"] = float(np.mean((obo - 90.0) ** 2))
    if distortion:
        out["graph_enh__oct_distortion_index"] = float(np.mean(distortion))
    if qelong:
        out["graph_enh__quadratic_elongation"] = float(np.mean(qelong))
    if bvs:
        out["graph_enh__bond_valence_sum"] = float(np.mean(bvs))
    ra, rb, ro = ionic_radius(a_elem), ionic_radius(b_elem), 1.40
    if np.isfinite(ra) and np.isfinite(rb):
        out["graph_enh__goldschmidt_t"] = float((ra + ro) / (math.sqrt(2.0) * (rb + ro)))
        out["graph_enh__octahedral_factor"] = float(rb / ro)
    return out


def strain_paths(strain_dir: Path, material_id: str, formula: str) -> dict[int, Path]:
    out = {}
    if not strain_dir.exists():
        return out
    for p in strain_dir.glob(f"{material_id}_{formula}_*strain_*.cif"):
        m = re.search(r"strain_(-?\d+)\.cif$", p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def strain_response(strain_dir: Path, row: pd.Series) -> dict[str, float]:
    out = {
        "graph_enh__strain_n": 0.0,
        "graph_enh__volume_strain_slope": np.nan,
        "graph_enh__volume_strain_curvature": np.nan,
        "graph_enh__BO_strain_slope": np.nan,
        "graph_enh__BO_strain_curvature": np.nan,
    }
    paths = strain_paths(strain_dir, str(row["id"]), str(row["formula"]))
    vals = []
    for strain, path in sorted(paths.items()):
        s = load_structure(path)
        if s is None:
            continue
        lf = local_features(s, str(row.get("A_elem", "")), str(row.get("B_elem", "")))
        vals.append((float(strain), float(s.volume / max(len(s), 1)), safe_float(lf["graph_enh__BO_mean"])))
    out["graph_enh__strain_n"] = float(len(vals))
    if len(vals) >= 3:
        x = np.asarray([v[0] for v in vals], dtype=float)
        vol = np.asarray([v[1] for v in vals], dtype=float)
        coef = np.polyfit(x, vol, deg=2)
        out["graph_enh__volume_strain_curvature"] = float(coef[0])
        out["graph_enh__volume_strain_slope"] = float(coef[1])
        bo = np.asarray([v[2] for v in vals], dtype=float)
        mask = np.isfinite(bo)
        if mask.sum() >= 3:
            coef = np.polyfit(x[mask], bo[mask], deg=2)
            out["graph_enh__BO_strain_curvature"] = float(coef[0])
            out["graph_enh__BO_strain_slope"] = float(coef[1])
    return out


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    rows = []
    for _, row in df.iterrows():
        struct = load_structure(row.get("struct_path"))
        rec = {
            "id": row["id"],
            "formula": row["formula"],
        }
        rec.update(local_features(struct, str(row.get("A_elem", "")), str(row.get("B_elem", ""))))
        rec.update(strain_response(args.strain_dir, row))
        rows.append(rec)
    enh = pd.DataFrame(rows)
    out = df.merge(enh, on=["id", "formula"], how="left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    meta = {
        "input": str(args.data),
        "out": str(args.out),
        "strain_dir": str(args.strain_dir),
        "n_rows": int(len(out)),
        "enhanced_columns": [c for c in out.columns if c.startswith("graph_enh__")],
        "feature_construction_uses_labels": False,
    }
    (args.out.parent / "Ueff_all_U_test_enhanced_modalities_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
