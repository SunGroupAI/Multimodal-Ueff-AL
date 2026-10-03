#!/usr/bin/env python
"""Build a full U_test multimodal table independent of the old strict table.

The table is keyed by U_test.csv rows. It keeps every row that has a matching
CIF in the structure directory and writes derived features for downstream
physics-text, element, and CGCNN workflows.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element, Structure


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_folder(folder_name: object) -> tuple[str | None, str | None]:
    text = str(folder_name)
    m = re.match(r"^(mp-\d+)_([A-Za-z0-9]+)_", text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def safe_element(symbol: str | None) -> Element | None:
    if not symbol:
        return None
    try:
        return Element(symbol)
    except Exception:
        return None


def infer_sites(formula: str) -> dict[str, object]:
    out = {"A_elem": None, "B_elem": None, "is_standard_ABO3": False, "n_elements": np.nan}
    try:
        comp = Composition(formula)
    except Exception:
        return out
    items = [(el.symbol, float(amount)) for el, amount in comp.items()]
    out["n_elements"] = len(items)
    non_o = [(el, amt) for el, amt in items if el != "O"]
    match = re.match(r"^([A-Z][a-z]?)([A-Z][a-z]?)O3$", formula)
    if match:
        out["A_elem"], out["B_elem"] = match.group(1), match.group(2)
        out["is_standard_ABO3"] = True
        return out

    # For non-standard U_test rows, choose the U-bearing center as the first
    # transition/f-block element, otherwise the least abundant non-O element.
    center = None
    for el, _ in non_o:
        obj = safe_element(el)
        if obj is not None and obj.block in {"d", "f"}:
            center = el
            break
    if center is None and non_o:
        center = sorted(non_o, key=lambda x: (x[1], x[0]))[0][0]
    a_candidates = [el for el, _ in non_o if el != center]
    out["A_elem"] = a_candidates[0] if a_candidates else None
    out["B_elem"] = center
    return out


def element_props(symbol: str | None, prefix: str) -> dict[str, float]:
    el = safe_element(symbol)
    block_map = {"s": 1.0, "p": 2.0, "d": 3.0, "f": 4.0}
    common_ox = list(el.common_oxidation_states) if el else []
    return {
        f"{prefix}_Z": float(el.Z) if el else np.nan,
        f"{prefix}_group": float(el.group) if el and el.group else np.nan,
        f"{prefix}_period": float(el.row) if el else np.nan,
        f"{prefix}_block_id": block_map.get(str(el.block), np.nan) if el else np.nan,
        f"{prefix}_X": float(el.X) if el and el.X is not None else np.nan,
        f"{prefix}_atomic_radius": float(el.atomic_radius) if el and el.atomic_radius is not None else np.nan,
        f"{prefix}_common_ox_min": float(min(common_ox)) if common_ox else np.nan,
        f"{prefix}_common_ox_max": float(max(common_ox)) if common_ox else np.nan,
        f"{prefix}_n_common_ox": float(len(common_ox)),
    }


def structure_features(cif_path: Path, b_elem: str | None) -> dict[str, object]:
    out: dict[str, object] = {
        "struct_path": str(cif_path),
        "n_sites": np.nan,
        "volume": np.nan,
        "volume_per_atom": np.nan,
        "density": np.nan,
        "spacegroup_symbol": "NEED_DATA",
        "spacegroup_number": np.nan,
        "B_O_coordination": np.nan,
        "B_O_distance_mean": np.nan,
        "B_O_distance_std": np.nan,
        "B_O_distance_min": np.nan,
        "B_O_distance_max": np.nan,
    }
    if not cif_path.exists():
        return out
    try:
        struct = Structure.from_file(cif_path)
    except Exception:
        return out
    out["n_sites"] = float(len(struct))
    out["volume"] = float(struct.volume)
    out["volume_per_atom"] = float(struct.volume / max(len(struct), 1))
    try:
        out["density"] = float(struct.density)
    except Exception:
        pass
    try:
        sg, num = struct.get_space_group_info()
        out["spacegroup_symbol"] = sg
        out["spacegroup_number"] = float(num)
    except Exception:
        pass
    if b_elem:
        coords, dists = [], []
        for site in struct:
            if site.specie.symbol != b_elem:
                continue
            oxy = [float(n.nn_distance) for n in struct.get_neighbors(site, 3.1) if n.specie.symbol == "O"]
            if oxy:
                coords.append(len(oxy))
                dists.extend(oxy)
        if coords:
            out["B_O_coordination"] = float(np.mean(coords))
        if dists:
            arr = np.asarray(dists, dtype=float)
            out["B_O_distance_mean"] = float(arr.mean())
            out["B_O_distance_std"] = float(arr.std())
            out["B_O_distance_min"] = float(arr.min())
            out["B_O_distance_max"] = float(arr.max())
    return out


def read_text(formula: str, text_dir: Path) -> str:
    path = text_dir / f"{formula}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--u_test", type=Path, default=root / "U_test.csv")
    parser.add_argument("--structure_dir", type=Path, default=root / "structures_ABO3_only")
    parser.add_argument(
        "--extra_structure_dirs",
        type=Path,
        nargs="*",
        default=[root / "SCR" / "project" / "data" / "structures", root / "ALIGNN" / "abo3_strain_con" / "cif_structures"],
    )
    parser.add_argument("--text_dir", type=Path, default=root / "text_by_formula")
    parser.add_argument("--outdir", type=Path, default=root / "processed" / "u_test_full_multimodal_20260623")
    return parser.parse_args()


def find_structure_path(row: pd.Series, structure_roots: list[Path]) -> Path | None:
    formula = str(row["formula"])
    material_id = str(row["id"])
    direct_names = [f"{formula}.cif", f"{material_id}.cif"]
    for root in structure_roots:
        for name in direct_names:
            p = root / name
            if p.exists():
                return p
    # ALIGNN strain folders use names like mp-..._Formula_SG_strain_0.cif.
    for root in structure_roots:
        matches = sorted(root.glob(f"{material_id}_{formula}_*strain_0.cif"))
        if matches:
            return matches[0]
    for root in structure_roots:
        matches = sorted(root.glob(f"{material_id}_{formula}_*.cif"))
        if matches:
            return matches[0]
    return None


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.u_test, encoding="utf-8-sig")
    parsed = labels["folder_name"].apply(parse_folder)
    labels["id"] = parsed.apply(lambda x: x[0])
    labels["formula"] = parsed.apply(lambda x: x[1])
    labels["U_eV"] = pd.to_numeric(labels["Ueff_eV"], errors="coerce")
    labels = labels[labels["id"].notna() & labels["formula"].notna() & labels["U_eV"].notna()].copy()
    labels = labels.sort_values(["id", "formula"]).drop_duplicates(["id", "formula"], keep="last")

    rows = []
    missing = []
    structure_roots = [args.structure_dir] + list(args.extra_structure_dirs)
    for _, row in labels.iterrows():
        formula = str(row["formula"])
        cif_path = find_structure_path(row, structure_roots)
        if cif_path is None:
            missing.append({"id": row["id"], "formula": formula, "searched_roots": [str(p) for p in structure_roots]})
            continue
        sites = infer_sites(formula)
        rec: dict[str, object] = {
            "id": str(row["id"]),
            "formula": formula,
            "folder_name": str(row["folder_name"]),
            "folder_path": str(row.get("folder_path", "")),
            "output_wsy": str(row.get("output_wsy", "")),
            "u_test_status": str(row.get("status", "")),
            "U_eV": float(row["U_eV"]),
            "A_elem": sites["A_elem"],
            "B_elem": sites["B_elem"],
            "is_standard_ABO3": bool(sites["is_standard_ABO3"]),
            "n_elements": sites["n_elements"],
            "raw_text": read_text(formula, args.text_dir),
        }
        rec.update(element_props(rec["A_elem"], "A"))
        rec.update(element_props(rec["B_elem"], "B"))
        rec.update(structure_features(cif_path, rec["B_elem"]))
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_parquet = args.outdir / "features_u_test_full_multimodal.parquet"
    out_csv = args.outdir / "features_u_test_full_multimodal_index.csv"
    out.to_parquet(out_parquet, index=False)
    out[["id", "formula", "U_eV", "A_elem", "B_elem", "is_standard_ABO3", "struct_path"]].to_csv(out_csv, index=False)
    meta = {
        "u_test": str(args.u_test),
        "structure_dir": str(args.structure_dir),
        "extra_structure_dirs": [str(p) for p in args.extra_structure_dirs],
        "text_dir": str(args.text_dir),
        "out_parquet": str(out_parquet),
        "n_u_test_valid": int(len(labels)),
        "n_rows_with_cif": int(len(out)),
        "n_missing_cif": int(len(missing)),
        "missing_cif_examples": missing[:20],
        "n_standard_ABO3": int(out["is_standard_ABO3"].sum()) if not out.empty else 0,
        "feature_construction_uses_labels": False,
    }
    (args.outdir / "u_test_full_multimodal_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
