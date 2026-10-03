#!/usr/bin/env python3
"""Build structured U-related knowledge cards for ABO3 perovskites.

The cards are designed as a safer text modality for DFT+U active learning:
they summarize U-relevant chemistry, structure, spin/magnetism, and literature
priors without using the target DFT+U label in the embedding paragraph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import pandas as pd


CARD_FIELDS = [
    "material_id",
    "formula",
    "A_site",
    "B_site",
    "likely_A_oxidation_state",
    "likely_B_oxidation_state",
    "B_site_d_electron_count",
    "B_site_period",
    "B_site_group",
    "B_site_coordination_environment",
    "correlated_orbital",
    "expected_correlation_strength",
    "possible_spin_states",
    "magnetic_relevance",
    "expected_U_sensitivity",
    "band_gap_relevance",
    "metal_oxygen_hybridization_relevance",
    "literature_U_values",
    "literature_U_min",
    "literature_U_max",
    "literature_U_mean",
    "literature_U_median",
    "literature_U_count",
    "literature_source_count",
    "source_confidence",
    "text_summary_for_embedding",
    "warnings",
]

ID_CANDIDATES = ["material_id", "id", "mp_id", "MP-ID"]
FORMULA_CANDIDATES = ["formula", "formula_norm", "pretty_formula"]
STRUCTURE_CANDIDATES = ["structure_path", "struct_path", "cif_path", "structure_file"]
TARGET_COLUMNS = {"u_ev", "ueff_ev", "u", "target", "label"}

COMMON_A_OX = {
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Ag": 1,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Cd": 2,
    "Hg": 2,
    "Zn": 2,
    "La": 3,
    "Ce": 3,
    "Pr": 3,
    "Nd": 3,
    "Sm": 3,
    "Eu": 3,
    "Gd": 3,
    "Tb": 3,
    "Dy": 3,
    "Ho": 3,
    "Er": 3,
    "Tm": 3,
    "Yb": 3,
    "Lu": 3,
    "Y": 3,
    "Sc": 3,
}

NONMETAL_B = {"B", "C", "N", "P", "As", "Sb", "Se", "Te", "Si", "Ge", "Sn", "Pb", "Al", "In"}


@dataclass
class LiteratureHit:
    value: float
    source: str
    matched_by: str
    note: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing material table: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path).reset_index(drop=True)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires pyarrow or fastparquet. Use the project "
                "environment if available, for example "
                "python."
            ) from exc
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path).reset_index(drop=True)
    return pd.read_csv(path).reset_index(drop=True)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_col:
            return lower_to_col[candidate.lower()]
    return None


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def as_float(value: Any) -> float | None:
    value = clean_scalar(value)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def fmt_need(value: Any) -> str:
    value = clean_scalar(value)
    if value is None or value == "":
        return "NEED_DATA"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_formula(formula: str) -> dict[str, float]:
    formula = str(formula or "").strip()
    try:
        from pymatgen.core import Composition  # type: ignore

        comp = Composition(formula)
        return {str(el): float(amount) for el, amount in comp.get_el_amt_dict().items()}
    except Exception:
        parts = re.findall(r"([A-Z][a-z]?)([0-9.]*)", formula)
        out: dict[str, float] = {}
        for element, amount in parts:
            out[element] = out.get(element, 0.0) + float(amount or 1.0)
        return out


def infer_sites(row: pd.Series, formula: str) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    a_site = fmt_need(row.get("A_elem"))
    b_site = fmt_need(row.get("B_elem"))
    if a_site != "NEED_DATA" and b_site != "NEED_DATA":
        return a_site, b_site, warnings

    comp = parse_formula(formula)
    elements = [el for el in comp if el != "O"]
    if len(elements) >= 2:
        if a_site == "NEED_DATA":
            a_site = elements[0]
        if b_site == "NEED_DATA":
            b_site = elements[1]
    else:
        warnings.append("NEED_DATA: could not infer distinct A/B sites from formula.")
    return a_site, b_site, warnings


def element_info(symbol: str) -> tuple[int | None, int | None, str]:
    if not symbol or symbol == "NEED_DATA":
        return None, None, "NEED_DATA"
    try:
        from pymatgen.core import Element  # type: ignore

        el = Element(symbol)
        block = getattr(el, "block", None) or "NEED_DATA"
        return int(el.row), int(el.group), str(block)
    except Exception:
        return None, None, "NEED_DATA"


def likely_oxidation_states(row: pd.Series, a_site: str, b_site: str) -> tuple[int | None, int | None, list[str]]:
    warnings: list[str] = []
    b_from_row = as_float(row.get("oxid_B"))
    if b_from_row is not None:
        b_ox = int(round(b_from_row))
        a_ox = 6 - b_ox
        return a_ox, b_ox, warnings

    a_ox = COMMON_A_OX.get(a_site)
    if a_ox is None:
        warnings.append("NEED_DATA: ambiguous A-site oxidation state; used no fixed oxidation assignment.")
        return None, None, warnings
    b_ox = 6 - a_ox
    if b_ox < 1 or b_ox > 6:
        warnings.append("NEED_DATA: inferred B-site oxidation state is outside a typical ABO3 range.")
        return a_ox, None, warnings
    return a_ox, b_ox, warnings


def d_electron_count(row: pd.Series, b_site: str, b_ox: int | None, group: int | None, block: str) -> tuple[int | None, list[str]]:
    warnings: list[str] = []
    from_row = as_float(row.get("d_electron_B"))
    if from_row is not None:
        return max(0, int(round(from_row))), warnings
    if b_site in NONMETAL_B:
        return 0, warnings
    if block == "d" and group is not None and b_ox is not None:
        return max(0, int(round(group - b_ox))), warnings
    warnings.append("NEED_DATA: B-site d-electron count could not be inferred reliably.")
    return None, warnings


def resolve_structure_path(path_text: Any, roots: list[Path]) -> Path | None:
    text = fmt_need(path_text)
    if text == "NEED_DATA":
        return None
    path = Path(text)
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return None


def coordination_environment(row: pd.Series, b_site: str, roots: list[Path]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    struct_col_value = row.get("structure_path", row.get("struct_path"))
    path = resolve_structure_path(struct_col_value, roots)
    if path is None:
        warnings.append("NEED_DATA: structure path missing or unresolved; assumed ABO3 B-site oxygen coordination.")
        return "BO6 octahedral expected for ABO3 perovskite; exact B-O coordination NEED_DATA.", warnings
    try:
        from pymatgen.core import Structure  # type: ignore

        with py_warnings.catch_warnings():
            py_warnings.simplefilter("ignore")
            structure = Structure.from_file(path)
        counts: list[int] = []
        distances: list[float] = []
        for site in structure:
            site_symbol = getattr(site.specie, "symbol", str(site.specie))
            if str(site_symbol) != b_site:
                continue
            neigh = structure.get_neighbors(site, r=3.0)
            o_neigh = [nn for nn in neigh if str(getattr(nn.specie, "symbol", nn.specie)) == "O"]
            if o_neigh:
                counts.append(len(o_neigh))
                distances.extend(float(nn.nn_distance) for nn in o_neigh)
        if counts:
            count_text = f"B-O coordination count median {int(round(median(counts)))}"
            if distances:
                dist_text = f"mean B-O distance {sum(distances) / len(distances):.3f} A"
                return f"{count_text}; {dist_text}; local environment from structure file.", warnings
            return f"{count_text}; local environment from structure file.", warnings
        warnings.append("NEED_DATA: no B-O neighbors found within 3.0 A in parsed structure.")
    except Exception as exc:
        warnings.append(f"NEED_DATA: failed to parse structure for coordination ({type(exc).__name__}).")
    return "BO6 octahedral expected for ABO3 perovskite; exact B-O coordination NEED_DATA.", warnings


def correlated_orbital(b_site: str, period: int | None, block: str) -> str:
    if block == "d" and period is not None:
        shell = {4: "3d", 5: "4d", 6: "5d"}.get(period, "d")
        return f"B-site {shell} orbitals"
    if block == "f":
        return "B-site f orbitals if partially occupied; NEED_DATA for correlated subspace"
    if b_site == "NEED_DATA":
        return "NEED_DATA"
    return "No obvious partially filled transition-metal d subspace; verify correlated orbital if DFT+U is used"


def classify_correlation(block: str, period: int | None, d_count: int | None, b_site: str) -> str:
    if d_count == 0:
        return "low or weak; d0 cation usually less U-sensitive unless defects or band-edge localization matter"
    if block == "d" and period == 4:
        return "high; 3d oxide cation likely has localized d states"
    if block == "d" and period == 5:
        return "moderate; 4d states are more extended but can remain correlated"
    if block == "d" and period == 6:
        return "low-to-moderate; 5d states are extended, with spin-orbit and covalency possibly important"
    if b_site in NONMETAL_B:
        return "low; no clear transition-metal d shell"
    return "NEED_DATA"


def spin_states(d_count: int | None, block: str) -> str:
    if d_count is None:
        return "NEED_DATA"
    if d_count == 0:
        return "closed-shell d0; spin state usually not central unless defects or A-site magnetism matter"
    if block == "d" and d_count in {4, 5, 6, 7}:
        return "multiple high-spin/intermediate-spin/low-spin states possible depending on ligand field and magnetism"
    if block == "d":
        return "spin polarization may matter; enumerate plausible magnetic states in DFT setup"
    return "NEED_DATA"


def magnetic_relevance(row: pd.Series, d_count: int | None, b_site: str) -> str:
    hint = as_float(row.get("mag_init_hint"))
    if hint is not None and hint > 0:
        return f"magnetic initialization hint present ({hint:g}); magnetic ordering can affect fitted U"
    if d_count is None:
        return "NEED_DATA"
    if d_count == 0:
        return "usually low for B-site d0 cation; check A-site f electrons or defects separately"
    return "magnetic order and moment may affect localization and U; NEED_DATA for measured/calculated moment"


def u_sensitivity(correlation: str, d_count: int | None) -> str:
    if d_count == 0:
        return "low-to-moderate; U may mostly affect empty d states and band alignment"
    if correlation.startswith("high"):
        return "high; U can change localization, magnetic moments, band gap, and relative spin states"
    if correlation.startswith("moderate"):
        return "moderate; U sensitivity depends on bandwidth, covalency, and filling"
    if correlation.startswith("low-to-moderate"):
        return "material-dependent; 5d bandwidth/covalency may reduce U sensitivity"
    return "NEED_DATA"


def band_gap_relevance(d_count: int | None, correlation: str) -> str:
    if d_count == 0:
        return "band-gap and band-edge shifts may be more relevant than occupied d localization; NEED_DATA for measured gap/metallicity"
    if "high" in correlation or "moderate" in correlation:
        return "U can open or change a gap, or alter metallicity; NEED_DATA for experimental/computed gap"
    return "NEED_DATA"


def hybridization_relevance(period: int | None, d_count: int | None) -> str:
    if d_count is None:
        return "NEED_DATA"
    if d_count == 0:
        return "metal-oxygen hybridization affects conduction-band d states and charge-transfer character"
    if period == 4:
        return "B 3d-O 2p hybridization controls localization, charge-transfer character, and effective U"
    if period in {5, 6}:
        return "extended 4d/5d-O 2p hybridization can reduce localization and change effective U"
    return "NEED_DATA"


def find_literature_dirs(root: Path, requested: list[Path] | None, include_docs: bool) -> list[Path]:
    if requested:
        return [p if p.is_absolute() else root / p for p in requested if (p if p.is_absolute() else root / p).exists()]
    out = []
    names = ["papers", "literature", "references"]
    if include_docs:
        names.append("docs")
    for name in names:
        p = root / name
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def discover_curated_csvs(lit_dirs: list[Path], requested: list[Path] | None) -> list[Path]:
    paths: list[Path] = []
    if requested:
        paths.extend(requested)
    for folder in lit_dirs:
        paths.extend(folder.glob("*.csv"))
        paths.extend(folder.glob("*.tsv"))
    seen = set()
    out = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def u_value_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        norm = col.lower().replace(" ", "_")
        is_u_like = "u" in norm and any(token in norm for token in ["u", "ueff", "hubbard"])
        is_lit_like = any(token in norm for token in ["literature", "lit", "reported", "ref", "prior", "hubbard", "ueff"])
        if is_u_like and (is_lit_like or norm not in TARGET_COLUMNS):
            cols.append(col)
    return cols


def load_curated_literature(csv_paths: list[Path]) -> dict[str, list[LiteratureHit]]:
    hits: dict[str, list[LiteratureHit]] = {}
    for path in csv_paths:
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        except Exception:
            continue
        formula_col = first_existing_column(df, ["formula", "Formula", "composition"])
        id_col = first_existing_column(df, ID_CANDIDATES)
        source_col = first_existing_column(df, ["source", "citation", "doi", "reference", "paper", "url"])
        u_cols = u_value_columns(df)
        if not u_cols or (formula_col is None and id_col is None):
            continue
        for _, row in df.iterrows():
            keys = []
            if formula_col is not None and clean_scalar(row.get(formula_col)):
                keys.append(str(row.get(formula_col)).strip())
            if id_col is not None and clean_scalar(row.get(id_col)):
                keys.append(str(row.get(id_col)).strip())
            source = fmt_need(row.get(source_col)) if source_col else path.name
            for u_col in u_cols:
                value = as_float(row.get(u_col))
                if value is None:
                    continue
                for key in keys:
                    hits.setdefault(key, []).append(
                        LiteratureHit(value=value, source=str(source), matched_by=u_col, note=f"curated:{path}")
                    )
    return hits


def scan_text_literature(lit_dirs: list[Path], formulas: set[str], b_sites: set[str], max_file_mb: float) -> dict[str, list[LiteratureHit]]:
    hits: dict[str, list[LiteratureHit]] = {}
    text_suffixes = {".txt", ".md", ".rst"}
    u_pattern = re.compile(
        r"\b(?:Ueff|U_eff|Hubbard\s+U|U)\b[^.\n;:]{0,60}?"
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:eV|ev)\b",
        re.IGNORECASE,
    )
    for folder in lit_dirs:
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            if path.stat().st_size > max_file_mb * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            values = [float(m.group(1)) for m in u_pattern.finditer(text)]
            if not values:
                continue
            matched_keys = {formula for formula in formulas if formula and formula in text}
            for b in b_sites:
                if b and re.search(rf"\b{re.escape(b)}\b", text):
                    matched_keys.add(f"B:{b}")
            for key in matched_keys:
                for value in values:
                    hits.setdefault(key, []).append(
                        LiteratureHit(value=value, source=str(path), matched_by="text_regex", note="literature_prior")
                    )
    return hits


def collect_literature_hits(
    material_id: str,
    formula: str,
    b_site: str,
    curated: dict[str, list[LiteratureHit]],
    scanned: dict[str, list[LiteratureHit]],
) -> list[LiteratureHit]:
    hits: list[LiteratureHit] = []
    for key in [material_id, formula]:
        hits.extend(curated.get(key, []))
        hits.extend(scanned.get(key, []))
    if not hits and b_site != "NEED_DATA":
        hits.extend(curated.get(f"B:{b_site}", []))
        hits.extend(scanned.get(f"B:{b_site}", []))
    dedup: dict[tuple[float, str], LiteratureHit] = {}
    for hit in hits:
        dedup[(round(hit.value, 6), hit.source)] = hit
    return list(dedup.values())


def summarize_literature(hits: list[LiteratureHit]) -> dict[str, Any]:
    values = [hit.value for hit in hits if math.isfinite(hit.value)]
    sources = sorted({hit.source for hit in hits})
    if not values:
        return {
            "literature_U_values": [],
            "literature_U_min": "NEED_DATA",
            "literature_U_max": "NEED_DATA",
            "literature_U_mean": "NEED_DATA",
            "literature_U_median": "NEED_DATA",
            "literature_U_count": 0,
            "literature_source_count": 0,
            "literature_sources": [],
        }
    return {
        "literature_U_values": [{"value_eV": hit.value, "source": hit.source, "matched_by": hit.matched_by} for hit in hits],
        "literature_U_min": min(values),
        "literature_U_max": max(values),
        "literature_U_mean": sum(values) / len(values),
        "literature_U_median": median(values),
        "literature_U_count": len(values),
        "literature_source_count": len(sources),
        "literature_sources": sources,
    }


def confidence_level(warnings: list[str], literature_count: int, ox_known: bool) -> str:
    if not ox_known:
        return "low"
    if literature_count > 0 and len(warnings) <= 1:
        return "high"
    if len(warnings) <= 2:
        return "medium"
    return "low"


def build_summary(card: dict[str, Any], has_lit: bool) -> str:
    parts = [
        f"{card['formula']} has B-site {card['B_site']} with likely oxidation {card['likely_B_oxidation_state']} "
        f"and nominal d-electron count {card['B_site_d_electron_count']}.",
        f"The relevant correlated subspace is {card['correlated_orbital']}; expected correlation strength is {card['expected_correlation_strength']}.",
        f"Local coordination: {card['B_site_coordination_environment']}.",
        f"Spin/magnetism: {card['possible_spin_states']}; {card['magnetic_relevance']}.",
        f"U sensitivity: {card['expected_U_sensitivity']}. Band-gap relevance: {card['band_gap_relevance']}. "
        f"Metal-oxygen hybridization: {card['metal_oxygen_hybridization_relevance']}.",
    ]
    if has_lit:
        parts.append(
            f"Literature prior U values are available: count {card['literature_U_count']}, "
            f"range {card['literature_U_min']} to {card['literature_U_max']} eV."
        )
    else:
        parts.append(
            "No exact-material literature U prior was found; use B-site cation, oxidation state, and oxide-family chemistry as the prior."
        )
    return " ".join(parts)


def make_card(row: pd.Series, id_col: str, formula_col: str, roots: list[Path], literature_hits: list[LiteratureHit]) -> dict[str, Any]:
    warnings: list[str] = []
    material_id = fmt_need(row.get(id_col))
    formula = fmt_need(row.get(formula_col))
    a_site, b_site, site_warnings = infer_sites(row, formula)
    warnings.extend(site_warnings)

    period, group, block = element_info(b_site)
    row_period = as_float(row.get("B_period"))
    row_group = as_float(row.get("B_group"))
    row_block = clean_scalar(row.get("B_block"))
    if row_period is not None:
        period = int(row_period)
    if row_group is not None:
        group = int(row_group)
    if row_block:
        block = str(row_block)

    a_ox, b_ox, ox_warnings = likely_oxidation_states(row, a_site, b_site)
    warnings.extend(ox_warnings)
    d_count, d_warnings = d_electron_count(row, b_site, b_ox, group, block)
    warnings.extend(d_warnings)
    coord, coord_warnings = coordination_environment(row, b_site, roots)
    warnings.extend(coord_warnings)

    lit = summarize_literature(literature_hits)
    correlation = classify_correlation(block, period, d_count, b_site)
    card: dict[str, Any] = {
        "material_id": material_id,
        "formula": formula,
        "A_site": a_site,
        "B_site": b_site,
        "likely_A_oxidation_state": a_ox if a_ox is not None else "NEED_DATA",
        "likely_B_oxidation_state": b_ox if b_ox is not None else "NEED_DATA",
        "B_site_d_electron_count": d_count if d_count is not None else "NEED_DATA",
        "B_site_period": period if period is not None else "NEED_DATA",
        "B_site_group": group if group is not None else "NEED_DATA",
        "B_site_coordination_environment": coord,
        "correlated_orbital": correlated_orbital(b_site, period, block),
        "expected_correlation_strength": correlation,
        "possible_spin_states": spin_states(d_count, block),
        "magnetic_relevance": magnetic_relevance(row, d_count, b_site),
        "expected_U_sensitivity": u_sensitivity(correlation, d_count),
        "band_gap_relevance": band_gap_relevance(d_count, correlation),
        "metal_oxygen_hybridization_relevance": hybridization_relevance(period, d_count),
        "source_confidence": "NEED_DATA",
        "warnings": [],
    }
    card.update(lit)
    card["source_confidence"] = confidence_level(warnings, int(card["literature_U_count"]), b_ox is not None)
    card["warnings"] = sorted(set(warnings)) or []
    card["text_summary_for_embedding"] = build_summary(card, bool(card["literature_U_count"]))
    return {field: card.get(field, "NEED_DATA") for field in CARD_FIELDS}


def write_jsonl(cards: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for card in cards:
            fh.write(json.dumps(card, ensure_ascii=False) + "\n")


def write_csv(cards: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CARD_FIELDS)
        writer.writeheader()
        for card in cards:
            row = card.copy()
            for key in ["literature_U_values", "warnings"]:
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)


def write_schema(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# U Knowledge Card Schema",
        "",
        "Generated by `scripts/build_u_knowledge_cards.py`.",
        "",
        "The card is a structured, U-focused text modality for ABO3 DFT+U active learning. It is intended to summarize chemistry and literature priors without using the target DFT+U label directly in `text_summary_for_embedding`.",
        "",
        "## Outputs",
        "",
        "- `processed/u_knowledge_cards.jsonl`: one JSON object per material.",
        "- `processed/u_knowledge_cards.csv`: flat CSV version; list-valued fields are JSON strings.",
        "",
        "## Field Definitions",
        "",
        "| Field | Type | Meaning |",
        "|---|---|---|",
        "| `material_id` | string | Material identifier from `material_id`, `id`, `mp_id`, or `MP-ID`; `NEED_DATA` if missing. |",
        "| `formula` | string | Formula from the material table. |",
        "| `A_site`, `B_site` | string | Inferred ABO3 A/B cations, preferring explicit `A_elem` and `B_elem` columns. |",
        "| `likely_A_oxidation_state`, `likely_B_oxidation_state` | number or `NEED_DATA` | Charge-balance oxidation estimates using explicit `oxid_B` when available, otherwise common A-site states. Ambiguity lowers confidence. |",
        "| `B_site_d_electron_count` | number or `NEED_DATA` | Nominal B-site d count from table or group minus oxidation for d-block cations. |",
        "| `B_site_period`, `B_site_group` | number or `NEED_DATA` | Period/group from table or periodic-table lookup. |",
        "| `B_site_coordination_environment` | string | Structure-derived B-O coordination when a structure is readable; otherwise an ABO3 octahedral expectation marked with `NEED_DATA`. |",
        "| `correlated_orbital` | string | Likely DFT+U correlated subspace, usually B-site 3d/4d/5d. |",
        "| `expected_correlation_strength` | string | Qualitative prior based on B-site block, period, and d count. |",
        "| `possible_spin_states` | string | U-relevant spin-state possibilities; ambiguous spin states are stated explicitly. |",
        "| `magnetic_relevance` | string | Whether magnetic ordering/moments may affect U. |",
        "| `expected_U_sensitivity` | string | Qualitative effect of U on localization, gap, spin, and magnetic moment. |",
        "| `band_gap_relevance` | string | Whether U is likely to affect band gap or metallicity; missing measured gap remains `NEED_DATA`. |",
        "| `metal_oxygen_hybridization_relevance` | string | Role of B d-O 2p covalency in effective U. |",
        "| `literature_U_values` | list | Literature-prior U values as objects with `value_eV`, `source`, and `matched_by`. These are never taken from the target label column. |",
        "| `literature_U_min`, `literature_U_max`, `literature_U_mean`, `literature_U_median` | number or `NEED_DATA` | Summary statistics over literature-prior U values. |",
        "| `literature_U_count` | integer | Number of extracted literature-prior U values. |",
        "| `literature_source_count` | integer | Number of distinct sources contributing literature-prior U values. |",
        "| `source_confidence` | `high`, `medium`, or `low` | Confidence based on oxidation-state certainty, structure parsing, and literature support. |",
        "| `text_summary_for_embedding` | string | Compact U-focused paragraph. It excludes target labels and generic material description. |",
        "| `warnings` | list | Missing data, ambiguity, parsing failures, and provenance warnings. |",
        "",
        "## Literature Rules",
        "",
        "- Target-label columns such as `U_eV` or `Ueff_eV` in the input material table are not used as literature priors.",
        "- Curated CSV/TSV files can provide literature priors when they contain formula or material ID columns and U-like reported/literature/reference columns.",
        "- Text files in `papers/`, `literature/`, `references/`, or explicitly requested folders are scanned only for explicit U/Ueff/Hubbard U values with eV units.",
        "- `docs/` is not auto-scanned by default because project documentation can contain derived labels, metrics, or audit text rather than independent literature priors. Use `--include-docs-literature` or `--literature-dir docs` only when that folder contains literature notes intended as sources.",
        "- If no exact material literature U is available, the summary falls back to B-site cation, oxidation state, and oxide-family chemistry.",
        "- Missing values are written as `NEED_DATA`; literature U values are never fabricated.",
        "",
        "## Reproducibility",
        "",
        "Default command:",
        "",
        "```powershell",
        "& 'python' .\\scripts\\build_u_knowledge_cards.py `",
        "  --materials .\\processed\\multimodal_dataset.csv",
        "```",
        "",
        "Outputs are written to `processed/` and `docs/`. Raw/source data are read only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--materials", type=Path, default=root / "processed/multimodal_dataset.csv")
    ap.add_argument("--out-jsonl", type=Path, default=root / "processed/u_knowledge_cards.jsonl")
    ap.add_argument("--out-csv", type=Path, default=root / "processed/u_knowledge_cards.csv")
    ap.add_argument("--schema-doc", type=Path, default=root / "docs/u_knowledge_card_schema.md")
    ap.add_argument("--literature-dir", type=Path, action="append", help="Optional local literature folder; can be repeated.")
    ap.add_argument("--curated-csv", type=Path, action="append", help="Optional curated CSV/TSV with literature U priors; can be repeated.")
    ap.add_argument(
        "--include-docs-literature",
        action="store_true",
        help="Also auto-scan docs/. Use only if docs/ contains literature notes rather than project outputs.",
    )
    ap.add_argument("--max-text-file-mb", type=float, default=5.0)
    args = ap.parse_args()

    df = read_table(args.materials)
    id_col = first_existing_column(df, ID_CANDIDATES)
    formula_col = first_existing_column(df, FORMULA_CANDIDATES)
    if formula_col is None:
        raise ValueError("Material table must contain a formula column.")
    if id_col is None:
        df = df.copy()
        df["material_id"] = [f"row_{i:05d}" for i in range(len(df))]
        id_col = "material_id"

    struct_col = first_existing_column(df, STRUCTURE_CANDIDATES)
    if struct_col is not None and struct_col != "structure_path":
        df = df.copy()
        df["structure_path"] = df[struct_col]

    roots = [root, root / "SCR", root / "SCR/project", args.materials.parent]
    literature_dirs = find_literature_dirs(root, args.literature_dir, args.include_docs_literature)
    curated_csvs = discover_curated_csvs(literature_dirs, args.curated_csv)
    curated_hits = load_curated_literature(curated_csvs)

    formulas = {str(v).strip() for v in df[formula_col].dropna().tolist()}
    b_col = first_existing_column(df, ["B_elem", "B_site"])
    b_sites = {str(v).strip() for v in df[b_col].dropna().tolist()} if b_col else set()
    scanned_hits = scan_text_literature(literature_dirs, formulas, b_sites, args.max_text_file_mb)

    cards: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        material_id = fmt_need(row.get(id_col))
        formula = fmt_need(row.get(formula_col))
        a_site, b_site, _ = infer_sites(row, formula)
        lit_hits = collect_literature_hits(material_id, formula, b_site, curated_hits, scanned_hits)
        cards.append(make_card(row, id_col, formula_col, roots, lit_hits))

    write_jsonl(cards, args.out_jsonl)
    write_csv(cards, args.out_csv)
    write_schema(args.schema_doc)

    n_lit = sum(1 for card in cards if card["literature_U_count"])
    print(
        json.dumps(
            {
                "rows": len(cards),
                "cards_with_literature_u": n_lit,
                "literature_dirs": [str(p) for p in literature_dirs],
                "curated_csvs": [str(p) for p in curated_csvs],
                "outputs": {
                    "jsonl": str(args.out_jsonl),
                    "csv": str(args.out_csv),
                    "schema": str(args.schema_doc),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
