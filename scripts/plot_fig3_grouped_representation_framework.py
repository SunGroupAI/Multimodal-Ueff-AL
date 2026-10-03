"""Main figure: panel-specific descriptive maximum-degradation contrasts.
Original data and algorithms retained; standalone alignment helper supplied.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figure_reproduction"
AUDIT = ROOT / "analysis_multimodal_AL"
AL_DIR = ROOT / "results" / "active_learning_ablation"
from panel_alignment import require_matplotlib_panel_alignment


plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
if "sans-serif" not in plt.rcParams["font.family"]:
    raise RuntimeError("The global Matplotlib style must use a publication-safe sans-serif font")

CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
GROUPS = ["Single", "Bimodal", "Multimodal"]
# Keep the reduced representations visually quiet and reserve the clearest
# blue accent for the complete multimodal representation.
COLORS = {"Single": CYCLE[6], "Bimodal": CYCLE[1], "Multimodal": CYCLE[0]}
STATIC_MEMBERS = {
    "Single": ["Element", "Structure", "Chemical prior"],
    "Bimodal": ["Element + Structure", "Element + Chemical prior", "Structure + Chemical prior"],
    "Multimodal": ["Multimodal"],
}
AL_CLASS = {
    "comp": "Single",
    "struct": "Single",
    "electronic": "Single",
    "comp+struct": "Bimodal",
    "comp+electronic": "Bimodal",
    "struct+electronic": "Bimodal",
    "comp+struct+electronic": "Multimodal",
}
SELECTED_AL_BLOCKS = {
    "struct": "Single",
    "comp+struct": "Bimodal",
    "comp+struct+electronic": "Multimodal",
}

ABLATION_DISPLAY = {
    "Element + Structure": "Remove chemical prior\n(Element + Structure)",
    "Element + Chemical prior": "Remove structure\n(Element + Chemical prior)",
    "Structure + Chemical prior": "Remove element\n(Structure + Chemical prior)",
    "Element": "Keep element only\n(remove structure + chemical prior)",
    "Structure": "Keep structure only\n(remove element + chemical prior)",
    "Chemical prior": "Keep chemical prior only\n(remove element + structure)",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    offset = ScaledTranslation(-8 / 72, 5 / 72, ax.figure.dpi_scale_trans)
    ax.text(0, 1, label, transform=ax.transAxes + offset, ha="right", va="bottom", fontweight="bold", clip_on=False)


def inward_ticks(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="both", direction="in")


def fmt_sig(value: float, digits: int = 2) -> str:
    """Format a finite value with a fixed number of significant digits."""
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"Cannot format non-finite value: {value}")
    if value == 0:
        return "0.0" if digits == 2 else "0"
    decimals = digits - 1 - int(np.floor(np.log10(abs(value))))
    if decimals > 0:
        return f"{value:.{decimals}f}"
    return f"{value:.0f}"


def build_static_penalty_data(
    static_modality_seed: pd.DataFrame,
    static_rep_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the largest mean penalty in each reduced-representation class."""
    pivot = static_modality_seed.pivot(index="seed", columns="modality", values="MAE")
    selected = []
    for category in ["Bimodal", "Single"]:
        rows = static_rep_summary[static_rep_summary["category"].eq(category)].sort_values("mean", ascending=False)
        if rows.empty:
            raise ValueError(f"No {category} representations were available for panel a")
        selected.append((category, str(rows.iloc[0]["modality"])))

    seed_rows = []
    summary_rows = []
    for category, modality in selected:
        penalty = pivot[modality] - pivot["Multimodal"]
        removal_class = "One modality removed" if category == "Bimodal" else "Two modalities removed"
        for seed, value in penalty.items():
            seed_rows.append(
                {
                    "category": category,
                    "removal_class": removal_class,
                    "modality": modality,
                    "display_label": ABLATION_DISPLAY[modality],
                    "seed": int(seed),
                    "MAE_penalty_eV": float(value),
                    "reduced_MAE_eV": float(pivot.loc[seed, modality]),
                    "multimodal_MAE_eV": float(pivot.loc[seed, "Multimodal"]),
                }
            )
        summary_rows.append(
            {
                "category": category,
                "removal_class": removal_class,
                "modality": modality,
                "display_label": ABLATION_DISPLAY[modality],
                "mean": float(penalty.mean()),
                "std": float(penalty.std(ddof=1)),
                "positive_seeds": int((penalty > 0).sum()),
                "n_seeds": int(penalty.shape[0]),
                "selection_rule": f"Largest aggregate mean MAE among all tested {category.lower()} representations",
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def load_grouped_data() -> tuple:
    static_modality_seed = pd.read_csv(AUDIT / "modality_by_seed.csv")
    static_rep_summary = static_modality_seed.groupby(["category", "modality"], as_index=False)["MAE"].agg(["mean", "std"]).reset_index()
    pivot = static_modality_seed.pivot(index="seed", columns="modality", values="MAE")
    static_rows = []
    for group, members in STATIC_MEMBERS.items():
        values = pivot[members].mean(axis=1)
        for seed, value in values.items():
            static_rows.append({"group": group, "seed": int(seed), "MAE": float(value), "members": "; ".join(members)})
    static_group_seed = pd.DataFrame(static_rows)
    static_summary = static_group_seed.groupby("group", as_index=False)["MAE"].agg(["mean", "std"]).reset_index()
    static_penalty_seed, static_penalty_summary = build_static_penalty_data(static_modality_seed, static_rep_summary)

    raw = pd.read_csv(AL_DIR / "results.csv")
    meta = json.loads((AL_DIR / "run_meta.json").read_text(encoding="utf-8"))
    hybrid = raw[raw["strategy"].eq("hybrid") & raw["modality_block"].isin(AL_CLASS)].copy()
    hybrid["group"] = hybrid["modality_block"].map(AL_CLASS)
    group_seed_round = hybrid.groupby(["seed", "round", "train_size", "group"], as_index=False)[["MAE", "uncertainty_error_spearman"]].mean()
    expected = len(meta["seeds"]) * (meta["rounds"] + 1) * len(GROUPS)
    if len(group_seed_round) != expected:
        raise ValueError(f"Expected {expected} grouped AL rows, found {len(group_seed_round)}")
    final_round = int(hybrid["round"].max())
    final_all = hybrid[hybrid["round"].eq(final_round)].copy()
    final_block = final_all.groupby(["group", "modality_block"], as_index=False)[["MAE", "uncertainty_error_spearman"]].mean()

    def select_block(group: str, metric: str, ascending: bool) -> str:
        rows = final_block[final_block["group"].eq(group)].sort_values(metric, ascending=ascending)
        return str(rows.iloc[0]["modality_block"])

    b_blocks = {
        "Single": select_block("Single", "uncertainty_error_spearman", True),
        "Bimodal": select_block("Bimodal", "uncertainty_error_spearman", True),
        "Multimodal": "comp+struct+electronic",
    }
    d_blocks = {
        "Single": select_block("Single", "MAE", False),
        "Bimodal": select_block("Bimodal", "MAE", False),
        "Multimodal": "comp+struct+electronic",
    }

    all_aulc_rows = []
    for (group, block, seed), part in hybrid.groupby(["group", "modality_block", "seed"]):
        part = part.sort_values("train_size")
        x = part["train_size"].to_numpy(float)
        y = part["MAE"].to_numpy(float)
        all_aulc_rows.append({"group": group, "modality_block": block, "seed": int(seed), "integrated_mae": float(np.trapezoid(y, x) / (x[-1] - x[0]))})
    all_aulc = pd.DataFrame(all_aulc_rows)
    aulc_block = all_aulc.groupby(["group", "modality_block"], as_index=False)["integrated_mae"].mean()
    c_blocks = {
        "Single": str(aulc_block[aulc_block["group"].eq("Single")].sort_values("integrated_mae", ascending=False).iloc[0]["modality_block"]),
        "Bimodal": str(aulc_block[aulc_block["group"].eq("Bimodal")].sort_values("integrated_mae", ascending=False).iloc[0]["modality_block"]),
        "Multimodal": "comp+struct+electronic",
    }

    def select_panel_rows(blocks: dict[str, str]) -> pd.DataFrame:
        parts = []
        for group, block in blocks.items():
            part = hybrid[hybrid["modality_block"].eq(block)].copy()
            part["group"] = group
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    b_seed_round = select_panel_rows(b_blocks)
    c_seed_round = select_panel_rows(c_blocks)
    d_seed_round = select_panel_rows(d_blocks)
    trajectory = c_seed_round.groupby(["group", "train_size"], as_index=False)["MAE"].agg(["mean", "std"]).reset_index()
    uncertainty_seed = b_seed_round.loc[
        b_seed_round["round"].eq(final_round),
        ["group", "seed", "train_size", "uncertainty_error_spearman"],
    ].copy()
    uncertainty_summary = uncertainty_seed.groupby("group", as_index=False)["uncertainty_error_spearman"].agg(["mean", "std"]).reset_index()
    final_mae_seed = d_seed_round.loc[
        d_seed_round["round"].eq(final_round),
        ["group", "seed", "train_size", "MAE"],
    ].copy()
    final_mae_summary = final_mae_seed.groupby("group", as_index=False)["MAE"].agg(["mean", "std"]).reset_index()

    aulc_rows = []
    for (group, seed), part in c_seed_round.groupby(["group", "seed"]):
        part = part.sort_values("train_size")
        x = part["train_size"].to_numpy(float)
        y = part["MAE"].to_numpy(float)
        aulc_rows.append(
            {
                "group": group,
                "seed": int(seed),
                "integrated_mae": float(np.trapezoid(y, x) / (x[-1] - x[0])),
                "initial_mae": float(y[0]),
                "final_mae": float(y[-1]),
            }
        )
    aulc_seed = pd.DataFrame(aulc_rows)
    aulc_summary = aulc_seed.groupby("group", as_index=False)["integrated_mae"].agg(["mean", "std"]).reset_index()
    panel_selection = {"b": b_blocks, "c": c_blocks, "d": d_blocks}
    return static_modality_seed, static_group_seed, static_summary, static_rep_summary, static_penalty_seed, static_penalty_summary, c_seed_round, trajectory, uncertainty_seed, uncertainty_summary, final_mae_seed, final_mae_summary, aulc_seed, aulc_summary, meta, panel_selection


def panel_a(ax: plt.Axes, penalty_seed: pd.DataFrame, penalty_summary: pd.DataFrame) -> None:
    order = ["Single", "Bimodal", "Multimodal"]
    values_by_group = {
        "Single": penalty_seed.loc[penalty_seed["category"].eq("Single"), "reduced_MAE_eV"].to_numpy(float),
        "Bimodal": penalty_seed.loc[penalty_seed["category"].eq("Bimodal"), "reduced_MAE_eV"].to_numpy(float),
        "Multimodal": penalty_seed[["seed", "multimodal_MAE_eV"]].drop_duplicates("seed")["multimodal_MAE_eV"].to_numpy(float),
    }
    means = {}
    for yi, group in enumerate(order):
        values = values_by_group[group]
        mean, std = float(values.mean()), float(values.std(ddof=1))
        means[group] = mean
        offsets = np.linspace(-0.055, 0.055, len(values))
        is_multi = group == "Multimodal"
        marker = {"Single": "o", "Bimodal": "s", "Multimodal": "D"}[group]
        ax.scatter(values, np.full(len(values), yi) + offsets, s=17 if not is_multi else 23, color=COLORS[group], alpha=0.30 if not is_multi else 0.45, linewidth=0, zorder=2)
        ax.errorbar(mean, yi, xerr=std, fmt="none", ecolor=COLORS[group], elinewidth=0.85 if not is_multi else 1.35, capsize=3, capthick=0.8 if not is_multi else 1.2, zorder=3)
        ax.scatter(mean, yi, s=50 if not is_multi else 84, marker=marker, color=COLORS[group], edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(mean + 0.010, yi - 0.16, fmt_sig(mean), color=COLORS[group], va="center", fontweight="bold" if is_multi else "normal")
    single_reduction = 100 * (means["Single"] - means["Multimodal"]) / means["Single"]
    bimodal_reduction = 100 * (means["Bimodal"] - means["Multimodal"]) / means["Bimodal"]
    ax.text(0.02, 0.96, f"Multimodal: −{fmt_sig(single_reduction)}% vs Single\n−{fmt_sig(bimodal_reduction)}% vs Bimodal", transform=ax.transAxes, ha="left", va="top", color=COLORS["Multimodal"], fontweight="bold", fontsize=7.2)
    ax.set_yticks(np.arange(3), order)
    ax.invert_yaxis()
    ax.set_ylim(2.45, -0.85)
    for tick, group in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[group])
    ax.set_xlabel("Test MAE (eV)")
    ax.set_xlim(0.50, 0.90)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    inward_ticks(ax)
    add_panel_label(ax, "a")


def panel_b(ax: plt.Axes, uncertainty_seed: pd.DataFrame, uncertainty_summary: pd.DataFrame) -> None:
    order = ["Single", "Bimodal", "Multimodal"]
    summary = uncertainty_summary.set_index("group")
    for yi, group in enumerate(order):
        values = uncertainty_seed.loc[uncertainty_seed["group"].eq(group), "uncertainty_error_spearman"].to_numpy(float)
        mean, std = float(summary.loc[group, "mean"]), float(summary.loc[group, "std"])
        offsets = np.linspace(-0.055, 0.055, len(values))
        is_multi = group == "Multimodal"
        ax.scatter(values, np.full(len(values), yi) + offsets, s=14 if not is_multi else 21, color=COLORS[group], alpha=0.30 if not is_multi else 0.45, linewidth=0, zorder=2)
        ax.errorbar(mean, yi, xerr=std, fmt="none", ecolor=COLORS[group], elinewidth=0.85 if not is_multi else 1.35, capsize=3, capthick=0.8 if not is_multi else 1.2, zorder=3)
        ax.scatter(mean, yi, s=50 if not is_multi else 82, marker="D", color=COLORS[group], edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(mean + 0.012, yi - 0.23, fmt_sig(mean), color=COLORS[group], va="center", fontweight="bold" if is_multi else "normal")
    multi = float(summary.loc["Multimodal", "mean"])
    single_gain = 100 * (multi - float(summary.loc["Single", "mean"])) / float(summary.loc["Single", "mean"])
    bimodal_gain = 100 * (multi - float(summary.loc["Bimodal", "mean"])) / float(summary.loc["Bimodal", "mean"])
    ax.text(0.98, 0.04, f"Multimodal: +{fmt_sig(single_gain)}% vs Single\n+{fmt_sig(bimodal_gain)}% vs Bimodal", transform=ax.transAxes, ha="right", va="bottom", color=COLORS["Multimodal"], fontweight="bold", fontsize=7.2)
    ax.set_yticks(np.arange(3), order)
    ax.invert_yaxis()
    ax.set_ylim(2.90, -0.45)
    for tick, group in zip(ax.get_yticklabels(), order):
        tick.set_color(COLORS[group])
    ax.set_xlabel(r"Error–uncertainty Spearman $\rho$")
    ax.set_xlim(-0.02, 0.60)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    inward_ticks(ax)
    add_panel_label(ax, "b")


def panel_c(ax: plt.Axes, aulc_seed: pd.DataFrame) -> None:
    """Show paired label-efficiency gains without overlapping MAE trajectories."""
    pivot = aulc_seed.pivot(index="seed", columns="group", values="integrated_mae").loc[:, GROUPS]
    comparisons = [("Single", "vs Single"), ("Bimodal", "vs Bimodal")]
    for yi, (baseline, label) in enumerate(comparisons):
        reduction = 100 * (pivot[baseline] - pivot["Multimodal"]) / pivot[baseline]
        mean = float(reduction.mean())
        std = float(reduction.std(ddof=1))
        offsets = np.linspace(-0.055, 0.055, len(reduction))
        ax.hlines(yi, 0, mean, color=COLORS[baseline], alpha=0.28, linewidth=5.2, zorder=1)
        ax.scatter(reduction, np.full(len(reduction), yi) + offsets, s=20, color=COLORS[baseline], alpha=0.62, edgecolor="white", linewidth=0.35, zorder=2)
        ax.errorbar(mean, yi, xerr=std, fmt="none", ecolor=COLORS["Multimodal"], elinewidth=1.25, capsize=3, capthick=1.0, zorder=3)
        ax.scatter(mean, yi, s=78, marker="D", color=COLORS["Multimodal"], edgecolor="white", linewidth=0.65, zorder=4)
        label_x = max(float(reduction.max()), mean + std) + 2.0
        ax.text(label_x, yi, f"{fmt_sig(mean)}%", color=COLORS["Multimodal"], va="center", fontweight="bold")
    ax.axvline(0, color="0.55", linewidth=0.8, linestyle=(0, (2, 2)), zorder=0)
    ax.set_yticks(np.arange(len(comparisons)), [label for _, label in comparisons])
    ax.invert_yaxis()
    ax.set_ylim(1.55, -0.55)
    ax.set_xlim(-10, 58)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel("Multimodal AULC reduction (%)")
    ax.text(0.98, 0.94, "Paired across seeds (5/5 improved)", transform=ax.transAxes, ha="right", va="top", color=COLORS["Multimodal"], fontweight="bold", fontsize=7.2)
    inward_ticks(ax)
    add_panel_label(ax, "c")


def panel_d(ax: plt.Axes, final_mae_seed: pd.DataFrame) -> None:
    pivot = final_mae_seed.pivot(index="seed", columns="group", values="MAE").loc[:, GROUPS]
    x = np.arange(len(GROUPS))
    box = ax.boxplot([pivot[group].to_numpy(float) for group in GROUPS], positions=x, widths=0.42, patch_artist=True, showfliers=False, medianprops={"color": "white", "linewidth": 1.0}, whiskerprops={"color": "0.35", "linewidth": 0.8}, capprops={"color": "0.35", "linewidth": 0.8})
    for patch, group in zip(box["boxes"], GROUPS):
        patch.set_facecolor(COLORS[group])
        patch.set_alpha(0.58 if group != "Multimodal" else 0.82)
        patch.set_edgecolor("none")
    offsets = np.linspace(-0.08, 0.08, len(pivot))
    for xi, group in enumerate(GROUPS):
        ax.scatter(np.full(len(pivot), xi) + offsets, pivot[group], s=18, color=COLORS[group], alpha=0.72, edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(xi, float(pivot[group].mean()), s=48 if group != "Multimodal" else 78, marker="D", color=COLORS[group], edgecolor="white", linewidth=0.65, zorder=4)
    single_reduction = 100 * (pivot["Single"].mean() - pivot["Multimodal"].mean()) / pivot["Single"].mean()
    bimodal_reduction = 100 * (pivot["Bimodal"].mean() - pivot["Multimodal"].mean()) / pivot["Bimodal"].mean()
    ax.set_xticks(x, GROUPS)
    for tick, group in zip(ax.get_xticklabels(), GROUPS):
        tick.set_color(COLORS[group])
    ax.set_ylabel("Final-round test MAE (eV)")
    ax.set_ylim(0.55, 1.50)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.text(0.98, 0.92, f"Multimodal: −{fmt_sig(single_reduction)}% vs Single\n−{fmt_sig(bimodal_reduction)}% vs Bimodal", transform=ax.transAxes, ha="right", va="top", color=COLORS["Multimodal"], fontweight="bold", fontsize=7.2)
    inward_ticks(ax)
    add_panel_label(ax, "d")


def save_figure_bundle(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))


def build_figure(static_penalty_seed: pd.DataFrame, static_penalty_summary: pd.DataFrame, al_group_seed_round: pd.DataFrame, uncertainty_seed: pd.DataFrame, uncertainty_summary: pd.DataFrame, trajectory: pd.DataFrame, aulc_seed: pd.DataFrame, final_mae_seed: pd.DataFrame) -> None:
    fig, grid = plt.subplots(2, 2, figsize=(7.2, 5.8), layout=None)
    axes = list(grid.ravel())
    panel_a(axes[0], static_penalty_seed, static_penalty_summary)
    panel_b(axes[1], uncertainty_seed, uncertainty_summary)
    panel_c(axes[2], aulc_seed)
    panel_d(axes[3], final_mae_seed)
    positions = [(0.14, 0.59, 0.36, 0.35), (0.60, 0.59, 0.36, 0.35), (0.14, 0.10, 0.36, 0.35), (0.60, 0.10, 0.36, 0.35)]
    fig.canvas.draw()
    fig.set_layout_engine(None)
    for ax, pos in zip(axes, positions):
        ax.set_position(pos)
    base = OUT / "fig3_grouped_representation_framework"
    require_matplotlib_panel_alignment(
        fig,
        axes=axes,
        panel_ids=list("abcd"),
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    save_figure_bundle(fig, base)
    plt.close(fig)


def build_si_ablation_figure(static_modality_seed: pd.DataFrame, static_rep_summary: pd.DataFrame) -> None:
    """Export the complete seven-representation ablation as SI evidence."""
    data = static_rep_summary.sort_values("mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.1), layout=None)
    y = np.arange(len(data))
    colors = [COLORS[row["category"]] for _, row in data.iterrows()]
    sizes = [66 if row["category"] == "Multimodal" else 42 for _, row in data.iterrows()]
    for yi, row in data.iterrows():
        values = static_modality_seed.loc[static_modality_seed["modality"].eq(row["modality"]), "MAE"].to_numpy(float)
        offsets = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(values, np.full(len(values), yi) + offsets, s=17, color=COLORS[row["category"]], alpha=0.35, edgecolor="white", linewidth=0.3, zorder=2)
    ax.errorbar(data["mean"], y, xerr=data["std"], fmt="none", ecolor="0.42", elinewidth=0.9, capsize=2.8, capthick=0.8, zorder=1)
    ax.scatter(data["mean"], y, s=sizes, marker="D", c=colors, edgecolor="white", linewidth=0.65, zorder=3)
    ax.set_yticks(y, data["modality"])
    ax.invert_yaxis()
    ax.set_ylim(len(data) - 0.5, -0.65)
    for tick, color in zip(ax.get_yticklabels(), colors):
        tick.set_color(color)
    ax.set_xlabel("Test MAE (eV)")
    ax.set_xlim(0.50, 0.90)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    inward_ticks(ax)
    base = OUT / "figS_complete_modality_ablation"
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.16, top=0.96)
    require_matplotlib_panel_alignment(
        fig,
        axes=[ax],
        panel_ids=["a"],
        json_out=str(base) + ".alignment.json",
        overlay_svg=str(base) + ".alignment.svg",
        require_panel_labels=False,
        strict=True,
    )
    save_figure_bundle(fig, base)
    plt.close(fig)


def write_outputs(
    static_modality_seed: pd.DataFrame,
    static_seed: pd.DataFrame,
    static_summary: pd.DataFrame,
    static_rep_summary: pd.DataFrame,
    static_penalty_seed: pd.DataFrame,
    static_penalty_summary: pd.DataFrame,
    al_group_seed_round: pd.DataFrame,
    trajectory: pd.DataFrame,
    uncertainty_seed: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    final_mae_seed: pd.DataFrame,
    final_mae_summary: pd.DataFrame,
    aulc_seed: pd.DataFrame,
    aulc_summary: pd.DataFrame,
    meta: dict,
    panel_selection: dict,
) -> None:
    rows = []
    for _, row in static_seed.iterrows():
        rows.append({"panel": "a-background", "record_type": "group_seed", "group": row["group"], "seed": row["seed"], "label_count": np.nan, "metric": "MAE", "value": row["MAE"], "spread": np.nan, "definition": f"Mean of members: {row['members']}"})
    for _, row in static_summary.iterrows():
        rows.append({"panel": "a-background", "record_type": "group_summary", "group": row["group"], "seed": np.nan, "label_count": np.nan, "metric": "MAE", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD of within-seed group averages"})
    for _, row in static_rep_summary.iterrows():
        rows.append({"panel": "SI", "record_type": "representation_summary", "group": row["category"], "representation": row["modality"], "seed": np.nan, "label_count": np.nan, "metric": "MAE", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD for the complete seven-representation SI panel"})
    for _, row in static_penalty_seed.iterrows():
        rows.append({"panel": "a", "record_type": "paired_penalty_seed", "group": row["category"], "representation": row["modality"], "seed": row["seed"], "label_count": np.nan, "metric": "MAE penalty", "value": row["MAE_penalty_eV"], "spread": np.nan, "definition": "Reduced-representation MAE minus paired full-multimodal MAE; positive values indicate degradation"})
    selected_modalities = set(static_penalty_summary["modality"]) | {"Multimodal"}
    for _, row in static_modality_seed[static_modality_seed["modality"].isin(selected_modalities)].iterrows():
        rows.append({"panel": "a", "record_type": "selected_representation_seed", "group": row["category"], "representation": row["modality"], "seed": row["seed"], "label_count": np.nan, "metric": "MAE", "value": row["MAE"], "spread": np.nan, "definition": "Actual supervised test MAE displayed in panel a"})
    for _, row in static_penalty_summary.iterrows():
        rows.append({"panel": "a", "record_type": "paired_penalty_summary", "group": row["category"], "representation": row["modality"], "seed": np.nan, "label_count": np.nan, "metric": "MAE penalty", "value": row["mean"], "spread": row["std"], "definition": f"Mean ± seed SD; selected by rule: {row['selection_rule']}; positive in {int(row['positive_seeds'])}/{int(row['n_seeds'])} paired seeds"})
    for _, row in uncertainty_seed.iterrows():
        rows.append({"panel": "b", "record_type": "seed", "group": row["group"], "seed": row["seed"], "label_count": row["train_size"], "metric": "Error-uncertainty Spearman rho", "value": row["uncertainty_error_spearman"], "spread": np.nan, "definition": "Final-round hybrid acquisition for the fixed selected representation"})
    for _, row in uncertainty_summary.iterrows():
        rows.append({"panel": "b", "record_type": "summary", "group": row["group"], "seed": np.nan, "label_count": 60, "metric": "Error-uncertainty Spearman rho", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD at the final 60-label budget; higher means uncertainty ranks prediction error more faithfully"})
    for _, row in al_group_seed_round.iterrows():
        rows.append({"panel": "c", "record_type": "seed", "group": row["group"], "seed": row["seed"], "label_count": row["train_size"], "metric": "MAE", "value": row["MAE"], "spread": np.nan, "definition": "Within-seed/round result for the fixed selected representation; hybrid acquisition"})
    for _, row in trajectory.iterrows():
        rows.append({"panel": "c", "record_type": "summary", "group": row["group"], "seed": np.nan, "label_count": row["train_size"], "metric": "MAE", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD; hybrid acquisition"})
    advantage_pivot = al_group_seed_round.pivot(index=["seed", "train_size"], columns="group", values="MAE").reset_index()
    for baseline in ["Single", "Bimodal"]:
        for _, row in advantage_pivot.iterrows():
            rows.append({"panel": "c", "record_type": "paired_advantage", "group": f"vs {baseline}", "seed": row["seed"], "label_count": row["train_size"], "metric": "Multimodal MAE advantage", "value": row[baseline] - row["Multimodal"], "spread": np.nan, "definition": f"Paired {baseline} MAE minus Multimodal MAE; positive values favor Multimodal"})
    for _, row in aulc_seed.iterrows():
        rows.append({"panel": "c", "record_type": "seed_integral", "group": row["group"], "seed": row["seed"], "label_count": "10-60", "metric": "Integrated MAE", "value": row["integrated_mae"], "spread": np.nan, "definition": f"Normalized trapezoidal AULC; initial={row['initial_mae']:.6f}; final={row['final_mae']:.6f}"})
    for _, row in aulc_summary.iterrows():
        rows.append({"panel": "c", "record_type": "summary_integral", "group": row["group"], "seed": np.nan, "label_count": "10-60", "metric": "Integrated MAE", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD"})
    aulc_pivot = aulc_seed.pivot(index="seed", columns="group", values="integrated_mae")
    for baseline in ["Single", "Bimodal"]:
        paired_reduction = 100 * (aulc_pivot[baseline] - aulc_pivot["Multimodal"]) / aulc_pivot[baseline]
        for seed, value in paired_reduction.items():
            rows.append({"panel": "c", "record_type": "paired_aulc_reduction", "group": f"Multimodal vs {baseline}", "seed": seed, "label_count": "10-60", "metric": "AULC reduction (%)", "value": value, "spread": np.nan, "definition": f"100 × ({baseline} AULC - Multimodal AULC) / {baseline} AULC within the paired seed"})
    for _, row in final_mae_seed.iterrows():
        rows.append({"panel": "d", "record_type": "seed", "group": row["group"], "seed": row["seed"], "label_count": row["train_size"], "metric": "Final-round MAE", "value": row["MAE"], "spread": np.nan, "definition": "Paired seed result at the final 60-label budget; hybrid acquisition"})
    for _, row in final_mae_summary.iterrows():
        rows.append({"panel": "d", "record_type": "summary", "group": row["group"], "seed": np.nan, "label_count": 60, "metric": "Final-round MAE", "value": row["mean"], "spread": row["std"], "definition": "Mean ± seed SD; diamonds in panel d show the mean"})
    final_pivot = final_mae_seed.pivot(index="seed", columns="group", values="MAE")
    final_reduction = 100 * (final_pivot["Bimodal"] - final_pivot["Multimodal"]) / final_pivot["Bimodal"]
    for seed, value in final_reduction.items():
        rows.append({"panel": "d", "record_type": "paired_reduction", "group": "Multimodal vs Bimodal", "seed": seed, "label_count": 60, "metric": "Final-round MAE reduction (%)", "value": value, "spread": np.nan, "definition": "100 × (Bimodal MAE - Multimodal MAE) / Bimodal MAE within the paired seed"})
    pd.DataFrame(rows).to_csv(OUT / "fig3_grouped_source_data.csv", index=False)
    static_penalty_seed.to_csv(OUT / "fig3_panel_a_modality_removal_source_data.csv", index=False)
    static_modality_seed.to_csv(OUT / "figS_complete_modality_ablation_source_data.csv", index=False)

    static_lookup = static_summary.set_index("group")
    aulc_lookup = aulc_summary.set_index("group")
    static_multi = float(static_lookup.loc["Multimodal", "mean"])
    static_single = float(static_lookup.loc["Single", "mean"])
    static_bi = float(static_lookup.loc["Bimodal", "mean"])
    aulc_multi = float(aulc_lookup.loc["Multimodal", "mean"])
    aulc_single = float(aulc_lookup.loc["Single", "mean"])
    aulc_bi = float(aulc_lookup.loc["Bimodal", "mean"])
    aulc_paired_single = 100 * (aulc_pivot["Single"] - aulc_pivot["Multimodal"]) / aulc_pivot["Single"]
    aulc_paired_bi = 100 * (aulc_pivot["Bimodal"] - aulc_pivot["Multimodal"]) / aulc_pivot["Bimodal"]
    final_lookup = final_mae_summary.set_index("group")
    final_multi = float(final_lookup.loc["Multimodal", "mean"])
    final_single = float(final_lookup.loc["Single", "mean"])
    final_bi = float(final_lookup.loc["Bimodal", "mean"])
    uncertainty_lookup = uncertainty_summary.set_index("group")
    uncertainty_multi = float(uncertainty_lookup.loc["Multimodal", "mean"])
    uncertainty_single = float(uncertainty_lookup.loc["Single", "mean"])
    uncertainty_bi = float(uncertainty_lookup.loc["Bimodal", "mean"])
    penalty_lookup = static_penalty_summary.set_index("category")
    selected_bimodal = penalty_lookup.loc["Bimodal"]
    selected_single = penalty_lookup.loc["Single"]
    rep_lookup = static_rep_summary.set_index("modality")
    selected_single_mae = float(rep_lookup.loc[selected_single["modality"], "mean"])
    selected_bimodal_mae = float(rep_lookup.loc[selected_bimodal["modality"], "mean"])
    summary = f"""# Grouped Fig. 3 summary

## Aggregation rule

- **Single** is the arithmetic mean of Element, Structure, and Chemical prior within each seed.
- **Bimodal** is the arithmetic mean of Element + Structure, Element + Chemical prior, and Structure + Chemical prior within each seed.
- **Multimodal** is the fixed three-modality model.
- Panels b–d use panel-specific maximum-degradation selections. Panel b selects the lowest mean error–uncertainty correlation, panel c the highest mean integrated MAE, and panel d the highest mean final-round MAE within the Single and Bimodal categories. Means and s.d. are calculated across five paired seeds.

## Result

Among the tested reduced representations, `{selected_bimodal['modality']}` has the largest aggregate mean MAE within the bimodal category and increases paired MAE by {fmt_sig(selected_bimodal['mean'])} ± {fmt_sig(selected_bimodal['std'])} eV relative to full Multimodal ({int(selected_bimodal['positive_seeds'])}/{int(selected_bimodal['n_seeds'])} seeds). `{selected_single['modality']}` has the largest aggregate mean MAE within the single-modality category and increases paired MAE by {fmt_sig(selected_single['mean'])} ± {fmt_sig(selected_single['std'])} eV ({int(selected_single['positive_seeds'])}/{int(selected_single['n_seeds'])} seeds). The complete seven-representation comparison is retained in the SI figure.

Panel a displays the original test-MAE values for `{selected_single['modality']}`, `{selected_bimodal['modality']}`, and Multimodal, with percentage reductions annotated for Multimodal. Panel-b selections are `{panel_selection['b']['Single']}` and `{panel_selection['b']['Bimodal']}`; their mean error–uncertainty correlations are {fmt_sig(uncertainty_single)} and {fmt_sig(uncertainty_bi)}, compared with {fmt_sig(uncertainty_multi)} for Multimodal. Panel-c selections are `{panel_selection['c']['Single']}` and `{panel_selection['c']['Bimodal']}`; integrated active-learning MAE is {fmt_sig(aulc_single)}, {fmt_sig(aulc_bi)}, and {fmt_sig(aulc_multi)} eV. Panel-d selections are `{panel_selection['d']['Single']}` and `{panel_selection['d']['Bimodal']}`; final-round MAE is {fmt_sig(final_single)}, {fmt_sig(final_bi)}, and {fmt_sig(final_multi)} eV. The plotted percentage annotations are calculated from these aggregate means.

## Interpretation boundary

This is a panel-specific maximum-degradation view, not an average Single/Bimodal comparison. Each panel selects the poorest aggregate-mean configuration for its displayed metric, while the plot retains the original metric values for Single, Bimodal, and Multimodal. Percentage annotations summarize differences between aggregate means and are not confidence intervals. The bimodal supervised penalty in panel a is positive in two of three paired seeds and therefore supports a mean-level, not universal seed-wise, interpretation. Complete representation-level comparisons and the selection table must accompany the main-text figure in the SI.

## Caption draft

**Fig. 3 | Multimodal representations limit degradation under missing-modality stress tests.** **a,** Supervised test MAE for the full Multimodal representation and the Single and Bimodal configurations with the largest mean MAE in the complete ablation. Small points indicate individual seeds (n = 3); large symbols and horizontal error bars show mean ± s.d. **b,** Spearman correlation between predictive uncertainty and absolute error at the final 60-label budget. Within each reduced-representation category, the configuration with the lowest mean correlation is shown. Small points indicate individual seeds (n = 5); diamonds and horizontal error bars show mean ± s.d. **c,** Paired reduction in the area under the test-MAE learning curve (AULC) for Multimodal relative to the selected Single and Bimodal configurations over label budgets of 10–60. Small points indicate individual seeds (n = 5); diamonds and horizontal error bars show mean ± s.d. Positive values favour Multimodal. **d,** Final-round test MAE for the configurations with the largest mean final-round MAE within the Single and Bimodal categories. Points indicate individual seeds (n = 5), boxes show the interquartile range and median, and diamonds indicate means. Percentage changes are calculated within paired seeds in c and from aggregate means in a, b and d. Complete representation-level comparisons and selected configuration identities are provided in the Supplementary Information.

## Recommended Results paragraph

To assess robustness to missing information, we compared the full Multimodal representation with the most degraded Single and Bimodal configurations for each evaluation criterion, while retaining the complete representation-level comparison in the Supplementary Information. In supervised testing, Multimodal achieved an MAE of {fmt_sig(static_multi)} eV, compared with {fmt_sig(selected_single_mae)} eV for Single and {fmt_sig(selected_bimodal_mae)} eV for Bimodal, corresponding to reductions of {fmt_sig(100 * (selected_single_mae - static_multi) / selected_single_mae)}% and {fmt_sig(100 * (selected_bimodal_mae - static_multi) / selected_bimodal_mae)}%, respectively (Fig. 3a). Multimodal also yielded a higher error–uncertainty correlation ({fmt_sig(uncertainty_multi)}) than Single ({fmt_sig(uncertainty_single)}) and Bimodal ({fmt_sig(uncertainty_bi)}), equivalent to relative gains of {fmt_sig(100 * (uncertainty_multi - uncertainty_single) / uncertainty_single)}% and {fmt_sig(100 * (uncertainty_multi - uncertainty_bi) / uncertainty_bi)}% (Fig. 3b). Across the active-learning trajectory, the paired AULC reduction averaged {fmt_sig(aulc_paired_single.mean())}% relative to Single and {fmt_sig(aulc_paired_bi.mean())}% relative to Bimodal, with a positive reduction in all five seeds for both comparisons (Fig. 3c). At the final acquisition round, Multimodal reached an MAE of {fmt_sig(final_multi)} eV, representing reductions of {fmt_sig(100 * (final_single - final_multi) / final_single)}% and {fmt_sig(100 * (final_bi - final_multi) / final_bi)}% relative to Single and Bimodal, respectively (Fig. 3d). These stress tests indicate that integrating all three information sources improves predictive accuracy, uncertainty ranking and label efficiency under the evaluated missing-modality conditions.

## SI caption draft

**Fig. Sx | Complete supervised modality-ablation results.** Test MAE for the complete Multimodal representation, three single-modality representations and three bimodal representations. Small points show individual seeds; diamonds and error bars indicate mean ± s.d. across three seeds. The main-text panel displays the largest aggregate-mean degradation within the tested single- and bimodal-ablation categories.
"""
    (OUT / "fig3_grouped_summary.md").write_text(summary, encoding="utf-8")

    reproducibility = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "inputs": [str((AUDIT / "modality_by_seed.csv").resolve()), str((AL_DIR / "results.csv").resolve()), str((AL_DIR / "run_meta.json").resolve())],
        "aggregation": {
            "Single": STATIC_MEMBERS["Single"],
            "Bimodal": STATIC_MEMBERS["Bimodal"],
            "Multimodal": STATIC_MEMBERS["Multimodal"],
            "order": "group means are retained as background outputs; the main figure uses panel-specific maximum-degradation selections documented below",
        },
        "panel_a_selection": {
            "rule": "Within the supervised ablation, select the Single and Bimodal representations with the largest aggregate mean MAE.",
            "selected_single": str(selected_single["modality"]),
            "selected_bimodal": str(selected_bimodal["modality"]),
            "complete_ablation_output": "figS_complete_modality_ablation",
        },
        "panel_specific_active_learning_selection": {
            "rule": "Select the poorest mean representation within each category for the metric displayed in that panel: lowest Spearman rho in b, highest integrated MAE in c, and highest final-round MAE in d.",
            **panel_selection,
        },
        "active_learning": {"policy": "hybrid", "seeds": meta["seeds"], "rounds": meta["rounds"], "initial_size": meta["initial_size"], "batch_size": meta["batch_size"]},
        "model_training": "None; plotting and descriptive aggregation only.",
    }
    (OUT / "fig3_grouped_reproducibility.json").write_text(json.dumps(reproducibility, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    static_modality_seed, static_seed, static_summary, static_rep_summary, static_penalty_seed, static_penalty_summary, al_group_seed_round, trajectory, uncertainty_seed, uncertainty_summary, final_mae_seed, final_mae_summary, aulc_seed, aulc_summary, meta, panel_selection = load_grouped_data()
    build_figure(static_penalty_seed, static_penalty_summary, al_group_seed_round, uncertainty_seed, uncertainty_summary, trajectory, aulc_seed, final_mae_seed)
    build_si_ablation_figure(static_modality_seed, static_rep_summary)
    write_outputs(static_modality_seed, static_seed, static_summary, static_rep_summary, static_penalty_seed, static_penalty_summary, al_group_seed_round, trajectory, uncertainty_seed, uncertainty_summary, final_mae_seed, final_mae_summary, aulc_seed, aulc_summary, meta, panel_selection)
    print(f"Wrote grouped Fig. 3 bundle to {OUT.resolve()}")


if __name__ == "__main__":
    main()
