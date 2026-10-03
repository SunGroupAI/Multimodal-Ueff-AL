"""Portable entry points. Run from the extracted release root."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def summaries():
    out = ROOT / "outputs" / "numeric_reanalysis"
    out.mkdir(parents=True, exist_ok=True)
    static = pd.read_csv(ROOT / "analysis_multimodal_AL" / "modality_by_seed.csv")
    raw = pd.read_csv(ROOT / "results" / "active_learning_ablation" / "results.csv")
    raw = raw[(raw.strategy == "hybrid") & (raw.model == "extratrees") & (raw.fusion == "early")].copy()
    raw["category"] = raw.modality_block.map(lambda x: "Multimodal" if x.count("+") == 2 else ("Bimodal" if "+" in x else "Single"))
    integral_rows = []
    for (category, block, seed), g in raw.groupby(["category", "modality_block", "seed"]):
        g = g.sort_values("train_size")
        x, y = g.train_size.to_numpy(float), g.MAE.to_numpy(float)
        integral_rows.append({"category": category, "block": block, "seed": seed,
                              "AULC": float(np.trapezoid(y, x) / (x[-1] - x[0]))})
    aulc = pd.DataFrame(integral_rows)
    aulc.to_csv(out / "all_representation_aulc_by_seed.csv", index=False)
    means = aulc.groupby(["category", "block"]).AULC.mean().reset_index()
    pivot = aulc.pivot(index="seed", columns="block", values="AULC")
    multi = pivot["comp+struct+electronic"]
    comparisons = []
    for _, row in means[means.category != "Multimodal"].iterrows():
        reduction = 100 * (pivot[row.block] - multi) / pivot[row.block]
        comparisons.append({"category": row.category, "block": row.block,
                            "mean_reduction_pct": float(reduction.mean()),
                            "sd_reduction_pct": float(reduction.std(ddof=1)),
                            "positive_seeds": int((reduction > 0).sum())})
    pd.DataFrame(comparisons).to_csv(out / "all_aulc_comparisons.csv", index=False)
    selected = {}
    for category in ["Single", "Bimodal"]:
        selected[category] = means[means.category == category].sort_values("AULC", ascending=False).iloc[0].block
    static.groupby(["category", "modality"])[["MAE", "RMSE", "R2", "Spearman"]].agg(["mean", "std"]).to_csv(out / "supervised_all_representations.csv")
    final = raw[raw["round"] == raw["round"].max()]
    final.groupby(["category", "modality_block"])[["MAE", "uncertainty_error_spearman"]].agg(["mean", "std"]).to_csv(out / "al_final_all_representations.csv")
    info = {"label_budget": [10, 60], "selected_c_blocks": selected,
            "selection": "largest mean AULC within category", "comparisons": comparisons}
    (out / "summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["summary", "smoke", "supervised", "active-learning", "figure"], default="summary")
    args = parser.parse_args()
    if args.mode == "summary":
        summaries()
        return
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT / "config")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if args.mode == "figure":
        cmd = [sys.executable, "scripts/plot_fig3_grouped_representation_framework.py"]
    elif args.mode == "supervised":
        cmd = [sys.executable, "scripts/search_u_test_trifusion.py", "--data", "data/processed/Ueff_all_U_test_enhanced_modalities.csv",
               "--knowledge", "processed/u_knowledge_cards.csv", "--cgcnn_dir", "experiments/cgcnn_u_test_all_structures_20260623",
               "--split_dir_pattern", "processed/cgcnn_datasets/u_test_all_structures_seed{seed}",
               "--outdir", "outputs/supervised", "--seeds", "0", "1", "2", "--folds", "5"]
    else:
        smoke = args.mode == "smoke"
        cmd = [sys.executable, "scripts/run_active_learning_ablation.py", "--data", "data/processed/Ueff_clean_main.csv",
               "--models", "extratrees", "--fusion", "early", "--initial_size", "10", "--batch_size", "5",
               "--rounds", "1" if smoke else "10", "--test_size", "0.2", "--expected_n", "126",
               "--output", "outputs/smoke" if smoke else "outputs/active_learning",
               "--strategies", *( ["hybrid"] if smoke else ["random", "uncertainty", "diversity", "hybrid"] ),
               "--seeds", *( ["0"] if smoke else ["0", "1", "2", "3", "4"] )]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
