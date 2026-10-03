"""Verify included file integrity, identities, splits and numerical snapshots."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file(), f"Missing {entry['path']}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"Checksum mismatch: {entry['path']}"
    for path in (ROOT / "scripts").rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    tables = {}
    for name, n in [("Ueff_clean_main.csv", 126), ("Ueff_all_U_test_enhanced_modalities.csv", 180)]:
        df = pd.read_csv(ROOT / "data" / "processed" / name)
        assert len(df) == n and not df.id.duplicated().any()
        assert np.isfinite(pd.to_numeric(df.U_eV)).all()
        assert all(any(c.startswith(prefix) for c in df) for prefix in ["comp__", "struct__", "elec__"])
        tables[name] = df
    df = tables["Ueff_all_U_test_enhanced_modalities.csv"]
    for seed in [0, 1, 2]:
        folder = ROOT / "processed" / "cgcnn_datasets" / f"u_test_all_structures_seed{seed}"
        parts = [pd.read_csv(folder / f"{part}.csv") for part in ["train", "val", "test"]]
        ids = [set(p.material_id.astype(str)) for p in parts]
        assert not any(ids[i] & ids[j] for i, j in [(0, 1), (0, 2), (1, 2)])
        assert set.union(*ids) == set(df.id.astype(str))
        for part in parts:
            expected = df.set_index("id").U_eV.loc[part.material_id.astype(str)].to_numpy(float)
            assert np.allclose(part.U_eV, expected, atol=1e-12, rtol=0)
        emb_dir = ROOT / "experiments" / "cgcnn_u_test_all_structures_20260623" / f"seed{seed}"
        emb = np.load(emb_dir / "cgcnn_embeddings.npy", allow_pickle=False)
        idx = pd.read_csv(emb_dir / "cgcnn_embedding_index.csv")
        assert emb.shape == (180, 128) and np.isfinite(emb).all()
        assert not idx.material_id.duplicated().any() and set(idx.material_id.astype(str)) == set(df.id.astype(str))
    raw = pd.read_csv(ROOT / "results" / "active_learning_ablation" / "results.csv")
    assert set(raw.seed) == set(range(5)) and set(raw["round"]) == set(range(11))
    assert len(raw) == 5 * 7 * 4 * 11
    assert not raw.duplicated(["seed", "modality_block", "fusion", "model", "strategy", "round"]).any()
    assert set(raw.model) == {"extratrees"} and set(raw.fusion) == {"early"}
    assert (raw.train_size == 10 + raw["round"] * 5).all()
    archived = pd.read_csv(ROOT / "analysis_multimodal_AL" / "modality_by_seed.csv")
    assert len(archived) == 21
    detail = pd.read_csv(ROOT / "experiments" / "u_test_trifusion_search_enhanced_graph_20260623" / "trifusion_search_by_seed.csv")
    best = pd.read_csv(ROOT / "experiments" / "u_test_trifusion_search_enhanced_graph_20260623" / "ablation" / "ablation_best_by_block.csv")
    mapping = {"element": "Element", "graph": "Structure", "text": "Chemical prior",
               "element+graph": "Element + Structure", "element+text": "Element + Chemical prior",
               "graph+text": "Structure + Chemical prior", "element+graph+text": "Multimodal"}
    for _, row in best.iterrows():
        selected = detail[(detail.variant == row.variant) & (detail.block == row.block) & (detail.fusion == row.fusion)]
        a = archived[archived.modality == mapping[row.block]].sort_values("seed")
        b = selected.groupby("seed")[["MAE", "RMSE", "R2", "Spearman"]].mean().sort_index()
        assert np.allclose(a[["MAE", "RMSE", "R2", "Spearman"]], b, rtol=0, atol=1e-12)
    print(f"PASS: {len(manifest['files'])} hashes; 126/180 inputs; three split/embedding sets; 1540 AL records; supervised source-data mapping.")


if __name__ == "__main__":
    main()
