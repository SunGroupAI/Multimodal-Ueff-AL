#!/usr/bin/env python
"""Train CGCNN on prepared crystal graphs and export graph embeddings.

Requires an environment with torch and pymatgen. In this workspace,
``python`` has the required packages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


CGCNN_ROOT = repo_root() / "con-CDVAE" / "Con-CDVAE" / "cgcnn"
sys.path.insert(0, str(CGCNN_ROOT))

from cgcnn.data import CIFData, collate_pool  # noqa: E402
from cgcnn.model import CrystalGraphConvNet  # noqa: E402


class Normalizer:
    def __init__(self, values: torch.Tensor) -> None:
        self.mean = torch.mean(values)
        self.std = torch.std(values)
        if not torch.isfinite(self.std) or float(self.std) < 1e-12:
            self.std = torch.tensor(1.0, dtype=values.dtype)

    def norm(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean) / self.std

    def denorm(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}


def make_loader(data_dir: Path, target: str, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = CIFData(str(data_dir), target, file="%s.csv" % split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_pool)


def to_device(batch_input, device: torch.device):
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = batch_input
    return (
        atom_fea.to(device),
        nbr_fea.to(device),
        nbr_fea_idx.to(device),
        [idx.to(device) for idx in crystal_atom_idx],
    )


def crystal_embedding(model: CrystalGraphConvNet, batch_input) -> torch.Tensor:
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = batch_input
    atom_fea = model.embedding(atom_fea)
    for conv_func in model.convs:
        atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
    crys_fea = model.pooling(atom_fea, crystal_atom_idx)
    crys_fea = model.conv_to_fc(model.conv_to_fc_softplus(crys_fea))
    crys_fea = model.conv_to_fc_softplus(crys_fea)
    if hasattr(model, "fcs") and hasattr(model, "softpluses"):
        for fc, softplus in zip(model.fcs, model.softpluses):
            crys_fea = softplus(fc(crys_fea))
    return crys_fea


def evaluate(model, loader, normalizer, device):
    model.eval()
    y_true, y_pred, ids = [], [], []
    with torch.no_grad():
        for batch_input, target, cif_ids in loader:
            batch_input = to_device(batch_input, device)
            output = model(*batch_input)
            pred = normalizer.denorm(output.detach().cpu()).view(-1).numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(target.view(-1).numpy().tolist())
            ids.extend(cif_ids)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "R2": float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else float("nan"),
        "ids": ids,
        "y_true": y_true_arr,
        "y_pred": y_pred_arr,
    }


def export_embeddings(model, data_dir: Path, target: str, batch_size: int, device: torch.device, outdir: Path) -> dict:
    loader = make_loader(data_dir, target, "all", batch_size, False)
    model.eval()
    ids, embeddings = [], []
    with torch.no_grad():
        for batch_input, _, cif_ids in loader:
            batch_input = to_device(batch_input, device)
            h = crystal_embedding(model, batch_input).detach().cpu().numpy()
            embeddings.append(h)
            ids.extend(cif_ids)
    matrix = np.vstack(embeddings)
    np.save(outdir / "cgcnn_embeddings.npy", matrix)
    pd.DataFrame({"material_id": ids}).to_csv(outdir / "cgcnn_embedding_index.csv", index=False)
    return {"n_embeddings": int(matrix.shape[0]), "embedding_dim": int(matrix.shape[1])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--target", default="U_eV")
    parser.add_argument("--outdir", type=Path, default=repo_root() / "results" / "cgcnn_abo3_u")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--atom_fea_len", type=int, default=64)
    parser.add_argument("--h_fea_len", type=int, default=128)
    parser.add_argument("--n_conv", type=int, default=3)
    parser.add_argument("--n_h", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_loader = make_loader(args.data_dir, args.target, "train", args.batch_size, True)
    val_loader = make_loader(args.data_dir, args.target, "val", args.batch_size, False)
    test_loader = make_loader(args.data_dir, args.target, "test", args.batch_size, False)

    sample_dataset = CIFData(str(args.data_dir), args.target, file="train.csv")
    structures, _, _ = sample_dataset[0]
    model = CrystalGraphConvNet(
        structures[0].shape[-1],
        structures[1].shape[-1],
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=False,
    ).to(device)

    train_targets = torch.tensor(pd.read_csv(args.data_dir / "train.csv")[args.target].values, dtype=torch.float32).view(-1, 1)
    normalizer = Normalizer(train_targets)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    history = []
    best_val = float("inf")
    best_path = args.outdir / "cgcnn_model_best.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_input, target, _ in train_loader:
            batch_input = to_device(batch_input, device)
            target_normed = normalizer.norm(target).to(device)
            pred = model(*batch_input)
            loss = criterion(pred, target_normed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, normalizer, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_MAE": val["MAE"], "val_RMSE": val["RMSE"], "val_R2": val["R2"]}
        history.append(row)
        if val["MAE"] < best_val:
            best_val = val["MAE"]
            serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            torch.save({"model": model.state_dict(), "normalizer": normalizer.state_dict(), "args": serializable_args}, best_path)
        print("epoch=%03d train_loss=%.4f val_MAE=%.4f" % (epoch + 1, row["train_loss"], row["val_MAE"]))

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, normalizer, device)
    pd.DataFrame(history).to_csv(args.outdir / "cgcnn_training_history.csv", index=False)
    pd.DataFrame({"material_id": test["ids"], "y_true": test["y_true"], "y_pred": test["y_pred"]}).to_csv(args.outdir / "cgcnn_test_predictions.csv", index=False)
    embed_meta = export_embeddings(model, args.data_dir, args.target, args.batch_size, device, args.outdir)
    meta = {
        "data_dir": str(args.data_dir),
        "target": args.target,
        "device": str(device),
        "best_val_MAE": best_val,
        "test_metrics": {k: test[k] for k in ["MAE", "RMSE", "R2"]},
        "embedding": embed_meta,
        "normalizer": normalizer.state_dict(),
        "reproducibility_note": "Prepare data with scripts/prepare_cgcnn_dataset.py, then run this script with the recorded arguments.",
    }
    (args.outdir / "cgcnn_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Test MAE %.4f RMSE %.4f R2 %.4f" % (test["MAE"], test["RMSE"], test["R2"]))
    print("Wrote CGCNN outputs to %s" % args.outdir)


if __name__ == "__main__":
    main()
