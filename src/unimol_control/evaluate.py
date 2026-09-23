import argparse
import sys
from pathlib import Path

import gin
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

dir_path = Path(__file__).parent.absolute()
sys.path.append(str(dir_path))

import gin_config
from models import ToxicGNN
from trainer import YieldDataset


def evaluate(
    *,
    task: str,
    cfg: str,
    model_path: str,
    batch_size: int,
    device: str,
    preprocessed_dir: str,
    split: str = "valid",
    file_path: str = "data/aquatox/data/common_intersection_model.csv",
    roles_path: str = "data/aquatox/manifests/cv_roles.csv.gz",
    conformer_root: str = "data/aquatox/conformers",
    conformer_audit_path: str = "data/aquatox/audit/conformer_audit.csv",
    outer_fold_id: int = 0,
    split_protocol: str = "scaffold",
):
    device = device if torch.cuda.is_available() else "cpu"

    gin.parse_config_files_and_bindings([cfg], bindings=['run_name="eval"'])

    model = ToxicGNN()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    dataset = YieldDataset(
        task=task,
        split=split,
        preprocessed_dir=preprocessed_dir,
        sep=",",
        contrastive=False,
        file_path=file_path,
        split_type="manifest",
        roles_path=roles_path,
        split_protocol=split_protocol,
        outer_fold_id=outer_fold_id,
        conformer_root=conformer_root,
        conformer_audit_path=conformer_audit_path,
        unimol_precompute=True,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)

    label_scaler = joblib.load(Path(preprocessed_dir) / f"label_scaler_{task}_train.pkl")
    mse_metric = MeanSquaredError().to(device)
    mae_metric = MeanAbsoluteError().to(device)
    r2_metric = R2Score().to(device)

    with torch.no_grad():
        for batch in dataloader:
            graph, duration_values, effect_onehots, labels, ghs_classes, smiles_list = batch
            if isinstance(graph, tuple):
                graph = tuple(t.to(device) for t in graph)
            else:
                graph = graph.to(device)
            duration_values = duration_values.to(device)
            effect_onehots = effect_onehots.to(device)
            labels = labels.to(device)

            outputs = model(
                graph,
                duration_values,
                effect_onehots,
                smiles_list=smiles_list,
            )
            preds_scaled = outputs[task]

            preds_log10 = label_scaler.inverse_transform(
                preds_scaled.cpu().numpy().reshape(-1, 1)
            ).squeeze()
            labels_log10 = label_scaler.inverse_transform(
                labels.cpu().numpy().reshape(-1, 1)
            ).squeeze()

            preds_tensor = torch.tensor(np.atleast_1d(preds_log10), device=device, dtype=torch.float32)
            labels_tensor = torch.tensor(np.atleast_1d(labels_log10), device=device, dtype=torch.float32)

            mse_metric.update(preds_tensor, labels_tensor)
            mae_metric.update(preds_tensor, labels_tensor)
            r2_metric.update(preds_tensor, labels_tensor)

    metrics = {
        "task": task,
        "mse": mse_metric.compute().item(),
        "mae": mae_metric.compute().item(),
        "r2": r2_metric.compute().item(),
    }
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ToxicGNN Uni-Mol checkpoint.")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--cfg", type=str, default="src/unimol_control/configs/unimol_scaffold.gin")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preprocessed_dir", type=str, default="preprocessed_graphs_local/aquatox")
    parser.add_argument("--file_path", type=str, default="data/aquatox/data/common_intersection_model.csv")
    parser.add_argument("--roles_path", type=str, default="data/aquatox/manifests/cv_roles.csv.gz")
    parser.add_argument("--conformer_root", type=str, default="data/aquatox/conformers")
    parser.add_argument("--conformer_audit_path", type=str, default="data/aquatox/audit/conformer_audit.csv")
    parser.add_argument("--outer_fold_id", type=int, choices=range(5), default=0)
    parser.add_argument("--split_protocol", type=str, choices=("scaffold", "molecule"), default="scaffold")
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(
        task=args.task,
        cfg=args.cfg,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        preprocessed_dir=args.preprocessed_dir,
        split=args.split,
        file_path=args.file_path,
        roles_path=args.roles_path,
        conformer_root=args.conformer_root,
        conformer_audit_path=args.conformer_audit_path,
        outer_fold_id=args.outer_fold_id,
        split_protocol=args.split_protocol,
    )
