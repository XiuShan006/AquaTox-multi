from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import dgl
import joblib
import numpy as np
import pandas as pd
import torch

from rfm.evaluate import _load_checkpoint, _reconstruct_model
from rfm.featurizers import ReactionFeaturizer

TASKS = (
    "fish_EC50", "fish_EC10",
    "aquatic_invertebrates_EC50", "aquatic_invertebrates_EC10",
    "algae_EC50", "algae_EC10",
)
EFFECTS = ("DVP", "GRO", "ITX", "MOR", "MPH", "POP", "REP")
MEDIA_TYPES = ("FW", "SW", "OTHER")
REQUIRED_COLUMNS = ("smiles", "Duration_Value", "effect", "media_type")


def load_fold_model(checkpoint_path: str | Path,
                    duration_scalers: Mapping[str, str | Path],
                    device: str = "cpu") -> dict:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    model = _reconstruct_model(checkpoint).to(device)
    model.eval()
    missing = sorted(set(checkpoint["tasks"]) - set(duration_scalers))
    if missing:
        raise ValueError(f"Missing duration scaler paths for tasks: {missing}")
    scalers = {}
    for task in checkpoint["tasks"]:
        path = Path(duration_scalers[task])
        if not path.is_file():
            raise FileNotFoundError(
                f"Training duration scaler not found for {task}: {path}. "
                "Prepare the fold preprocessing artifacts before prediction."
            )
        scalers[task] = joblib.load(path)
    return {"model": model, "checkpoint": checkpoint, "scalers": scalers,
            "device": device, "path": checkpoint_path}


def find_duration_scalers(preprocessed_dir: str | Path, fold_id: int,
                          tasks: Sequence[str] = TASKS) -> dict[str, Path]:
    root = Path(preprocessed_dir)
    found = {}
    for task in tasks:
        matches = sorted(root.glob(
            f"duration_scaler_*_fold{int(fold_id)}_{task}.pkl"
        ))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one training duration scaler for {task} in {root}; "
                f"found {len(matches)}. Run the training preprocessing step first."
            )
        found[task] = matches[0]
    return found


def _one_hot(values: Sequence[str], categories: Sequence[str]) -> torch.Tensor:
    index = {value: i for i, value in enumerate(categories)}
    try:
        return torch.tensor([[1.0 if index[value] == i else 0.0
                              for i in range(len(categories))]
                             for value in values], dtype=torch.float32)
    except KeyError as exc:
        raise ValueError(f"Unknown condition value: {exc.args[0]!r}") from exc


def _validate_records(records: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(records.columns))
    if missing:
        raise ValueError(f"Prediction input is missing columns: {missing}")
    if records.empty:
        raise ValueError("Prediction input must contain at least one row")
    durations = pd.to_numeric(records["Duration_Value"], errors="coerce")
    if durations.isna().any() or (durations < 0).any():
        raise ValueError("Duration_Value must contain non-negative numeric values")


def predict_ensemble(folds: Sequence[dict], records: pd.DataFrame,
                     tasks: Sequence[str] = TASKS) -> pd.DataFrame:
    _validate_records(records)
    requested = list(tasks)
    unknown = sorted(set(requested) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    if not folds:
        raise ValueError("At least one loaded fold is required")

    smiles = records["smiles"].astype(str).tolist()
    featurizer = ReactionFeaturizer()
    graphs = dgl.batch([featurizer.featurize_smiles_single(s) for s in smiles])
    effects = _one_hot(records["effect"].astype(str).tolist(), EFFECTS)
    media = _one_hot(records["media_type"].astype(str).tolist(), MEDIA_TYPES)
    durations = records["Duration_Value"].astype(float).to_numpy()
    output = records.copy().reset_index(drop=True)
    output.insert(0, "molecule_index", np.arange(len(output)))

    per_fold = {task: [] for task in requested}
    with torch.inference_mode():
        for fold in folds:
            device = fold["device"]
            fold_graph = graphs.to(device)
            fold_effects = effects.to(device)
            fold_media = media.to(device)
            for task in requested:
                scaler = fold["scalers"].get(task)
                if scaler is None:
                    raise ValueError(f"Fold has no duration scaler for {task}")
                scaled_duration = scaler.transform(
                    np.log1p(durations).reshape(-1, 1)
                ).reshape(-1)
                result = fold["model"](
                    fold_graph,
                    torch.tensor(scaled_duration, dtype=torch.float32, device=device),
                    fold_effects,
                    fold_media,
                    smiles_list=smiles,
                    requested_tasks=[task],
                )[task]
                params = fold["checkpoint"]["label_scalers"][task]
                mean, scale = float(params["mean"]), float(params["scale"])
                values = result["mu"].detach().cpu().numpy() * scale + mean
                per_fold[task].append(values)

    for task, values in per_fold.items():
        stacked = np.stack(values)
        output[f"pred_{task}_log10_mg_per_l"] = stacked.mean(axis=0)
        if len(folds) > 1:
            output[f"pred_{task}_fold_sd_log10_mg_per_l"] = stacked.std(axis=0, ddof=1)
    return output
