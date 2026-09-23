
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from rfm.artifacts import atomic_write_text
from rfm.models import SingleTaskYieldGNN, YieldGNN
from rfm.project_layout import AQUATOX_DATA_ROOT, AQUATOX_PREPROCESSED_ROOT
from rfm.trainer import YieldDataset
from rfm.trainer.trainer import _state_dict_digest, _task_private_state_keys


MODEL_REGISTRY = {
    "YieldGNN": YieldGNN,
    "SingleTaskYieldGNN": SingleTaskYieldGNN,
}
REQUIRED_CHECKPOINT_KEYS = {
    "schema_version", "model_class", "model_spec", "model", "variant_name",
    "model_kind", "regression_mode", "tasks", "seed", "training_spec",
    "data_contract", "label_scalers",
}
EC_PAIRS = {
    "fish_EC50": "fish_EC10",
    "fish_EC10": "fish_EC50",
    "aquatic_invertebrates_EC50": "aquatic_invertebrates_EC10",
    "aquatic_invertebrates_EC10": "aquatic_invertebrates_EC50",
    "algae_EC50": "algae_EC10",
    "algae_EC10": "algae_EC50",
}
DIAGNOSTIC_METADATA_COLUMNS = [
    "variant_name",
    "model_kind",
    "regression_mode",
    "dataset_layer",
    "protocol",
    "outer_fold_id",
    "seed",
    "checkpoint_sha256",
]
GATE_EXPORT_COLUMNS = [
    *DIAGNOSTIC_METADATA_COLUMNS,
    "sample_id",
    "task",
]
ORDERING_EXPORT_COLUMNS = [
    *DIAGNOSTIC_METADATA_COLUMNS,
    "anchor_sample_id",
    "standard_molecule_id",
    "anchor_task",
    "paired_task",
    "pred_ec50",
    "pred_ec10",
    "pred_gap",
    "violation",
    "ordering_context",
]
ROUTING_EXPORT_COLUMNS = [
    *DIAGNOSTIC_METADATA_COLUMNS,
    "sample_id",
    "task",
    "group_residual_ratio",
    "adapter_residual_ratio",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    root = relative_to.expanduser().resolve()
    try:
        recorded_path = resolved.relative_to(root)
    except ValueError:
        recorded_path = resolved
    return {
        "path": str(recorded_path),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _validate_output_contract(
    *,
    checkpoint_path: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    ordering: pd.DataFrame,
    resources: dict[str, Any],
    routing: pd.DataFrame | None = None,
) -> str:
    if predictions.empty:
        raise ValueError("Cannot publish an empty outer-test prediction table")

    actual_checkpoint_sha256 = _sha256(checkpoint_path)
    for name, frame in (
        ("predictions", predictions),
        ("metrics", metrics),
        ("gates", gates),
        ("ordering", ordering),
        ("routing", routing if routing is not None else pd.DataFrame()),
    ):
        if frame.empty:
            continue
        if "checkpoint_sha256" not in frame.columns:
            raise ValueError(f"{name} table is missing checkpoint_sha256")
        declared_hashes = frame["checkpoint_sha256"].astype(str).unique().tolist()
        if declared_hashes != [actual_checkpoint_sha256]:
            raise ValueError(
                f"{name} checkpoint_sha256 does not match the selected checkpoint"
            )

    if resources.get("status") != "completed":
        raise ValueError("Evaluation resources must have status='completed'")
    if resources.get("prediction_rows") != len(predictions):
        raise ValueError(
            "Evaluation resource prediction_rows does not match the prediction table"
        )
    task_inventory = resources.get("task_checkpoint_inventory", {})
    if not isinstance(task_inventory, dict):
        raise ValueError("Evaluation task checkpoint inventory is malformed")
    if task_inventory:
        if "task_checkpoint_sha256" not in predictions.columns:
            raise ValueError("Task-wise predictions are missing checkpoint SHA-256")
        expected_tasks = set(predictions["task"].astype(str))
        if set(task_inventory) != expected_tasks:
            raise ValueError("Task-wise checkpoint inventory does not match predictions")
        for task, record in task_inventory.items():
            if not isinstance(record, dict):
                raise ValueError(f"Task checkpoint inventory is malformed for {task}")
            path = Path(str(record.get("path", ""))).expanduser().resolve()
            digest = str(record.get("sha256", ""))
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"Task checkpoint artifact changed for {task}")
            declared = predictions.loc[
                predictions["task"].astype(str).eq(task),
                "task_checkpoint_sha256",
            ].astype(str).unique().tolist()
            if declared != [digest]:
                raise ValueError(f"Task checkpoint metadata mismatch for {task}")
    return actual_checkpoint_sha256


def _atomic_write_frame(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if target.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif target.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"Unsupported table extension: {target.suffix}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint payload must be a dictionary")
    missing = sorted(REQUIRED_CHECKPOINT_KEYS - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing schema-v1 keys: {missing}")
    if checkpoint["schema_version"] != 1:
        raise ValueError(
            f"Unsupported checkpoint schema_version={checkpoint['schema_version']!r}"
        )
    return checkpoint


def _resolve_contract(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[Path, Path, str, str, int]:
    contract = checkpoint["data_contract"]
    required = {
        "dataset_layer", "cv_protocol", "outer_fold_id", "dataset_sha256",
        "manifest_sha256",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"Checkpoint data_contract is missing {missing}")

    dataset_layer = args.dataset_layer or contract["dataset_layer"]
    cv_protocol = args.cv_protocol or contract["cv_protocol"]
    outer_fold_id = (
        args.outer_fold_id
        if args.outer_fold_id is not None
        else int(contract["outer_fold_id"])
    )
    for name, actual in (
        ("dataset_layer", dataset_layer),
        ("cv_protocol", cv_protocol),
        ("outer_fold_id", outer_fold_id),
    ):
        if actual != contract[name]:
            raise ValueError(
                f"Requested {name}={actual!r} does not match checkpoint {contract[name]!r}"
            )

    aquatox_root = args.aquatox_data_root.expanduser().resolve()
    model_csv = aquatox_root / "data" / f"{dataset_layer}_model.csv"
    manifest = aquatox_root / "manifests" / "cv_roles.csv.gz"
    for required_path in (model_csv, manifest):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    for path, expected, label in (
        (model_csv, contract["dataset_sha256"], "dataset"),
        (manifest, contract["manifest_sha256"], "manifest"),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"Frozen {label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    return model_csv, manifest, dataset_layer, cv_protocol, outer_fold_id


def _reconstruct_model(checkpoint: dict[str, Any]) -> torch.nn.Module:
    model_class = checkpoint["model_class"]
    if model_class not in MODEL_REGISTRY:
        raise ValueError(f"Checkpoint model_class is not allowed: {model_class!r}")
    model = MODEL_REGISTRY[model_class](**checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if list(model.tasks) != list(checkpoint["tasks"]):
        raise ValueError("Reconstructed model tasks do not match checkpoint tasks")
    if model.model_kind != checkpoint["model_kind"]:
        raise ValueError("Reconstructed model_kind does not match checkpoint")
    if model.regression_mode != checkpoint["regression_mode"]:
        raise ValueError("Reconstructed regression_mode does not match checkpoint")
    if checkpoint["model_spec"].get("fusion_mode", "direct") == "projected":
        preprocessing = checkpoint.get("feature_preprocessing")
        if not isinstance(preprocessing, dict):
            raise ValueError("Projected-fusion checkpoint is missing feature preprocessing")
        expected = {
            "fusion_mode": "projected",
            "fit_role": "outer_train",
            "fit_unit": "unique_standardized_smiles",
        }
        if any(preprocessing.get(key) != value for key, value in expected.items()):
            raise ValueError("Projected-fusion feature preprocessing contract is invalid")
        fitted = getattr(model.molecular_encoder, "descriptor_scaler_fitted", None)
        if fitted is None or not bool(fitted.item()):
            raise ValueError("Projected-fusion descriptor scaler was not restored")
    return model


def _resolve_taskwise_checkpoints(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    selection = checkpoint.get("training_spec", {}).get(
        "checkpoint_selection", "macro"
    )
    if selection == "macro":
        return {}
    if selection == "task_soup":
        manifest_path = checkpoint_path.parent / "task_soup_selection.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Task-soup checkpoint manifest is invalid") from exc
        if (
            manifest.get("status") != "completed"
            or manifest.get("selection") != "task_best_uniform_soup"
            or manifest.get("variant_name") != checkpoint["variant_name"]
            or int(manifest.get("seed", -1)) != int(checkpoint["seed"])
            or int(manifest.get("outer_fold_id", -1))
            != int(checkpoint["data_contract"]["outer_fold_id"])
            or manifest.get("soup_checkpoint") != checkpoint_path.name
            or manifest.get("soup_checkpoint_sha256") != _sha256(checkpoint_path)
            or checkpoint.get("selected_task") is not None
        ):
            raise ValueError("Task-soup checkpoint manifest contract failed")
        return {}
    if selection == "balanced":
        manifest_path = checkpoint_path.parent / "balanced_checkpoint_selection.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Balanced checkpoint manifest is invalid") from exc
        if (
            manifest.get("status") != "completed"
            or manifest.get("selection")
            != "macro_within_1pct_minimax_task_regret"
            or manifest.get("variant_name") != checkpoint["variant_name"]
            or int(manifest.get("seed", -1)) != int(checkpoint["seed"])
            or int(manifest.get("outer_fold_id", -1))
            != int(checkpoint["data_contract"]["outer_fold_id"])
            or manifest.get("balanced_checkpoint") != checkpoint_path.name
            or manifest.get("balanced_checkpoint_sha256") != _sha256(checkpoint_path)
            or checkpoint.get("selected_task") is not None
        ):
            raise ValueError("Balanced checkpoint manifest contract failed")
        return {}
    if selection == "task_private_finetune":
        manifest_path = checkpoint_path.parent / "task_private_finetune_selection.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Task-private checkpoint manifest is invalid") from exc
        task_records = manifest.get("tasks")
        if (
            manifest.get("status") != "completed"
            or manifest.get("selection")
            != "shared_base_task_private_adapter_head_finetune"
            or manifest.get("variant_name") != checkpoint["variant_name"]
            or int(manifest.get("seed", -1)) != int(checkpoint["seed"])
            or int(manifest.get("outer_fold_id", -1))
            != int(checkpoint["data_contract"]["outer_fold_id"])
            or manifest.get("merged_checkpoint") != checkpoint_path.name
            or manifest.get("merged_checkpoint_sha256") != _sha256(checkpoint_path)
            or checkpoint.get("selected_task") is not None
            or manifest.get("private_key_sets_pairwise_disjoint") is not True
            or not isinstance(task_records, dict)
            or set(task_records) != set(checkpoint["tasks"])
        ):
            raise ValueError("Task-private checkpoint manifest contract failed")
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("Task-private checkpoint has no model state")
        private_keys = _task_private_state_keys(state, checkpoint["tasks"])
        private_union = {
            name for task_keys in private_keys.values() for name in task_keys
        }
        shared_keys = sorted(set(state) - private_union)
        if (
            manifest.get("shared_state_sha256")
            != _state_dict_digest(state, shared_keys)
            or manifest.get("merged_state_sha256") != _state_dict_digest(state)
        ):
            raise ValueError("Task-private checkpoint state hash mismatch")
        for task, keys in private_keys.items():
            record = task_records.get(task)
            if (
                not isinstance(record, dict)
                or record.get("trainable_keys") != keys
                or int(record.get("trainable_parameter_tensors", -1)) != len(keys)
                or int(record.get("trainable_parameters", -1))
                != sum(int(state[name].numel()) for name in keys)
            ):
                raise ValueError(f"Task-private inventory mismatch for {task}")
        return {}
    if selection != "taskwise":
        raise ValueError(f"Unsupported checkpoint_selection={selection!r}")

    manifest_path = checkpoint_path.parent / "taskwise_selection.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Task-wise checkpoint manifest is invalid") from exc
    if (
        manifest.get("status") != "completed"
        or manifest.get("selection") != "taskwise_inner_rmse"
        or manifest.get("variant_name") != checkpoint["variant_name"]
        or int(manifest.get("seed", -1)) != int(checkpoint["seed"])
        or int(manifest.get("outer_fold_id", -1))
        != int(checkpoint["data_contract"]["outer_fold_id"])
        or manifest.get("selector_checkpoint") != checkpoint_path.name
        or manifest.get("selector_checkpoint_sha256") != _sha256(checkpoint_path)
    ):
        raise ValueError("Task-wise checkpoint manifest contract failed")

    task_records = manifest.get("tasks")
    if not isinstance(task_records, dict) or set(task_records) != set(
        checkpoint["tasks"]
    ):
        raise ValueError("Task-wise checkpoint inventory is incomplete")
    resolved: dict[str, dict[str, Any]] = {}
    checkpoint_root = checkpoint_path.parent.resolve()
    for task in checkpoint["tasks"]:
        record = task_records[task]
        if not isinstance(record, dict):
            raise ValueError(f"Task-wise checkpoint record is invalid for {task}")
        relative = Path(str(record.get("checkpoint", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError(f"Task-wise checkpoint path escapes its directory: {task}")
        task_path = (checkpoint_root / relative).resolve()
        if not task_path.is_relative_to(checkpoint_root) or not task_path.is_file():
            raise FileNotFoundError(task_path)
        digest = _sha256(task_path)
        if record.get("checkpoint_sha256") != digest:
            raise ValueError(f"Task-wise checkpoint SHA-256 mismatch for {task}")
        payload = _load_checkpoint(task_path)
        if (
            payload.get("selected_task") != task
            or payload.get("variant_name") != checkpoint["variant_name"]
            or payload.get("seed") != checkpoint["seed"]
            or payload.get("tasks") != checkpoint["tasks"]
            or payload.get("model_class") != checkpoint["model_class"]
            or payload.get("model_spec") != checkpoint["model_spec"]
            or payload.get("data_contract") != checkpoint["data_contract"]
            or payload.get("label_scalers") != checkpoint["label_scalers"]
            or payload.get("training_spec", {}).get("checkpoint_selection")
            != "taskwise"
        ):
            raise ValueError(f"Task-wise checkpoint payload mismatch for {task}")
        resolved[task] = {
            "path": task_path,
            "sha256": digest,
            "payload": payload,
        }
    return resolved


def _task_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_pred - y_true
    denominator = float(np.square(y_true - y_true.mean()).sum())
    r2 = (
        float("nan")
        if denominator == 0
        else 1.0 - float(np.square(residual).sum()) / denominator
    )
    pearson = (
        float("nan")
        if np.std(y_true) == 0 or np.std(y_pred) == 0
        else float(np.corrcoef(y_pred, y_true)[0, 1])
    )
    return {
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE": float(np.mean(np.abs(residual))),
        "R2": r2,
        "Pearson_r": pearson,
    }


def _verify_scaler(dataset: YieldDataset, expected: dict[str, float], task: str) -> None:
    actual_mean = float(dataset.label_scaler.mean_[0])
    actual_scale = float(dataset.label_scaler.scale_[0])
    if not np.isclose(actual_mean, expected["mean"], rtol=0, atol=1e-12):
        raise ValueError(f"Label scaler mean mismatch for {task}")
    if not np.isclose(actual_scale, expected["scale"], rtol=0, atol=1e-12):
        raise ValueError(f"Label scaler scale mismatch for {task}")


def evaluate(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    pd.DataFrame,
]:
    evaluation_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evaluation_started = time.perf_counter()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    model_csv, manifest, dataset_layer, cv_protocol, outer_fold_id = _resolve_contract(
        args, checkpoint
    )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _reconstruct_model(checkpoint).to(device)
    model.eval()
    taskwise_checkpoints = _resolve_taskwise_checkpoints(
        checkpoint_path, checkpoint
    )
    checkpoint_selection = checkpoint.get("training_spec", {}).get(
        "checkpoint_selection", "macro"
    )

    tasks = list(checkpoint["tasks"])
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError(f"Invalid checkpoint tasks: {tasks!r}")
    if checkpoint["model_kind"] == "single_task" and len(tasks) != 1:
        raise ValueError("Single-task checkpoint must contain exactly one task")

    datasets: dict[str, YieldDataset] = {}
    for task in tasks:
        if task not in checkpoint["label_scalers"]:
            raise ValueError(f"Checkpoint is missing label scaler metadata for {task}")
        dataset = YieldDataset(
            task=task,
            split_role="outer_test",
            preprocessed_dir=str(args.preprocessed_dir.expanduser().resolve()),
            file_path=str(model_csv),
            manifest_path=str(manifest),
            cv_protocol=cv_protocol,
            outer_fold_id=outer_fold_id,
            dataset_layer=dataset_layer,
        )
        _verify_scaler(dataset, checkpoint["label_scalers"][task], task)
        datasets[task] = dataset

    shared_graph_cache: dict[str, Any] = {}
    for dataset in datasets.values():
        dataset._graph_cache = shared_graph_cache
    for dataset in datasets.values():
        dataset.preprocess()
        dataset._preload_graphs()

    prediction_frames = []
    metric_rows = []
    gate_frames = []
    ordering_frames = []
    routing_frames = []
    checkpoint_sha256 = _sha256(checkpoint_path)
    is_mmoe = checkpoint["model_kind"] == "mmoe"
    is_multi_task = checkpoint["model_kind"] in {"mmoe", "grouped"}
    export_routing = bool(getattr(model, "supports_routing_diagnostics", False))
    inference_seconds = 0.0

    for task in tasks:
        selected_task_checkpoint = taskwise_checkpoints.get(task)
        if selected_task_checkpoint is not None:
            model.load_state_dict(
                selected_task_checkpoint["payload"]["model"], strict=True
            )
        task_checkpoint_sha256 = (
            selected_task_checkpoint["sha256"]
            if selected_task_checkpoint is not None
            else checkpoint_sha256
        )
        dataset = datasets[task]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset.collate,
            pin_memory=False,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        paired_task = EC_PAIRS.get(task)
        requested_tasks = [task]
        if is_multi_task and paired_task in tasks:
            requested_tasks.append(paired_task)

        sample_ids: list[str] = []
        y_true_batches = []
        y_pred_batches = []
        y_std_batches = []
        paired_pred_batches = []
        gate_batches = []
        group_residual_batches = []
        adapter_residual_batches = []
        scaler = dataset.label_scaler

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        task_inference_started = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                graph, duration, effect, media, labels, _ghs, smiles, batch_ids = batch
                forward_kwargs = {
                    "smiles_list": smiles,
                    "requested_tasks": requested_tasks,
                    "return_gate_weights": is_mmoe,
                }
                if export_routing:
                    forward_kwargs["return_routing_diagnostics"] = True
                outputs = model(
                    graph.to(device),
                    duration.to(device),
                    effect.to(device),
                    media.to(device),
                    **forward_kwargs,
                )
                output = outputs[task]
                y_pred_batches.append(
                    scaler.inverse_transform(
                        output["mu"].detach().cpu().numpy().reshape(-1, 1)
                    ).reshape(-1)
                )
                y_true_batches.append(
                    scaler.inverse_transform(labels.numpy().reshape(-1, 1)).reshape(-1)
                )
                if checkpoint["regression_mode"] == "heteroscedastic":
                    y_std_batches.append(
                        torch.exp(output["log_sigma"]).detach().cpu().numpy().reshape(-1)
                        * float(scaler.scale_[0])
                    )
                else:
                    y_std_batches.append(np.full(len(batch_ids), np.nan, dtype=float))
                if paired_task in outputs:
                    paired_scaler = datasets[paired_task].label_scaler
                    paired_pred_batches.append(
                        paired_scaler.inverse_transform(
                            outputs[paired_task]["mu"].detach().cpu().numpy().reshape(-1, 1)
                        ).reshape(-1)
                    )
                if is_mmoe:
                    gate_batches.append(output["gate_weights"].detach().cpu().numpy())
                if export_routing:
                    for key, destination in (
                        ("group_residual_ratio", group_residual_batches),
                        ("adapter_residual_ratio", adapter_residual_batches),
                    ):
                        value = output.get(key)
                        destination.append(
                            value.detach().cpu().numpy().reshape(-1)
                            if value is not None
                            else np.full(len(batch_ids), np.nan, dtype=float)
                        )
                sample_ids.extend(batch_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - task_inference_started

        if sample_ids != dataset.sample_ids:
            raise AssertionError(f"Outer-test sample order mismatch for {task}")
        if len(set(sample_ids)) != len(sample_ids):
            raise AssertionError(f"Duplicate outer-test sample IDs for {task}")
        y_true = np.concatenate(y_true_batches)
        y_pred = np.concatenate(y_pred_batches)
        y_std = np.concatenate(y_std_batches)
        if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
            raise FloatingPointError(f"Non-finite outer-test predictions for {task}")
        if checkpoint["regression_mode"] == "heteroscedastic" and (
            not np.isfinite(y_std).all() or (y_std <= 0).any()
        ):
            raise FloatingPointError(f"Invalid outer-test uncertainty for {task}")

        molecule_by_sample = dict(zip(dataset.sample_ids, dataset.molecule_ids))
        split_parent_by_sample = dict(
            zip(dataset.sample_ids, dataset.split_parent_ids)
        )
        split_group_by_sample = dict(zip(dataset.sample_ids, dataset.split_group_ids))
        common_columns = {
            "variant_name": checkpoint["variant_name"],
            "model_kind": checkpoint["model_kind"],
            "regression_mode": checkpoint["regression_mode"],
            "dataset_layer": dataset_layer,
            "protocol": cv_protocol,
            "outer_fold_id": outer_fold_id,
            "seed": checkpoint["seed"],
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_selection": checkpoint_selection,
            "task_checkpoint_sha256": task_checkpoint_sha256,
        }
        prediction_frames.append(pd.DataFrame({
            **common_columns,
            "sample_id": sample_ids,
            "standard_molecule_id": [molecule_by_sample[value] for value in sample_ids],
            "split_parent_id": [split_parent_by_sample[value] for value in sample_ids],
            "split_group_id": [split_group_by_sample[value] for value in sample_ids],
            "task": task,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_std": y_std,
        }))
        metric_rows.append({
            **common_columns, "task": task, "n": len(sample_ids),
            **_task_metrics(y_true, y_pred),
        })

        if is_mmoe:
            gates = np.concatenate(gate_batches)
            if not np.isfinite(gates).all():
                raise FloatingPointError(f"Non-finite gate weights for {task}")
            if not np.allclose(gates.sum(axis=1), 1.0, rtol=0, atol=1e-5):
                raise FloatingPointError(f"Gate weights do not sum to one for {task}")
            gate_data = {**common_columns, "sample_id": sample_ids, "task": task}
            for expert_index in range(gates.shape[1]):
                gate_data[f"gate_expert_{expert_index}"] = gates[:, expert_index]
            gate_frames.append(pd.DataFrame(gate_data))

        if export_routing:
            group_ratios = np.concatenate(group_residual_batches)
            adapter_ratios = np.concatenate(adapter_residual_batches)
            for name, values in (
                ("group_residual_ratio", group_ratios),
                ("adapter_residual_ratio", adapter_ratios),
            ):
                finite = values[np.isfinite(values)]
                if (finite < 0).any():
                    raise FloatingPointError(f"Negative {name} values for {task}")
            routing_frames.append(
                pd.DataFrame(
                    {
                        **common_columns,
                        "sample_id": sample_ids,
                        "task": task,
                        "group_residual_ratio": group_ratios,
                        "adapter_residual_ratio": adapter_ratios,
                    }
                )
            )

        if paired_pred_batches:
            paired_pred = np.concatenate(paired_pred_batches)
            if not np.isfinite(paired_pred).all():
                raise FloatingPointError(f"Non-finite paired predictions for {task}")
            if "EC50" in task:
                pred_ec50, pred_ec10 = y_pred, paired_pred
            else:
                pred_ec50, pred_ec10 = paired_pred, y_pred
            pred_gap = pred_ec50 - pred_ec10
            ordering_frames.append(pd.DataFrame({
                **common_columns,
                "anchor_sample_id": sample_ids,
                "standard_molecule_id": [molecule_by_sample[value] for value in sample_ids],
                "anchor_task": task,
                "paired_task": paired_task,
                "pred_ec50": pred_ec50,
                "pred_ec10": pred_ec10,
                "pred_gap": pred_gap,
                "violation": pred_gap < 0,
                "ordering_context": "same_covariate_counterfactual",
            }))

    predictions = pd.concat(prediction_frames, ignore_index=True)
    expected_ids = {value for dataset in datasets.values() for value in dataset.sample_ids}
    if predictions["sample_id"].duplicated().any():
        raise AssertionError("Duplicate sample IDs across outer-test prediction tables")
    if set(predictions["sample_id"]) != expected_ids:
        raise AssertionError("Outer-test predictions do not exactly cover the frozen fold")
    metrics = pd.DataFrame(metric_rows)
    gates = (
        pd.concat(gate_frames, ignore_index=True)
        if gate_frames
        else pd.DataFrame(columns=GATE_EXPORT_COLUMNS)
    )
    ordering = (
        pd.concat(ordering_frames, ignore_index=True)
        if ordering_frames
        else pd.DataFrame(columns=ORDERING_EXPORT_COLUMNS)
    )
    routing = (
        pd.concat(routing_frames, ignore_index=True)
        if routing_frames
        else pd.DataFrame(columns=ROUTING_EXPORT_COLUMNS)
    )
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        cuda_device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated = 0
        peak_reserved = 0
        cuda_device_name = None
    resources = {
        "schema_version": 1,
        "status": "completed",
        "stage": "outer_test_evaluation",
        "started_at_utc": evaluation_started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": float(time.perf_counter() - evaluation_started),
        "inference_seconds": float(inference_seconds),
        "prediction_rows": int(len(predictions)),
        "routing_rows": int(len(routing)),
        "samples_per_second": (
            float(len(predictions) / inference_seconds)
            if inference_seconds > 0
            else None
        ),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "cuda_device_name": cuda_device_name,
        "checkpoint_selection": checkpoint_selection,
        "task_checkpoint_inventory": {
            task: {
                "path": str(record["path"]),
                "sha256": record["sha256"],
            }
            for task, record in taskwise_checkpoints.items()
        },
    }
    return predictions, metrics, gates, ordering, resources, routing


def _write_outputs(
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    ordering: pd.DataFrame,
    resources: dict[str, Any],
    routing: pd.DataFrame | None = None,
) -> None:
    run_dir = args.output_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint_sha256 = _validate_output_contract(
        checkpoint_path=checkpoint_path,
        predictions=predictions,
        metrics=metrics,
        gates=gates,
        ordering=ordering,
        resources=resources,
        routing=routing,
    )
    training_resources = run_dir / "resources" / "training_resources.json"
    if training_resources.is_file():
        try:
            training_resource_payload = json.loads(
                training_resources.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid training resource record: {training_resources}") from exc
        if training_resource_payload.get("status") != "completed":
            raise ValueError("Training resources must have status='completed'")

    predictions_dir = run_dir / "predictions"
    published_tables = [
        (predictions, "outer_test_predictions"),
        (metrics, "outer_test_metrics"),
        (gates, "outer_test_gates"),
        (ordering, "outer_test_ordering_same_context"),
    ]
    if routing is not None:
        published_tables.append((routing, "outer_test_routing_diagnostics"))
    for frame, stem in published_tables:
        _atomic_write_frame(frame, predictions_dir / f"{stem}.csv")
        _atomic_write_frame(frame, predictions_dir / f"{stem}.parquet")

    resources_path = run_dir / "resources" / "evaluation_resources.json"
    atomic_write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n", resources_path
    )

    artifact_paths = [checkpoint_path, resources_path]
    task_checkpoint_inventory = resources.get("task_checkpoint_inventory", {})
    artifact_paths.extend(
        Path(str(record["path"]))
        for record in task_checkpoint_inventory.values()
    )
    if training_resources.is_file():
        artifact_paths.append(training_resources)
    for _frame, stem in published_tables:
        artifact_paths.extend(
            [predictions_dir / f"{stem}.csv", predictions_dir / f"{stem}.parquet"]
        )

    success = {
        "schema_version": 1,
        "status": "completed",
        "variant_name": str(predictions["variant_name"].iloc[0]),
        "model_kind": str(predictions["model_kind"].iloc[0]),
        "regression_mode": str(predictions["regression_mode"].iloc[0]),
        "outer_fold_id": int(predictions["outer_fold_id"].iloc[0]),
        "seed": int(predictions["seed"].iloc[0]),
        "selected_checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": resources.get("checkpoint_selection", "macro"),
        "task_checkpoint_inventory": task_checkpoint_inventory,
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "gate_rows": len(gates),
        "ordering_rows": len(ordering),
        "routing_rows": 0 if routing is None else len(routing),
        "artifacts": [
            _artifact_record(path, relative_to=run_dir) for path in artifact_paths
        ],
    }
    atomic_write_text(
        json.dumps(success, indent=2, sort_keys=True) + "\n", run_dir / "_SUCCESS.json"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root", dest="aquatox_data_root", type=Path,
        default=AQUATOX_DATA_ROOT,
    )
    parser.add_argument(
        "--dataset-layer", choices=("common_intersection", "curated_full"), default=None
    )
    parser.add_argument("--cv-protocol", choices=("scaffold", "molecule"), default=None)
    parser.add_argument("--outer-fold-id", type=int, choices=range(5), default=None)
    parser.add_argument(
        "--preprocessed-dir", type=Path,
        default=AQUATOX_PREPROCESSED_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    _write_outputs(cli_args, *evaluate(cli_args))
