from pathlib import Path
from typing import Dict, Literal, Optional, List
import hashlib
import json
import os
import random
import tempfile

import gin
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast

from .logger.logger_base import LoggerBase
from .optimizer_base import OptimizerBase
from .yield_dataset import YieldDataset
from .utils import infer_metric_direction


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if path.suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif path.suffix == ".parquet":
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"Unsupported table extension: {path.suffix}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@gin.configurable()
class YieldTrainer:
    def __init__(
        self,
        *,
        run_dir: str | Path,
        tasks: List[str],
        train_datasets: Dict[str, YieldDataset],
        valid_datasets: Dict[str, YieldDataset],
        train_batch_size: int,
        valid_batch_size: int,
        train_metrics: Dict[str, Metric],
        valid_metrics: Dict[str, Dict[str, Metric]],
        model: nn.Module,
        logger: LoggerBase,
        optimizer: OptimizerBase,
        n_epochs: int,
        preprocessed_dir: str,
        n_steps_per_epoch: int = 500,
        device: str = "auto",
        checkpoint_best: bool = False,
        best_metric: str = "avg_r2",
        metric_direction: Literal["auto", "min", "max"] = "auto",
        gradient_clipping_norm: float = 5.0,
        num_workers: int = 1,
        valid_every_n_epochs: int = 3,
        log_train_every_n_batches: int = 10,
        lambda_reg: float = 1e-3,
        lambda_cls: float = 0.1,
        lambda_ord: float = 0.05,
        patience: int = 20,
        validation_granularity: Literal["context_median", "sample"] = "context_median",
        seed: int = 42,
        variant_name: str = "toxic_unimol",
        config_path: Optional[str] = None,
    ):
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        assert metric_direction in ("auto", "min", "max")
        self.run_dir = Path(run_dir)
        self.tasks = tasks
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        if validation_granularity not in {"context_median", "sample"}:
            raise ValueError(
                "validation_granularity must be 'context_median' or 'sample'"
            )
        self.validation_granularity = validation_granularity
        self.seed = int(seed)
        self.variant_name = str(variant_name)
        self.config_path = (
            str(Path(config_path).expanduser().resolve()) if config_path else None
        )
        fold_ids = {int(dataset.outer_fold_id) for dataset in train_datasets.values()}
        protocols = {str(dataset.split_protocol) for dataset in train_datasets.values()}
        if len(fold_ids) != 1 or len(protocols) != 1:
            raise ValueError("Training datasets disagree on fold or split protocol")
        self.outer_fold_id = fold_ids.pop()
        self.split_protocol = protocols.pop()
        self.validation_history: List[Dict[str, float]] = []
        self.last_valid_predictions: Optional[pd.DataFrame] = None

        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics

        self.logger = logger
        self.optimizer = optimizer
        self.model = model
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch

        self.lambda_reg = lambda_reg
        self.patience = patience
        self.epochs_no_improve = 0
        self.early_stop = False
        self.scaler = GradScaler()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        for metric in self.train_metrics.values():
            metric.to(self.device)

        for task_metrics in self.valid_metrics.values():
            for metric in task_metrics.values():
                metric.to(self.device)

        self.checkpoint_best = checkpoint_best
        self.best_metric = best_metric
        self.metric_direction = (
            infer_metric_direction(self.best_metric)
            if metric_direction == "auto"
            else metric_direction
        )
        self.best_valid_metric = float("inf") if self.metric_direction == "min" else float("-inf")
        self.best_valid_metrics: Dict[str, float] = {}

        self.gradient_clipping_norm = gradient_clipping_norm
        self.valid_every_n_epochs = valid_every_n_epochs
        self.log_train_every_n_batches = log_train_every_n_batches

        self.model.to(self.device)
        self.optimizer.initialize(model=self.model)
        self.total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        self.pretrained_checkpoint = self._build_pretrained_checkpoint_contract()

        if self.validation_granularity == "sample":
            for task in self.tasks:
                train_dataset = self.train_datasets[task]
                valid_dataset = self.valid_datasets[task]
                if not train_dataset.manifest_split or not valid_dataset.manifest_split:
                    raise ValueError("Sample-level validation requires frozen manifest splits")
                if len(valid_dataset.sample_ids) != len(valid_dataset):
                    raise ValueError(f"Incomplete validation sample IDs for {task}")
                for attribute in (
                    "molecule_ids",
                    "split_parent_ids",
                    "split_group_ids",
                ):
                    if len(getattr(valid_dataset, attribute)) != len(valid_dataset):
                        raise ValueError(
                            f"Incomplete validation {attribute} for {task}"
                        )
        self.data_contract = self._build_data_contract()

        self.train_loaders = {
            task: DataLoader(
                train_datasets[task],
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=train_datasets[task].collate,
            )
            for task in tasks
        }
        self.valid_loaders = {
            task: DataLoader(
                valid_datasets[task],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=valid_datasets[task].collate,
            )
            for task in tasks
        }

        self.reg_loss_fn = nn.SmoothL1Loss()

        self.cls_loss_fn = nn.NLLLoss()
        self.lambda_cls = lambda_cls
        self.lambda_ord = lambda_ord

        self._ghs_thresholds = {
            task: [-1.0, 0.0, 1.0] if 'EC10' in task else [0.0, 1.0, 2.0]
            for task in tasks
        }

        self.ec_pairs = {
            "fish_EC50": "fish_EC10",
            "fish_EC10": "fish_EC50",
            "aquatic_invertebrates_EC50": "aquatic_invertebrates_EC10",
            "aquatic_invertebrates_EC10": "aquatic_invertebrates_EC50",
            "algae_EC50": "algae_EC10",
            "algae_EC10": "algae_EC50",
        }

        self._scaler_mean: Dict[str, torch.Tensor] = {}
        self._scaler_std: Dict[str, torch.Tensor] = {}
        for task in tasks:
            sc = self.train_datasets[task].label_scaler
            self._scaler_mean[task] = torch.tensor(
                sc.mean_[0], dtype=torch.float32
            )
            self._scaler_std[task] = torch.tensor(
                sc.scale_[0], dtype=torch.float32
            )

        self.train_iterators = {
            task: iter(self.train_loaders[task]) for task in self.tasks
        }

        dataset_sizes = {t: len(self.train_datasets[t]) for t in tasks}

        boost_factors = {
            "fish_EC50": 1.0,
            "fish_EC10": 1.0,
            "aquatic_invertebrates_EC50": 1.0,
            "aquatic_invertebrates_EC10": 1.0,
            "algae_EC50": 1.0,
            "algae_EC10": 1.0,
        }

        weighted_sizes = {
            t: dataset_sizes[t] * boost_factors.get(t, 1.0)
            for t in tasks
        }

        total_size = sum(weighted_sizes.values())
        if total_size > 0:
            self.task_prob = [weighted_sizes[t] / total_size for t in tasks]
        else:
            self.task_prob = [1.0/len(tasks)] * len(tasks)

        self.loss_weights = {
            t: 1.0 for t in tasks
        }

    def _ghs_soft_logprobs(self, mu: torch.Tensor, task: str) -> torch.Tensor:
        mean_t = self._scaler_mean[task].to(mu.device)
        std_t = self._scaler_std[task].to(mu.device)
        mu_log10 = mu.float() * std_t + mean_t

        thresholds = self._ghs_thresholds[task]
        T = 0.5

        p_below = [torch.sigmoid(-(mu_log10 - thr) / T) for thr in thresholds]

        p0 = p_below[0]
        p1 = (p_below[1] - p_below[0]).clamp(min=1e-8)
        p2 = (p_below[2] - p_below[1]).clamp(min=1e-8)
        p3 = (1.0 - p_below[2]).clamp(min=1e-8)
        probs = torch.stack([p0, p1, p2, p3], dim=-1).clamp(min=1e-8)
        return torch.log(probs / probs.sum(dim=-1, keepdim=True))

    def l2_regularization(self) -> torch.Tensor:
        return self.lambda_reg * sum(p.norm(2).pow(2) for p in self.model.parameters() if p.requires_grad)

    def _build_data_contract(self) -> Dict[str, object]:
        dataset = self.train_datasets[self.tasks[0]]
        model_table = Path(dataset.file_path)
        aquatox_root = model_table.parents[1]
        paths = {
            "model_table": model_table,
            "cv_roles": Path(dataset.roles_path) if dataset.roles_path else None,
            "conformer_audit": dataset.source_conformer_audit_path,
            "hash_manifest": aquatox_root / "manifest_hashes.json",
        }
        artifacts = {}
        for name, path in paths.items():
            if path is None:
                continue
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            artifacts[name] = {
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": _sha256_file(resolved),
            }
        return {
            "dataset_layer": "common_intersection",
            "protocol": self.split_protocol,
            "outer_fold_id": self.outer_fold_id,
            "artifacts": artifacts,
        }

    def _label_scaler_contract(self) -> Dict[str, Dict[str, float]]:
        return {
            task: {
                "mean": float(self.train_datasets[task].label_scaler.mean_[0]),
                "scale": float(self.train_datasets[task].label_scaler.scale_[0]),
            }
            for task in self.tasks
        }

    def _build_pretrained_checkpoint_contract(self) -> Optional[Dict[str, object]]:
        path_value = getattr(self.model, "unimol_checkpoint_path", None)
        if path_value is None:
            if self.variant_name == "unimol":
                raise ValueError("Formal Uni-Mol requires a bound Uni-Mol checkpoint")
            return None
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            if self.variant_name == "unimol":
                raise FileNotFoundError(path)
            return None
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    def _write_best_validation_predictions(self) -> None:
        if self.validation_granularity != "sample":
            return
        if self.last_valid_predictions is None or self.last_valid_predictions.empty:
            raise RuntimeError("Best checkpoint has no sample-level validation predictions")
        prediction_dir = self.run_dir / "predictions"
        _atomic_write_frame(
            self.last_valid_predictions,
            prediction_dir / "inner_val_best.parquet",
        )
        _atomic_write_frame(
            self.last_valid_predictions,
            prediction_dir / "inner_val_best.csv",
        )

    @torch.no_grad()
    def valid_step(self) -> Dict[str, float]:
        self.model.eval()

        task_losses = {t: [] for t in self.tasks}
        all_metrics: Dict[str, float] = {}
        label_scalers = {
            task: self.train_datasets[task].label_scaler for task in self.tasks
        }
        prediction_frames: List[pd.DataFrame] = []

        for task in self.tasks:
            loader = self.valid_loaders[task]
            dataset = self.valid_datasets[task]
            agg_dict: Dict[tuple, List] = {}
            sample_predictions: List[float] = []
            sample_labels: List[float] = []
            sample_ids: List[str] = []
            cursor = 0

            for batch in tqdm(loader, desc=f"Processing {task}", leave=False):
                try:
                    graph, duration_values, effect_onehots, labels, ghs_classes, smiles_list = batch

                    if isinstance(graph, tuple):
                        graph = tuple(t.to(self.device) for t in graph)
                    else:
                        graph = graph.to(self.device)
                    duration_values = duration_values.to(self.device)
                    effect_onehots = effect_onehots.to(self.device)
                    labels = labels.to(self.device)
                    ghs_classes = ghs_classes.to(self.device)

                    outputs = self.model(
                        graph,
                        duration_values,
                        effect_onehots,
                        smiles_list=smiles_list
                    )
                    logits = outputs[task]

                    reg_loss = self.reg_loss_fn(logits, labels)
                    if task in self._scaler_mean:
                        cls_log_probs = self._ghs_soft_logprobs(logits, task)
                        cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                        loss = reg_loss + self.lambda_cls * cls_loss
                    else:
                        loss = reg_loss
                    task_losses[task].append(loss.item())

                    logits_numpy = logits.cpu().numpy().reshape(-1, 1)
                    labels_numpy = labels.cpu().numpy().reshape(-1, 1)
                    preds_log10 = np.atleast_1d(
                        label_scalers[task].inverse_transform(logits_numpy).squeeze()
                    )
                    labels_log10 = np.atleast_1d(
                        label_scalers[task].inverse_transform(labels_numpy).squeeze()
                    )

                    if self.validation_granularity == "sample":
                        stop = cursor + len(smiles_list)
                        expected_smiles = dataset.smiles_list[cursor:stop]
                        if list(smiles_list) != list(expected_smiles):
                            raise AssertionError(
                                f"Validation sample order changed for {task}"
                            )
                        expected_labels = np.asarray(
                            dataset.labels_list[cursor:stop], dtype=float
                        )
                        if not np.allclose(
                            labels_log10,
                            expected_labels,
                            rtol=0.0,
                            atol=1e-5,
                        ):
                            raise AssertionError(
                                f"Validation labels disagree with frozen rows for {task}"
                            )
                        sample_predictions.extend(preds_log10.astype(float).tolist())
                        sample_labels.extend(expected_labels.tolist())
                        sample_ids.extend(dataset.sample_ids[cursor:stop])
                        cursor = stop
                    else:
                        effect_idx = effect_onehots.cpu().numpy().argmax(axis=1)
                        dur_rounded = duration_values.cpu().numpy().round(4)
                        for i, smi in enumerate(smiles_list):
                            key = (smi, int(effect_idx[i]), float(dur_rounded[i]))
                            agg_dict.setdefault(key, []).append(
                                (float(preds_log10[i]), float(labels_log10[i]))
                            )

                except Exception as e:
                    if self.validation_granularity == "sample":
                        raise RuntimeError(
                            f"Sample-level validation failed for task {task}"
                        ) from e
                    print(f"Error processing batch for task {task}: {str(e)}")
                    continue

            if self.validation_granularity == "sample":
                if cursor != len(dataset) or sample_ids != dataset.sample_ids:
                    raise AssertionError(
                        f"Incomplete sample-level validation coverage for {task}: "
                        f"{cursor}/{len(dataset)}"
                    )
                if len(set(sample_ids)) != len(sample_ids):
                    raise AssertionError(f"Duplicate validation sample IDs for {task}")
                all_preds = np.asarray(sample_predictions, dtype=float)
                all_labels = np.asarray(sample_labels, dtype=float)
                if not np.isfinite(all_preds).all() or not np.isfinite(all_labels).all():
                    raise FloatingPointError(
                        f"Non-finite sample-level validation values for {task}"
                    )
                all_metrics[f"{task}_n_samples"] = len(all_preds)
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "variant_name": self.variant_name,
                            "protocol": self.split_protocol,
                            "outer_fold_id": self.outer_fold_id,
                            "seed": self.seed,
                            "split_role": "inner_val",
                            "sample_id": sample_ids,
                            "standard_molecule_id": dataset.molecule_ids,
                            "split_parent_id": dataset.split_parent_ids,
                            "split_group_id": dataset.split_group_ids,
                            "task": task,
                            "y_true": all_labels,
                            "y_pred": all_preds,
                        }
                    )
                )
            else:
                all_preds = np.asarray(
                    [np.median([value[0] for value in values]) for values in agg_dict.values()],
                    dtype=float,
                )
                all_labels = np.asarray(
                    [np.median([value[1] for value in values]) for values in agg_dict.values()],
                    dtype=float,
                )
                all_metrics[f"{task}_n_agg"] = len(agg_dict)

            if all_preds.size == 0:
                raise RuntimeError(f"No validation predictions produced for {task}")
            preds_tensor = torch.tensor(all_preds, device=self.device, dtype=torch.float32)
            labels_tensor = torch.tensor(all_labels, device=self.device, dtype=torch.float32)
            if task in self.valid_metrics:
                for metric_fn in self.valid_metrics[task].values():
                    metric_fn.update(preds_tensor, labels_tensor)

            avg_loss = np.mean(task_losses[task]) if task_losses[task] else 0.0
            all_metrics[f"{task}_loss"] = float(avg_loss)

            if task in self.valid_metrics:
                for metric_name, metric_fn in self.valid_metrics[task].items():
                    try:
                        val = metric_fn.compute()
                        value = val.item() if isinstance(val, torch.Tensor) else float(val)
                        if self.validation_granularity == "sample" and not np.isfinite(value):
                            raise FloatingPointError(
                                f"Non-finite {metric_name} for sample-level validation/{task}"
                            )
                        all_metrics[f"{task}_{metric_name}"] = value
                    except Exception as e:
                        if self.validation_granularity == "sample":
                            raise
                        print(f"Warning: {e} for task '{task}' metric '{metric_name}'")
                        all_metrics[f"{task}_{metric_name}"] = None
                    finally:
                        metric_fn.reset()

        all_metrics["loss"] = float(
            np.mean([all_metrics[f"{task}_loss"] for task in self.tasks])
        )
        for metric_name, aggregate_name in (
            ("rmse", "avg_rmse"),
            ("r2", "avg_r2"),
            ("pearson", "avg_pearson"),
        ):
            values = [all_metrics.get(f"{task}_{metric_name}") for task in self.tasks]
            finite_values = [value for value in values if value is not None]
            if len(finite_values) != len(self.tasks):
                if self.validation_granularity == "sample":
                    raise RuntimeError(
                        f"Missing task metrics for sample-level {aggregate_name}"
                    )
                all_metrics[aggregate_name] = None
            else:
                all_metrics[aggregate_name] = float(np.mean(finite_values))

        self.last_valid_predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if self.validation_granularity == "sample"
            else None
        )

        self.logger.log_metrics(metrics=all_metrics, prefix="valid")

        self.model.train()
        return all_metrics

    def make_checkpoint(self, checkpoint_name: str, metrics: Optional[Dict[str, float]] = None):
        ckpt_dir = self.run_dir / "train" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dict = {
            "schema_version": 1,
            "variant_name": self.variant_name,
            "model_kind": "unimol_mmoe",
            "regression_mode": "deterministic",
            "tasks": list(self.tasks),
            "seed": self.seed,
            "data_contract": self.data_contract,
            "label_scalers": self._label_scaler_contract(),
            "pretrained_checkpoint": self.pretrained_checkpoint,
            "training_spec": {
                "selection_metric": self.best_metric,
                "metric_direction": self.metric_direction,
                "validation_granularity": self.validation_granularity,
                "validation_role": "inner_val",
                "n_epochs": self.n_epochs,
                "n_steps_per_epoch": self.n_steps_per_epoch,
                "valid_every_n_epochs": self.valid_every_n_epochs,
                "patience_validation_checks": self.patience,
            },
            "parameter_count": {
                "total": self.total_parameters,
                "trainable": self.trainable_parameters,
            },
            "config": (
                {
                    "path": self.config_path,
                    "sha256": _sha256_file(self.config_path),
                }
                if self.config_path
                else None
            ),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.optimizer.state_dict(),
            "metrics": metrics,
        }
        checkpoint_path = ckpt_dir / f"{checkpoint_name}.pt"
        temporary_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(ckpt_dict, temporary_path)
        os.replace(temporary_path, checkpoint_path)

    def train(self) -> Dict[str, float]:
        batch_no = 0
        losses = {t: [] for t in self.tasks}
        reg_losses_buf: list = []
        cls_losses_buf: list = []
        ord_losses_buf: list = []
        self.model.train()

        for epoch in range(self.n_epochs):
            if self.early_stop:
                print(f"[EarlyStop] at epoch={epoch}")
                break

            epoch_pbar = tqdm(range(self.n_steps_per_epoch), desc=f"Epoch {epoch+1}/{self.n_epochs}", leave=True)
            for step in epoch_pbar:
                selected_task = random.choices(self.tasks, weights=self.task_prob, k=1)[0]

                try:
                    batch = next(self.train_iterators[selected_task])
                except StopIteration:
                    self.train_iterators[selected_task] = iter(self.train_loaders[selected_task])
                    batch = next(self.train_iterators[selected_task])

                graph, duration_values, effect_onehots, labels, ghs_classes, smiles_list = batch

                if isinstance(graph, tuple):
                    graph = tuple(t.to(self.device) for t in graph)
                else:
                    graph = graph.to(self.device)
                duration_values = duration_values.to(self.device)
                effect_onehots = effect_onehots.to(self.device)
                labels = labels.to(self.device)
                ghs_classes = ghs_classes.to(self.device)

                with autocast():
                    outputs = self.model(
                        graph,
                        duration_values,
                        effect_onehots,
                        smiles_list=smiles_list
                    )
                    logits = outputs[selected_task]

                    reg_loss = self.reg_loss_fn(logits, labels)
                    if selected_task in self._scaler_mean:
                        cls_log_probs = self._ghs_soft_logprobs(logits, selected_task)
                        cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                        task_loss = reg_loss + self.lambda_cls * cls_loss
                        _cls_loss_val = cls_loss.item()
                    else:
                        task_loss = reg_loss
                        _cls_loss_val = 0.0

                    _ord_loss_val = 0.0
                    paired_task = self.ec_pairs.get(selected_task)
                    if (
                        paired_task
                        and paired_task in outputs
                        and self.lambda_ord > 0
                        and selected_task in self._scaler_mean
                        and paired_task in self._scaler_mean
                    ):
                        mean_sel = self._scaler_mean[selected_task].to(logits.device)
                        std_sel = self._scaler_std[selected_task].to(logits.device)
                        mean_pair = self._scaler_mean[paired_task].to(logits.device)
                        std_pair = self._scaler_std[paired_task].to(logits.device)

                        pred_sel_log10 = logits.float() * std_sel + mean_sel
                        pred_pair_log10 = outputs[paired_task].float() * std_pair + mean_pair
                        if 'EC50' in selected_task:
                            pred_ec50_log10, pred_ec10_log10 = pred_sel_log10, pred_pair_log10
                        else:
                            pred_ec50_log10, pred_ec10_log10 = pred_pair_log10, pred_sel_log10

                        ordinal_loss = torch.relu(pred_ec10_log10 - pred_ec50_log10).mean()
                        task_loss = task_loss + self.lambda_ord * ordinal_loss
                        _ord_loss_val = ordinal_loss.item()

                    _reg_loss_val = reg_loss.item()
                    loss_weight = self.loss_weights.get(selected_task, 1.0)
                    weighted_task_loss = task_loss * loss_weight

                l2_reg = self.l2_regularization()
                total_loss = weighted_task_loss + l2_reg

                if torch.isnan(total_loss):
                    print(f"NaN loss at epoch={epoch}, step={step}, skip this batch")
                    continue

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping_norm)
                self.scaler.step(self.optimizer.optimizer)
                self.scaler.update()

                losses[selected_task].append(task_loss.item())
                reg_losses_buf.append(_reg_loss_val)
                cls_losses_buf.append(_cls_loss_val)
                ord_losses_buf.append(_ord_loss_val)
                batch_no += 1

                for metric_name, metric_fn in self.train_metrics.items():
                    metric_fn.update(logits.detach().cpu(), labels.detach().cpu())

                if batch_no % self.log_train_every_n_batches == 0:
                    avg_reg = np.mean(reg_losses_buf) if reg_losses_buf else 0.0
                    avg_cls = np.mean(cls_losses_buf) if cls_losses_buf else 0.0
                    avg_ord = np.mean(ord_losses_buf) if ord_losses_buf else 0.0
                    l2_val = self.l2_regularization().item()

                    log_info = {
                        "reg_loss": avg_reg,
                        "cls_loss": avg_cls,
                        "cls_loss_weighted": avg_cls * self.lambda_cls,
                        "ord_loss": avg_ord,
                        "ord_loss_weighted": avg_ord * self.lambda_ord,
                        "l2_loss": l2_val,
                        "total_loss": avg_reg + avg_cls * self.lambda_cls + avg_ord * self.lambda_ord + l2_val,
                    }

                    for metric_name, metric_fn in self.train_metrics.items():
                        try:
                            val = metric_fn.compute()
                            if isinstance(val, torch.Tensor):
                                val = val.item()
                            log_info[metric_name] = val
                        except ValueError as e:
                            print(f"Warning: {e} for training metric '{metric_name}'. Setting it to None.")
                            log_info[metric_name] = None
                        finally:
                            metric_fn.reset()

                    self.logger.log_metrics(metrics=log_info, prefix="train")

                    losses = {tt: [] for tt in self.tasks}
                    reg_losses_buf.clear()
                    cls_losses_buf.clear()
                    ord_losses_buf.clear()
                    epoch_pbar.set_postfix(
                        reg=f"{avg_reg:.3f}",
                        cls=f"{avg_cls:.3f}",
                        cls_w=f"{avg_cls * self.lambda_cls:.3f}",
                        ord=f"{avg_ord:.3f}",
                    )

            if epoch % self.valid_every_n_epochs == 0 or epoch == self.n_epochs - 1:
                valid_metrics = self.valid_step()
                self.validation_history.append(dict(valid_metrics, epoch=epoch))
                _atomic_write_frame(
                    pd.DataFrame(self.validation_history),
                    self.run_dir / "validation_history.csv",
                )

                if self.checkpoint_best:
                    current_val = valid_metrics.get(self.best_metric, None)
                    if current_val is None:
                        print(f"Warning: best_metric='{self.best_metric}' not found or is None in valid_metrics.")
                        current_val = float("inf") if self.metric_direction == "min" else float("-inf")

                    if ((self.metric_direction == "min" and current_val < self.best_valid_metric)
                            or (self.metric_direction == "max" and current_val > self.best_valid_metric)):
                        self.logger.log_metrics(metrics=valid_metrics, prefix="best_valid")
                        self.best_valid_metrics = dict(valid_metrics, epoch=epoch)
                        self.best_valid_metric = current_val
                        self._write_best_validation_predictions()
                        self.make_checkpoint("best_model", metrics=self.best_valid_metrics)
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1

                    if self.epochs_no_improve >= self.patience:
                        print(f"No improvement after {self.patience} epochs, early stopping.")
                        self.early_stop = True
                else:
                    self.best_valid_metrics = dict(valid_metrics, epoch=epoch)

        self.make_checkpoint("last_model", metrics=self.best_valid_metrics)
        return self.best_valid_metrics

    def close(self):
        self.logger.close()
