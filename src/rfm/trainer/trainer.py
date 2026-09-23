import json
import hashlib
import math
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence
from torch.utils.data import RandomSampler, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import gin
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast

from rfm.artifacts import atomic_torch_save, atomic_write_text

from .logger.logger_base import LoggerBase
from .optimizer_base import OptimizerBase
from .yield_dataset import FEATURE_CACHE_VERSION, YieldDataset
from .utils import infer_metric_direction


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_digest(modules: Iterable[nn.Module]) -> str:
    digest = hashlib.sha256()
    seen: set[int] = set()
    for module_index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            digest.update(f"{module_index}:{name}".encode("utf-8"))
            value = parameter.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


TASK_PRIVATE_MODULE_NAMES = (
    "task_adapters",
    "task_trunks",
    "mu_heads",
    "sigma_heads",
)


def _task_private_state_keys(
    state: Mapping[str, torch.Tensor], tasks: Sequence[str]
) -> Dict[str, List[str]]:
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Task-private state requires unique tasks")
    resolved: Dict[str, List[str]] = {}
    for task in tasks:
        prefixes = tuple(f"{module_name}.{task}." for module_name in TASK_PRIVATE_MODULE_NAMES)
        keys = sorted(name for name in state if name.startswith(prefixes))
        if not keys:
            raise ValueError(f"No task-private state found for {task!r}")
        resolved[task] = keys
    flattened = [name for keys in resolved.values() for name in keys]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Task-private state key sets overlap")
    return resolved


def _state_dict_digest(
    state: Mapping[str, torch.Tensor], keys: Optional[Iterable[str]] = None
) -> str:
    selected = sorted(state if keys is None else keys)
    missing = sorted(set(selected) - set(state))
    if missing:
        raise ValueError(f"State digest keys are missing: {missing[:3]}")
    digest = hashlib.sha256()
    for name in selected:
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _merge_task_private_states(
    base_state: Mapping[str, torch.Tensor],
    task_states: Mapping[str, Mapping[str, torch.Tensor]],
    tasks: Sequence[str],
) -> tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    if set(task_states) != set(tasks):
        raise ValueError("Task-private merge sources are incomplete")
    private_keys = _task_private_state_keys(base_state, tasks)
    expected_keys = set(base_state)
    merged = {
        name: tensor.detach().cpu().clone() for name, tensor in base_state.items()
    }
    for task in tasks:
        source = task_states[task]
        if set(source) != expected_keys:
            raise ValueError(f"Task-private source state keys changed for {task}")
        owned = set(private_keys[task])
        for name, base_tensor in base_state.items():
            source_tensor = source[name]
            if (
                source_tensor.shape != base_tensor.shape
                or source_tensor.dtype != base_tensor.dtype
            ):
                raise ValueError(f"Task-private tensor contract changed for {task}/{name}")
            if name not in owned and not torch.equal(
                source_tensor.detach().cpu(), base_tensor.detach().cpu()
            ):
                raise ValueError(f"Non-target tensor changed for {task}/{name}")
        for name in private_keys[task]:
            merged[name] = source[name].detach().cpu().clone()
    return merged, private_keys


def _resolve_task_loss_weights(
    tasks: Iterable[str],
    task_loss_weights: Optional[Mapping[str, float]],
    *,
    use_pcgrad: bool,
) -> Dict[str, float]:
    resolved_tasks = list(tasks)
    if task_loss_weights is None:
        provided: Dict[str, float] = {}
    elif isinstance(task_loss_weights, Mapping):
        provided = dict(task_loss_weights)
    else:
        raise ValueError("task_loss_weights must be a mapping or None")
    unknown = sorted(set(provided) - set(resolved_tasks))
    if unknown:
        raise ValueError(f"Unknown task_loss_weights tasks: {unknown}")

    resolved = {task: 1.0 for task in resolved_tasks}
    for task, value in provided.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                f"task_loss_weights[{task!r}] must be finite and positive"
            )
        resolved[task] = float(value)
    if use_pcgrad and any(
        not math.isclose(weight, 1.0) for weight in resolved.values()
    ):
        raise ValueError("Non-uniform task_loss_weights do not support PCGrad")
    return resolved


def _uniform_task_soup_state(
    source_states: Sequence[Mapping[str, torch.Tensor]],
    macro_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not source_states:
        raise ValueError("Task-soup averaging requires at least one source state")
    expected_keys = set(macro_state)
    if any(set(state) != expected_keys for state in source_states):
        raise ValueError("Task-soup source state keys do not match macro-best")

    averaged_state: Dict[str, torch.Tensor] = {}
    for name, macro_tensor in macro_state.items():
        source_tensors = [state[name] for state in source_states]
        for tensor in source_tensors:
            if tensor.shape != macro_tensor.shape or tensor.dtype != macro_tensor.dtype:
                raise ValueError(f"Task-soup tensor contract changed for {name}")
        if macro_tensor.is_floating_point():
            accumulator = torch.zeros_like(macro_tensor, dtype=torch.float64)
            for tensor in source_tensors:
                accumulator.add_(tensor.detach().cpu().to(dtype=torch.float64))
            averaged_state[name] = (
                accumulator.div_(len(source_tensors)).to(dtype=macro_tensor.dtype)
            )
        else:
            averaged_state[name] = macro_tensor.detach().cpu().clone()
    return averaged_state


def _select_balanced_checkpoint_record(
    records: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    *,
    macro_tolerance: float = 0.01,
) -> Dict[str, Any]:
    if not records:
        raise ValueError("Balanced checkpoint selection requires validation records")
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Balanced checkpoint selection requires unique tasks")
    if (
        isinstance(macro_tolerance, bool)
        or not isinstance(macro_tolerance, (int, float))
        or not math.isfinite(float(macro_tolerance))
        or float(macro_tolerance) < 0.0
    ):
        raise ValueError("macro_tolerance must be finite and non-negative")

    normalized: List[Dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for index, record in enumerate(records):
        epoch = int(record.get("epoch", -1))
        if epoch < 0 or epoch in seen_epochs:
            raise ValueError("Balanced validation epochs must be unique and non-negative")
        seen_epochs.add(epoch)
        macro_rmse = float(record.get("avg_rmse", float("nan")))
        task_rmse = {
            task: float(record.get(f"{task}_rmse", float("nan"))) for task in tasks
        }
        if (
            not math.isfinite(macro_rmse)
            or macro_rmse < 0.0
            or any(not math.isfinite(value) or value < 0.0 for value in task_rmse.values())
        ):
            raise ValueError("Balanced validation RMSE values must be finite and non-negative")
        normalized.append(
            {
                "record_index": index,
                "epoch": epoch,
                "avg_rmse": macro_rmse,
                "task_rmse": task_rmse,
            }
        )

    macro_best = min(record["avg_rmse"] for record in normalized)
    macro_limit = macro_best * (1.0 + float(macro_tolerance))
    task_best = {
        task: min(record["task_rmse"][task] for record in normalized) for task in tasks
    }
    if any(value <= 0.0 for value in task_best.values()):
        raise ValueError("Balanced task-best RMSE values must be positive")

    eligible: List[Dict[str, Any]] = []
    output_records: List[Dict[str, Any]] = []
    for record in normalized:
        regrets = {
            task: record["task_rmse"][task] / task_best[task] - 1.0 for task in tasks
        }
        enriched = {
            **record,
            "eligible": record["avg_rmse"] <= macro_limit,
            "relative_regret": regrets,
            "maximum_relative_regret": max(regrets.values()),
        }
        output_records.append(enriched)
        if enriched["eligible"]:
            eligible.append(enriched)
    if not eligible:
        raise RuntimeError("Balanced checkpoint candidate set is empty")

    selected = min(
        eligible,
        key=lambda record: (
            record["maximum_relative_regret"],
            record["avg_rmse"],
            record["epoch"],
        ),
    )
    return {
        "macro_tolerance": float(macro_tolerance),
        "macro_best_rmse": macro_best,
        "macro_limit_rmse": macro_limit,
        "task_best_rmse": task_best,
        "eligible_count": len(eligible),
        "selected_record_index": int(selected["record_index"]),
        "selected_epoch": int(selected["epoch"]),
        "selected_macro_rmse": float(selected["avg_rmse"]),
        "selected_maximum_relative_regret": float(
            selected["maximum_relative_regret"]
        ),
        "validation_records": output_records,
    }


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
        test_datasets: Optional[Dict[str, YieldDataset]] = None,
        seed: int = 20260825,
        variant_name: str = "aquatox_multi",
        n_steps_per_epoch: Optional[int] = None,
        device: str = "auto",
        checkpoint_best: bool = False,
        checkpoint_selection: Literal[
            "macro", "taskwise", "task_soup", "balanced", "task_private_finetune"
        ] = "macro",
        best_metric: str = "avg_rmse",
        metric_direction: Literal["auto", "min", "max"] = "auto",
        gradient_clipping_norm: float = 10.0,
        num_workers: int = 1,
        valid_every_n_epochs: int = 3,
        log_train_every_n_batches: int = 10,
        lambda_reg: float = 1e-3,
        patience: int = 10,
        sampling_temperature: float = 2.0,
        molecular_sampling_temperature: float = 2.0,
        task_private_sampling_mode: Literal[
            "weighted_replacement", "uniform_without_replacement"
        ] = "weighted_replacement",
        task_loss_weights: Optional[Mapping[str, float]] = None,
        task_loss_normalization: Literal["none", "ema"] = "none",
        task_loss_ema_beta: float = 0.98,
        task_loss_ema_eps: float = 1e-8,
        task_loss_ema_warmup_epochs: int = 1,
        task_loss_ema_min_weight: float = 0.5,
        task_loss_ema_max_weight: float = 2.0,
        lambda_expert_diversity: float = 0.0,
        lambda_cls: float = 0.1,
        lambda_ord: float = 0.05,
        lambda_mu: float = 0.0,
        huber_beta: float = 1.0,
        ordering_start_fraction: float = 0.0,
        warmup_epochs: int = 5,
        scheduler_mode: Literal["epoch", "step"] = "epoch",
        warmup_fraction: float = 0.05,
        min_lr: float = 1e-6,
        gradient_diagnostics_every_n_epochs: int = 0,
        use_swa: bool = True,
        swa_lr: float = 1e-5,
        use_pcgrad: bool = False,
        pcgrad_scope: Literal["all", "hierarchical_shared"] = "all",
        training_scope: Literal["cv", "full_data"] = "cv",
        staged_finetuning: bool = False,
        finetune_checkpoint_path: str | Path | None = None,
        finetune_checkpoint_sha256: str | None = None,
        finetune_source_variant: str = "pretrained_reference",
        stage1_epochs: int = 8,
        stage1_lr: float = 2e-4,
        stage2_graph_lr: float = 2e-5,
        stage2_fusion_lr: float = 5e-5,
        stage2_sharing_lr: float = 1e-4,
        stage2_task_lr: float = 2e-4,
    ):
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        assert metric_direction in ("auto", "min", "max")
        self.run_dir = Path(run_dir)
        self.tasks = tasks
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets or {}
        self.seed = int(seed)
        self.variant_name = str(variant_name)
        if training_scope not in {"cv", "full_data"}:
            raise ValueError("training_scope must be 'cv' or 'full_data'")
        self.training_scope = training_scope
        if self.training_scope == "cv" and not self.valid_datasets:
            raise ValueError("CV training requires inner-validation datasets")
        if self.training_scope == "full_data" and (
            self.valid_datasets or self.test_datasets
        ):
            raise ValueError("Full-data training must not construct validation or test datasets")

        expected_task_set = set(tasks)
        expected_train_role = "full_train" if self.training_scope == "full_data" else "outer_train"
        for name, datasets, expected_role in (
            ("train", self.train_datasets, expected_train_role),
            ("validation", self.valid_datasets, "inner_val"),
            ("test", self.test_datasets, "outer_test"),
        ):
            if name in {"validation", "test"} and not datasets:
                continue
            if set(datasets) != expected_task_set:
                raise ValueError(
                    f"{name} dataset keys must exactly match tasks; "
                    f"missing={sorted(expected_task_set - set(datasets))}, "
                    f"extra={sorted(set(datasets) - expected_task_set)}"
                )
            for task_name, dataset in datasets.items():
                if dataset.task != task_name:
                    raise ValueError(
                        f"{name} dataset key {task_name!r} does not match dataset.task="
                        f"{dataset.task!r}"
                    )
                if dataset.split_role != expected_role:
                    raise ValueError(
                        f"{name} dataset {task_name!r} must use role {expected_role!r}, "
                        f"got {dataset.split_role!r}"
                    )

        reference_dataset = self.train_datasets[tasks[0]]
        self.reference_dataset = reference_dataset
        self.cv_protocol = reference_dataset.cv_protocol
        self.outer_fold_id = reference_dataset.outer_fold_id
        self.dataset_layer = reference_dataset.dataset_layer
        for datasets in (self.train_datasets, self.valid_datasets, self.test_datasets):
            for dataset in datasets.values():
                contract = (
                    dataset.cv_protocol,
                    dataset.outer_fold_id,
                    dataset.dataset_layer,
                    dataset.dataset_sha256,
                    dataset.manifest_sha256,
                )
                reference_contract = (
                    reference_dataset.cv_protocol,
                    reference_dataset.outer_fold_id,
                    reference_dataset.dataset_layer,
                    reference_dataset.dataset_sha256,
                    reference_dataset.manifest_sha256,
                )
                if contract != reference_contract:
                    raise ValueError(
                        "All train/inner-validation/outer-test datasets must share one "
                        f"frozen CV contract; got {contract} vs {reference_contract}"
                    )

        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics
        if isinstance(num_workers, bool) or int(num_workers) < 0:
            raise ValueError("num_workers must be a non-negative integer")
        self.num_workers = int(num_workers)

        self.logger = logger
        self.optimizer = optimizer
        self.model = model
        self.model_kind = getattr(model, "model_kind", "unknown")
        self.regression_mode = getattr(model, "regression_mode", None)
        if self.regression_mode not in {"heteroscedastic", "deterministic"}:
            raise ValueError(
                "Model must declare regression_mode as heteroscedastic or deterministic"
            )
        if list(getattr(model, "tasks", [])) != list(tasks):
            raise ValueError(
                f"Model tasks {getattr(model, 'tasks', None)!r} do not match trainer tasks {tasks!r}"
            )
        if self.model_kind not in {"mmoe", "grouped", "single_task"}:
            raise ValueError(f"Unsupported model_kind={self.model_kind!r}")
        if self.model_kind == "single_task" and len(tasks) != 1:
            raise ValueError("A single-task model requires exactly one trainer task")
        if isinstance(n_epochs, bool) or int(n_epochs) <= 0:
            raise ValueError("n_epochs must be positive")
        if isinstance(stage1_epochs, bool) or int(stage1_epochs) <= 0:
            raise ValueError("stage1_epochs must be positive")
        self.staged_finetuning = bool(staged_finetuning)
        self.stage1_epochs = int(stage1_epochs) if self.staged_finetuning else 0
        self.stage2_epochs = int(n_epochs) if self.staged_finetuning else 0
        self.n_epochs = int(n_epochs) + self.stage1_epochs
        if self.staged_finetuning and self.model_kind == "single_task":
            raise ValueError("Staged grouped-adapter fine-tuning requires a multi-task model")
        if (
            self.staged_finetuning
            and getattr(getattr(model, "molecular_encoder", None), "graph_encoder", None)
            is None
        ):
            raise ValueError("Staged fine-tuning requires an active graph encoder")
        if self.staged_finetuning and finetune_checkpoint_path is None:
            raise ValueError("staged_finetuning requires finetune_checkpoint_path")
        if self.staged_finetuning and (use_swa or use_pcgrad):
            raise ValueError("Staged fine-tuning does not support SWA or PCGrad")
        self.finetune_checkpoint_path = (
            Path(finetune_checkpoint_path).expanduser().resolve()
            if finetune_checkpoint_path is not None
            else None
        )
        self.finetune_checkpoint_sha256 = finetune_checkpoint_sha256
        self.finetune_source_variant = str(finetune_source_variant)
        stage_lrs = {
            "stage1_upper": stage1_lr,
            "graph_encoder": stage2_graph_lr,
            "feature_fusion": stage2_fusion_lr,
            "sharing": stage2_sharing_lr,
            "task_specific": stage2_task_lr,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in stage_lrs.values()
        ):
            raise ValueError("All staged fine-tuning learning rates must be positive and finite")
        self.stage_lrs = {name: float(value) for name, value in stage_lrs.items()}
        self.train_batch_size = int(train_batch_size)
        self.valid_batch_size = int(valid_batch_size)
        self.n_steps_per_epoch = (
            int(n_steps_per_epoch)
            if n_steps_per_epoch is not None
            else max(
                math.ceil(len(train_datasets[task]) / self.train_batch_size)
                for task in tasks
            )
        )
        if self.n_steps_per_epoch < 1:
            raise ValueError("n_steps_per_epoch must be positive")
        self.total_planned_steps = int(self.n_epochs * self.n_steps_per_epoch)
        self.stage_optimizer_step = 0
        self.stage_total_planned_steps = self.total_planned_steps
        self.current_stage = "standard"
        self.stage_audit: Dict[str, Any] = {}
        self.optimizer_group_audit: Dict[str, Any] = {}
        self.stage_resource_summaries: List[Dict[str, Any]] = []
        self.transfer_audit: Dict[str, Any] | None = None

        self.lambda_reg = lambda_reg
        self.patience = patience
        self.epochs_no_improve = 0
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp_enabled = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.early_stop = False
        if not isinstance(use_pcgrad, bool):
            raise ValueError("use_pcgrad must be a bool")
        if pcgrad_scope not in {"all", "hierarchical_shared"}:
            raise ValueError(
                "pcgrad_scope must be 'all' or 'hierarchical_shared'"
            )
        if use_pcgrad and pcgrad_scope == "hierarchical_shared" and self.model_kind != "grouped":
            raise ValueError(
                "hierarchical_shared PCGrad requires a grouped multi-task model"
            )
        self.use_pcgrad = use_pcgrad
        self.pcgrad_scope = pcgrad_scope
        self.pcgrad_parameter_scope_audit: Dict[str, Any] = {}
        if self.use_pcgrad and self.pcgrad_scope == "hierarchical_shared":
            self.pcgrad_parameter_scope_audit = self._pcgrad_parameter_scopes()[1]

        for metric in self.train_metrics.values():
            metric.to(self.device)

        for task_metrics in self.valid_metrics.values():
            for metric in task_metrics.values():
                metric.to(self.device)

        self.checkpoint_best = checkpoint_best
        if self.training_scope == "full_data" and self.checkpoint_best:
            raise ValueError("Full-data fixed-epoch training cannot select a best checkpoint")
        supported_checkpoint_selections = {
            "macro",
            "taskwise",
            "task_soup",
            "balanced",
            "task_private_finetune",
        }
        if checkpoint_selection not in supported_checkpoint_selections:
            raise ValueError(
                "checkpoint_selection must be 'macro', 'taskwise', 'task_soup', "
                "'balanced', or 'task_private_finetune'"
            )
        external_checkpoint_selections = {
            "taskwise",
            "task_soup",
            "balanced",
            "task_private_finetune",
        }
        if checkpoint_selection in external_checkpoint_selections and not self.checkpoint_best:
            raise ValueError("Advanced checkpoint selection requires checkpoint_best=True")
        if (
            checkpoint_selection in external_checkpoint_selections
            and self.training_scope != "cv"
        ):
            raise ValueError("Advanced checkpoint selection is only valid for CV training")
        if checkpoint_selection in external_checkpoint_selections and self.test_datasets:
            raise ValueError(
                "Advanced CV selection requires the external checkpoint evaluator"
            )
        if (
            checkpoint_selection == "task_soup"
            and self.model_kind not in {"mmoe", "grouped"}
        ):
            raise ValueError("Task-soup selection requires a multi-task model")
        if checkpoint_selection in {"task_soup", "balanced"} and use_swa:
            raise ValueError("Task-soup and balanced selection do not support SWA")
        if checkpoint_selection == "balanced" and (
            best_metric != "avg_rmse" or metric_direction not in {"auto", "min"}
        ):
            raise ValueError("Balanced selection requires minimizing avg_rmse")
        self.checkpoint_selection = checkpoint_selection
        if checkpoint_selection == "task_private_finetune":
            if self.model_kind != "grouped":
                raise ValueError("Task-private fine-tuning requires a grouped model")
            if not bool(getattr(self.model, "use_task_adapters", False)):
                raise ValueError("Task-private fine-tuning requires task adapters")
            if self.staged_finetuning or use_swa or use_pcgrad:
                raise ValueError(
                    "Task-private fine-tuning does not support staged transfer, SWA, or PCGrad"
                )
        elif task_private_sampling_mode != "weighted_replacement":
            raise ValueError(
                "A non-default task-private sampler requires task_private_finetune"
            )
        self.best_metric = best_metric
        self.metric_direction = (
            "min" if best_metric == "avg_rmse" else (
                infer_metric_direction(self.best_metric)
                if metric_direction == "auto"
                else metric_direction
            )
        )
        self.best_valid_metric = float("inf") if self.metric_direction == "min" else float("-inf")
        self.best_valid_metrics: Dict[str, float] = {}
        self.best_task_valid_metrics = {
            task: float("inf") for task in self.tasks
        }
        self.best_task_records: Dict[str, Dict[str, Any]] = {}
        self.best_task_model_states: Dict[str, Dict[str, torch.Tensor]] = {}
        self.best_task_loss_balancing_states: Dict[str, Dict[str, Any]] = {}
        self.best_task_prediction_frames: Dict[str, pd.DataFrame] = {}
        self.balanced_validation_records: List[Dict[str, Any]] = []
        self.balanced_model_states: List[Dict[str, torch.Tensor]] = []
        self.balanced_loss_balancing_states: List[Dict[str, Any]] = []
        self.task_private_finetune_audit: Dict[str, Any] = {}

        self.gradient_clipping_norm = gradient_clipping_norm
        self.valid_every_n_epochs = valid_every_n_epochs
        self.log_train_every_n_batches = log_train_every_n_batches
        if gradient_clipping_norm <= 0:
            raise ValueError("gradient_clipping_norm must be positive")
        if scheduler_mode not in {"epoch", "step"}:
            raise ValueError("scheduler_mode must be 'epoch' or 'step'")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if lambda_mu < 0:
            raise ValueError("lambda_mu must be non-negative")
        if huber_beta <= 0:
            raise ValueError("huber_beta must be positive")
        if not 0.0 <= ordering_start_fraction < 1.0:
            raise ValueError("ordering_start_fraction must be in [0, 1)")
        if gradient_diagnostics_every_n_epochs < 0:
            raise ValueError("gradient_diagnostics_every_n_epochs must be non-negative")
        self.scheduler_mode = scheduler_mode
        self.warmup_fraction = float(warmup_fraction)
        self.min_lr = float(min_lr)
        self.lambda_mu = float(lambda_mu)
        self.huber_beta = float(huber_beta)
        self.ordering_start_fraction = float(ordering_start_fraction)
        self.gradient_diagnostics_every_n_epochs = int(
            gradient_diagnostics_every_n_epochs
        )

        self.warmup_epochs = warmup_epochs
        if self.finetune_checkpoint_path is not None:
            self.transfer_audit = self._load_finetune_checkpoint(
                self.finetune_checkpoint_path,
                expected_sha256=self.finetune_checkpoint_sha256,
            )
        self.model.to(self.device)
        self.base_lrs: List[float] = []
        self.lr_scheduler = None
        self.warmup_steps = 0
        if self.staged_finetuning:
            self._configure_training_stage("stage1")
        else:
            self.optimizer.initialize(model=self.model)
            self._initialize_scheduler(stage_epochs=self.n_epochs)

        self.use_swa = use_swa
        self.swa_start = int(n_epochs * 0.8)
        self.swa_active = False
        if use_swa:
            try:
                from torch.optim.swa_utils import AveragedModel, SWALR
                self.swa_model = AveragedModel(self.model)
                self.swa_scheduler = SWALR(
                    self.optimizer.optimizer,
                    swa_lr=swa_lr,
                    anneal_epochs=max(1, int(n_epochs * 0.05)),
                    anneal_strategy='cos',
                )
                print(f"[SWA] 启用，从 epoch {self.swa_start} 开始，swa_lr={swa_lr}")
            except ImportError:
                print("[SWA] torch.optim.swa_utils 不可用，跳过 SWA")
                self.use_swa = False
                self.swa_model = None
                self.swa_scheduler = None
        else:
            self.swa_model = None
            self.swa_scheduler = None

        self.lambda_ord = lambda_ord
        self.ec_pairs = {
            "fish_EC50":          "fish_EC10",
            "fish_EC10":          "fish_EC50",
            "aquatic_invertebrates_EC50": "aquatic_invertebrates_EC10",
            "aquatic_invertebrates_EC10": "aquatic_invertebrates_EC50",
            "algae_EC50":         "algae_EC10",
            "algae_EC10":         "algae_EC50",
        }
        self.label_scaler_params: Dict[str, tuple] = {}
        for task in tasks:
            scaler = train_datasets[task].label_scaler
            self.label_scaler_params[task] = (
                float(scaler.mean_[0]),
                float(scaler.scale_[0]),
            )
        print(
            f"[Ordinal] loaded {len(self.label_scaler_params)} outer-train label scalers, "
            f"lambda_ord={lambda_ord}"
        )

        dataset_sequence = [
            dataset
            for dataset_dict in (
                self.train_datasets,
                self.valid_datasets,
                self.test_datasets,
            )
            for dataset in dataset_dict.values()
        ]
        shared_graph_cache = {}
        for dataset in dataset_sequence:
            dataset._graph_cache = shared_graph_cache
        for dataset in dataset_sequence:
            dataset.preprocess()
            dataset._preload_graphs()
        print(f"[GraphCache] verified {len(shared_graph_cache)} unique molecular graphs")
        self.feature_preprocessing = self._fit_model_feature_preprocessing(
            shared_graph_cache
        )

        _pin = False
        _persistent = self.num_workers > 0
        _prefetch = 2 if self.num_workers > 0 else None

        canonical_tasks = (
            "fish_EC50",
            "fish_EC10",
            "aquatic_invertebrates_EC50",
            "aquatic_invertebrates_EC10",
            "algae_EC50",
            "algae_EC10",
        )
        self.canonical_tasks = canonical_tasks
        self.molecular_sampling_temperature = float(molecular_sampling_temperature)
        if task_private_sampling_mode not in {
            "weighted_replacement",
            "uniform_without_replacement",
        }:
            raise ValueError(
                "task_private_sampling_mode must be 'weighted_replacement' or "
                "'uniform_without_replacement'"
            )
        self.task_private_sampling_mode = task_private_sampling_mode
        unknown_sampler_tasks = sorted(set(tasks) - set(canonical_tasks))
        if unknown_sampler_tasks:
            raise ValueError(f"No stable sampler seed registered for {unknown_sampler_tasks}")

        def _make_train_loader(ds, task_name):
            weights = ds.get_sample_weights(temperature=molecular_sampling_temperature)
            sampler_generator = torch.Generator()
            sampler_generator.manual_seed(seed + canonical_tasks.index(task_name))
            loader_generator = torch.Generator()
            loader_generator.manual_seed(
                seed + 1000 + canonical_tasks.index(task_name)
            )
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(weights),
                replacement=True,
                generator=sampler_generator,
            )
            print(f"[Trainer] {task_name}: 分子逆频率采样 T={molecular_sampling_temperature}, "
                  f"样本数={len(weights)}")
            return DataLoader(
                ds,
                batch_size=train_batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=ds.collate,
                pin_memory=_pin,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
                generator=loader_generator,
            )

        self.train_loaders = {
            task: _make_train_loader(train_datasets[task], task)
            for task in tasks
        }
        self.valid_loaders = {
            task: DataLoader(
                self.valid_datasets[task],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.valid_datasets[task].collate,
                pin_memory=_pin,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
            )
            for task in tasks
            if task in self.valid_datasets
        }
        self.test_loaders = {
            task: DataLoader(
                self.test_datasets[task],
                batch_size=valid_batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.test_datasets[task].collate,
                pin_memory=_pin,
                persistent_workers=_persistent,
                prefetch_factor=_prefetch,
            )
            for task in tasks
            if task in self.test_datasets
        }

        self.nll_loss_fn = nn.GaussianNLLLoss(eps=1e-6, full=False)
        self.mse_loss_fn = nn.MSELoss()
        self.huber_loss_fn = nn.SmoothL1Loss(beta=self.huber_beta)

        self.cls_loss_fn = nn.NLLLoss()
        self.lambda_cls = lambda_cls

        self._ghs_thresholds = {
            task: [-1.0, 0.0, 1.0] if 'EC10' in task else [0.0, 1.0, 2.0]
            for task in tasks
        }

        self.train_iterators = {
            task: iter(self.train_loaders[task]) for task in self.tasks
        }

        print(
            f"[Trainer] steps_per_epoch={self.n_steps_per_epoch}; "
            f"planned_steps={self.total_planned_steps}; scheduler={self.scheduler_mode}; "
            "each step consumes one batch from every configured task"
        )

        self.task_prob = None

        self.loss_weights = _resolve_task_loss_weights(
            tasks, task_loss_weights, use_pcgrad=self.use_pcgrad
        )
        self.task_loss_weight_sum = float(sum(self.loss_weights.values()))
        if task_loss_normalization not in {"none", "ema"}:
            raise ValueError("task_loss_normalization must be 'none' or 'ema'")
        for name, value in (
            ("task_loss_ema_beta", task_loss_ema_beta),
            ("task_loss_ema_eps", task_loss_ema_eps),
            ("task_loss_ema_min_weight", task_loss_ema_min_weight),
            ("task_loss_ema_max_weight", task_loss_ema_max_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= float(task_loss_ema_beta) < 1.0:
            raise ValueError("task_loss_ema_beta must be in [0, 1)")
        if float(task_loss_ema_eps) <= 0.0:
            raise ValueError("task_loss_ema_eps must be greater than zero")
        if (
            isinstance(task_loss_ema_warmup_epochs, bool)
            or int(task_loss_ema_warmup_epochs) < 0
        ):
            raise ValueError("task_loss_ema_warmup_epochs must be non-negative")
        if not 0.0 < float(task_loss_ema_min_weight) <= float(
            task_loss_ema_max_weight
        ):
            raise ValueError("EMA loss-weight bounds are invalid")
        if task_loss_normalization == "ema":
            if self.model_kind != "mmoe":
                raise ValueError("EMA task-loss normalization requires global MMoE")
            if self.use_pcgrad:
                raise ValueError("EMA task-loss normalization does not support PCGrad")
            if any(
                not math.isclose(weight, 1.0)
                for weight in self.loss_weights.values()
            ):
                raise ValueError(
                    "EMA task-loss normalization requires uniform static task weights"
                )
        if (
            isinstance(lambda_expert_diversity, bool)
            or not isinstance(lambda_expert_diversity, (int, float))
            or not math.isfinite(float(lambda_expert_diversity))
            or float(lambda_expert_diversity) < 0.0
        ):
            raise ValueError("lambda_expert_diversity must be finite and non-negative")
        if float(lambda_expert_diversity) > 0.0:
            if self.model_kind != "mmoe":
                raise ValueError("Expert diversity regularization requires global MMoE")
            if self.use_pcgrad:
                raise ValueError("Expert diversity regularization does not support PCGrad")

        self.task_loss_normalization = task_loss_normalization
        self.task_loss_ema_beta = float(task_loss_ema_beta)
        self.task_loss_ema_eps = float(task_loss_ema_eps)
        self.task_loss_ema_warmup_epochs = int(task_loss_ema_warmup_epochs)
        self.task_loss_ema_min_weight = float(task_loss_ema_min_weight)
        self.task_loss_ema_max_weight = float(task_loss_ema_max_weight)
        self.task_loss_ema_values = {task: 0.0 for task in self.tasks}
        self.task_loss_ema_updates = 0
        self.current_task_loss_weights = dict(self.loss_weights)
        self.lambda_expert_diversity = float(lambda_expert_diversity)
        if self.checkpoint_selection == "task_private_finetune":
            if self.task_loss_normalization != "none":
                raise ValueError(
                    "Task-private fine-tuning requires fixed task-loss scaling"
                )
            if any(not math.isclose(weight, 1.0) for weight in self.loss_weights.values()):
                raise ValueError(
                    "Task-private fine-tuning requires equal Stage-A task weights"
                )
            if not math.isclose(self.lambda_expert_diversity, 0.0):
                raise ValueError(
                    "Task-private fine-tuning does not support expert diversity regularization"
                )
        num_experts = int(getattr(getattr(self.model, "mmoe", None), "num_experts", 0))
        self.expert_pair_names = [
            f"expert_pair_cosine_{left}_{right}"
            for left in range(num_experts)
            for right in range(left + 1, num_experts)
        ]

    def _load_finetune_checkpoint(
        self,
        path: Path,
        *,
        expected_sha256: str | None,
    ) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = _sha256_file(path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256:
            raise ValueError(
                "Fine-tuning source checkpoint SHA-256 mismatch: "
                f"expected={expected_sha256}, actual={actual_sha256}"
            )
        checkpoint = torch.load(path, map_location="cpu")
        required = {
            "schema_version",
            "model_class",
            "model_spec",
            "model",
            "variant_name",
            "model_kind",
            "regression_mode",
            "tasks",
            "seed",
            "training_spec",
            "data_contract",
        }
        missing_metadata = sorted(required - set(checkpoint))
        if missing_metadata:
            raise ValueError(
                f"Fine-tuning source checkpoint is missing metadata: {missing_metadata}"
            )
        expected_metadata = {
            "schema_version": 1,
            "model_class": "YieldGNN",
            "variant_name": self.finetune_source_variant,
            "model_kind": "mmoe",
            "regression_mode": "heteroscedastic",
            "tasks": list(self.tasks),
            "seed": self.seed,
        }
        for name, expected in expected_metadata.items():
            if checkpoint.get(name) != expected:
                raise ValueError(
                    f"Fine-tuning source {name} mismatch: "
                    f"expected={expected!r}, actual={checkpoint.get(name)!r}"
                )

        contract = checkpoint["data_contract"]
        expected_contract = {
            "dataset_layer": self.dataset_layer,
            "cv_protocol": self.cv_protocol,
            "outer_fold_id": self.outer_fold_id,
            "dataset_sha256": self.reference_dataset.dataset_sha256,
            "manifest_sha256": self.reference_dataset.manifest_sha256,
        }
        for name, expected in expected_contract.items():
            if contract.get(name) != expected:
                raise ValueError(
                    f"Fine-tuning source data_contract.{name} mismatch: "
                    f"expected={expected!r}, actual={contract.get(name)!r}"
                )

        expected_training = {
            "lambda_cls": 0.1,
            "lambda_ord": 0.05,
            "lambda_mu": 0.5,
            "huber_beta": 1.0,
            "ordering_start_fraction": 0.0,
            "task_loss_reduction": "mean",
            "use_pcgrad": False,
            "use_swa": False,
        }
        training_spec = checkpoint["training_spec"]
        for name, expected in expected_training.items():
            actual = training_spec.get(name)
            if isinstance(expected, float):
                valid = isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                valid = actual == expected
            if not valid:
                raise ValueError(
                    f"Fine-tuning source training_spec.{name} mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )

        source_spec = checkpoint["model_spec"]
        target_spec = getattr(self.model, "model_spec", {})
        compatible_fields = (
            "num_effects",
            "hidden_dim",
            "num_layers",
            "num_attention_heads",
            "edge_in_dim",
            "node_in_dim",
            "fp_dim",
            "use_attention",
            "tasks",
            "expert_dim",
            "expert_hidden_dim",
            "mmoe_dropout",
            "gate_temperature",
            "dropout",
            "concat_type",
            "mlp_dropout",
            "attention_dropout",
            "gnn_type",
            "constant_effect_tasks",
            "regression_mode",
            "sigma_parameterization",
            "min_log_sigma",
            "max_log_sigma",
            "fusion_mode",
            "modality_dropout",
            "jk_mode",
            "pooling_mode",
            "message_dropout",
            "train_eps",
            "molecular_modalities",
        )
        legacy_source_defaults = {
            "expert_hidden_dim": source_spec.get("expert_dim"),
            "mmoe_dropout": source_spec.get("dropout"),
            "gate_temperature": 1.0,
            "molecular_modalities": [
                "graph",
                "ecfp",
                "maccs",
                "descriptors",
            ],
        }
        for name in compatible_fields:
            source_value = source_spec.get(name, legacy_source_defaults.get(name))
            if source_value != target_spec.get(name):
                raise ValueError(
                    f"Fine-tuning source model_spec.{name} mismatch: "
                    f"expected target={target_spec.get(name)!r}, "
                    f"actual source={source_value!r}"
                )

        source_training_scope = contract.get("training_scope", "cv")
        if source_training_scope != self.training_scope:
            raise ValueError(
                "Fine-tuning source training scope mismatch: "
                f"expected={self.training_scope!r}, actual={source_training_scope!r}"
            )

        source_state = checkpoint["model"]
        target_state = self.model.state_dict()
        source_keys = set(source_state)
        target_keys = set(target_state)
        shared_keys = source_keys & target_keys
        shape_mismatches = sorted(
            key
            for key in shared_keys
            if tuple(source_state[key].shape) != tuple(target_state[key].shape)
        )
        if shape_mismatches:
            raise ValueError(
                f"Fine-tuning source contains shape mismatches: {shape_mismatches}"
            )

        missing_keys = sorted(target_keys - source_keys)
        discarded_keys = sorted(source_keys - target_keys)
        allowed_missing_prefixes = []
        if getattr(self.model, "sharing_mode", "global_mmoe") == "grouped":
            allowed_missing_prefixes.append("grouped_sharing.")
        if getattr(self.model, "use_task_adapters", False):
            allowed_missing_prefixes.append("task_adapters.")
        if getattr(self.model, "condition_fusion_sharing", "shared") != "shared":
            allowed_missing_prefixes.append("condition_fusions.")
        if getattr(self.model, "molecular_fusion_sharing", "shared") != "shared":
            allowed_missing_prefixes.append("molecular_fusions.")
        if getattr(self.model, "graph_message_sharing", "shared") != "shared":
            allowed_missing_prefixes.append(
                "molecular_encoder.graph_encoder.routed_tails."
            )
        if getattr(self.model, "use_prefusion_task_adapters", False):
            allowed_missing_prefixes.append("prefusion_task_adapters.")
        invalid_missing = [
            key
            for key in missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        allowed_discarded_prefixes = (
            ["mmoe."]
            if getattr(self.model, "sharing_mode", "global_mmoe") == "grouped"
            else []
        )
        invalid_discarded = [
            key
            for key in discarded_keys
            if not any(key.startswith(prefix) for prefix in allowed_discarded_prefixes)
        ]
        if invalid_missing or invalid_discarded:
            raise ValueError(
                "Fine-tuning source key whitelist violation: "
                f"missing={invalid_missing}, discarded={invalid_discarded}"
            )

        result = self.model.load_state_dict(
            {key: source_state[key] for key in shared_keys}, strict=False
        )
        if sorted(result.missing_keys) != missing_keys or result.unexpected_keys:
            raise AssertionError(
                "Fine-tuning state load result differs from the audited key sets"
            )
        required_prefixes = (
            "molecular_encoder.",
            "feature_fusion.",
            "task_trunks.",
            "mu_heads.",
            "sigma_heads.",
        )
        for prefix in required_prefixes:
            if not any(key.startswith(prefix) for key in shared_keys):
                raise ValueError(f"Fine-tuning source did not load required prefix {prefix!r}")
        if getattr(self.model, "sharing_mode", "global_mmoe") == "global_mmoe" and not any(
            key.startswith("mmoe.") for key in shared_keys
        ):
            raise ValueError("Global-MMoE fine-tuning did not load the source MMoE")

        return {
            "schema_version": 1,
            "source_checkpoint": str(path),
            "source_checkpoint_sha256": actual_sha256,
            "source_variant": checkpoint["variant_name"],
            "source_seed": checkpoint["seed"],
            "source_outer_fold_id": contract["outer_fold_id"],
            "loaded_keys": sorted(shared_keys),
            "missing_target_keys": missing_keys,
            "discarded_source_keys": discarded_keys,
            "status": "passed",
        }

    def _encoder_modules(self) -> List[nn.Module]:
        modules = [self.model.molecular_encoder, self.model.feature_fusion]
        condition_fusions = getattr(self.model, "condition_fusions", None)
        if isinstance(condition_fusions, nn.Module):
            modules.append(condition_fusions)
        molecular_fusions = getattr(self.model, "molecular_fusions", None)
        if isinstance(molecular_fusions, nn.Module):
            modules.append(molecular_fusions)
        return modules

    def _upper_modules(self) -> List[nn.Module]:
        modules: List[nn.Module] = []
        for name in (
            "mmoe",
            "grouped_sharing",
            "prefusion_task_adapters",
            "task_adapters",
            "task_trunks",
            "mu_heads",
            "sigma_heads",
        ):
            module = getattr(self.model, name, None)
            if isinstance(module, nn.Module):
                modules.append(module)
        return modules

    @staticmethod
    def _parameter_ids(modules: Iterable[nn.Module]) -> set[int]:
        return {
            id(parameter)
            for module in modules
            for parameter in module.parameters()
        }

    def _stage2_parameter_groups(self) -> List[Dict[str, Any]]:
        graph_module = self.model.molecular_encoder.graph_encoder
        graph_ids = self._parameter_ids([graph_module])
        molecular_ids = self._parameter_ids([self.model.molecular_encoder])
        fusion_ids = (molecular_ids - graph_ids) | self._parameter_ids(
            [self.model.feature_fusion]
        )
        condition_fusions = getattr(self.model, "condition_fusions", None)
        if isinstance(condition_fusions, nn.Module):
            fusion_ids |= self._parameter_ids([condition_fusions])
        molecular_fusions = getattr(self.model, "molecular_fusions", None)
        if isinstance(molecular_fusions, nn.Module):
            fusion_ids |= self._parameter_ids([molecular_fusions])
        sharing_modules = [
            module
            for module in (
                getattr(self.model, "mmoe", None),
                getattr(self.model, "grouped_sharing", None),
            )
            if isinstance(module, nn.Module)
        ]
        sharing_ids = self._parameter_ids(sharing_modules)
        task_modules = [
            module
            for module in (
                getattr(self.model, "task_adapters", None),
                getattr(self.model, "prefusion_task_adapters", None),
                getattr(self.model, "task_trunks", None),
                getattr(self.model, "mu_heads", None),
                getattr(self.model, "sigma_heads", None),
            )
            if isinstance(module, nn.Module)
        ]
        task_ids = self._parameter_ids(task_modules)
        assignments = {
            "graph_encoder": graph_ids,
            "feature_fusion": fusion_ids,
            "sharing": sharing_ids,
            "task_specific": task_ids,
        }
        groups: List[Dict[str, Any]] = []
        for group_name, parameter_ids in assignments.items():
            parameters = [
                parameter
                for parameter in self.model.parameters()
                if id(parameter) in parameter_ids
            ]
            if not parameters:
                raise ValueError(f"Staged optimizer group {group_name!r} is empty")
            groups.append(
                {
                    "params": parameters,
                    "lr": self.stage_lrs[group_name],
                    "name": group_name,
                }
            )
        self._validate_parameter_groups(groups)
        return groups

    def _validate_parameter_groups(self, groups: List[Dict[str, Any]]) -> None:
        trainable = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        occurrences: Dict[int, int] = {}
        for group in groups:
            for parameter in group["params"]:
                occurrences[id(parameter)] = occurrences.get(id(parameter), 0) + 1
        missing = sorted(trainable[identifier] for identifier in set(trainable) - set(occurrences))
        duplicate = sorted(
            trainable[identifier]
            for identifier, count in occurrences.items()
            if identifier in trainable and count != 1
        )
        extra = sorted(
            name
            for identifier, name in {
                id(parameter): name for name, parameter in self.model.named_parameters()
            }.items()
            if identifier in occurrences and identifier not in trainable
        )
        if missing or duplicate or extra:
            raise ValueError(
                "Optimizer parameter groups must cover each trainable parameter exactly once: "
                f"missing={missing}, duplicate={duplicate}, extra={extra}"
            )

    def _initialize_scheduler(self, *, stage_epochs: int) -> None:
        self.base_lrs = [
            float(group["lr"]) for group in self.optimizer.optimizer.param_groups
        ]
        self.stage_total_planned_steps = int(stage_epochs * self.n_steps_per_epoch)
        self.warmup_steps = (
            max(1, int(round(self.stage_total_planned_steps * self.warmup_fraction)))
            if self.scheduler_mode == "step" and self.warmup_fraction > 0
            else 0
        )
        self.lr_scheduler = None
        if self.scheduler_mode == "epoch":
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer.optimizer,
                T_max=max(1, stage_epochs - self.warmup_epochs),
                eta_min=self.min_lr,
            )
        if (self.scheduler_mode == "epoch" and self.warmup_epochs > 0) or (
            self.scheduler_mode == "step" and self.warmup_steps > 0
        ):
            for param_group, base_lr in zip(
                self.optimizer.optimizer.param_groups, self.base_lrs
            ):
                param_group["lr"] = base_lr / 1000.0

    def _configure_training_stage(self, stage: Literal["stage1", "stage2"]) -> None:
        if not self.staged_finetuning:
            raise RuntimeError("Training stages are only available in staged_finetuning mode")
        if stage == "stage1":
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            for module in self._upper_modules():
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            upper_parameters = [
                parameter for parameter in self.model.parameters() if parameter.requires_grad
            ]
            groups = [
                {
                    "params": upper_parameters,
                    "lr": self.stage_lrs["stage1_upper"],
                    "name": "stage1_upper",
                }
            ]
            self._validate_parameter_groups(groups)
            stage_epochs = self.stage1_epochs
            self.stage_audit["encoder_before_stage1_sha256"] = _parameter_digest(
                self._encoder_modules()
            )
        else:
            stage1_after = _parameter_digest(self._encoder_modules())
            stage1_before = self.stage_audit.get("encoder_before_stage1_sha256")
            self.stage_audit["encoder_after_stage1_sha256"] = stage1_after
            self.stage_audit["stage1_encoder_unchanged"] = stage1_before == stage1_after
            if stage1_before != stage1_after:
                raise AssertionError("Encoder parameters changed during Stage 1")
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            groups = self._stage2_parameter_groups()
            stage_epochs = self.stage2_epochs
            self.stage_audit["encoder_before_stage2_sha256"] = _parameter_digest(
                self._encoder_modules()
            )
            self.epochs_no_improve = 0
            self.early_stop = False
            self.best_valid_metric = (
                float("inf") if self.metric_direction == "min" else float("-inf")
            )
            self.best_valid_metrics = {}

        self.optimizer.initialize(parameter_groups=groups)
        self.current_stage = stage
        self.stage_optimizer_step = 0
        self._initialize_scheduler(stage_epochs=stage_epochs)
        audit_rows = []
        parameter_names = {
            id(parameter): name for name, parameter in self.model.named_parameters()
        }
        for group in groups:
            names = [parameter_names[id(parameter)] for parameter in group["params"]]
            audit_rows.append(
                {
                    "name": group["name"],
                    "lr": self.base_lrs[len(audit_rows)],
                    "parameter_tensors": len(names),
                    "parameters": int(sum(parameter.numel() for parameter in group["params"])),
                    "parameter_names": names,
                }
            )
        self.optimizer_group_audit[stage] = audit_rows

    def _finalize_stage_audit(self) -> None:
        if not self.staged_finetuning or self.current_stage != "stage2":
            return
        stage2_after = _parameter_digest(self._encoder_modules())
        stage2_before = self.stage_audit.get("encoder_before_stage2_sha256")
        self.stage_audit["encoder_after_stage2_sha256"] = stage2_after
        self.stage_audit["stage2_encoder_updated"] = stage2_before != stage2_after
        if stage2_before == stage2_after:
            raise AssertionError("Encoder parameters did not update during Stage 2")

    def _fit_model_feature_preprocessing(
        self,
        shared_graph_cache: Dict[str, object],
    ) -> Dict[str, object]:
        molecular_encoder = getattr(self.model, "molecular_encoder", None)
        if not getattr(molecular_encoder, "requires_descriptor_scaler", False):
            return {"fusion_mode": "direct"}

        outer_train_smiles = sorted(
            {
                smiles
                for dataset in self.train_datasets.values()
                for smiles in dataset.smiles_list
            }
        )
        if not outer_train_smiles:
            raise ValueError("Cannot fit descriptor scaler without outer-train molecules")
        descriptor_rows = []
        for smiles in outer_train_smiles:
            graph = shared_graph_cache.get(smiles)
            if graph is None:
                raise KeyError(f"Outer-train graph is missing from the shared cache: {smiles}")
            fp = graph.ndata["fp"] if "fp" in graph.ndata else None
            if fp is None or fp.ndim != 2 or fp.shape[1] != 1214:
                raise ValueError(f"Invalid molecular feature tensor for {smiles!r}")
            descriptor_rows.append(fp[0, -23:].float())
        descriptors = torch.stack(descriptor_rows, dim=0)
        mean = descriptors.mean(dim=0)
        scale = descriptors.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        molecular_encoder.set_descriptor_scaler(mean, scale)
        metadata = {
            "fusion_mode": "projected",
            "descriptor_columns": list(range(1191, 1214)),
            "fit_role": (
                "full_train"
                if getattr(self, "training_scope", "cv") == "full_data"
                else "outer_train"
            ),
            "fit_unit": "unique_standardized_smiles",
            "n_unique_molecules": len(outer_train_smiles),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        }
        print(
            "[FeatureScaler] fitted 23 RDKit descriptors on "
            f"{len(outer_train_smiles)} unique outer-train molecules"
        )
        return metadata

    def _ghs_soft_logprobs(self, mu: torch.Tensor, task: str) -> torch.Tensor:
        if task not in self.label_scaler_params:
            return torch.full((mu.shape[0], 4), -torch.log(torch.tensor(4.0)),
                              device=mu.device)

        mean_t, scale_t = self.label_scaler_params[task]
        mu_log10 = mu.float() * scale_t + mean_t

        t = self._ghs_thresholds[task]
        T = 0.5

        p0 = torch.sigmoid(-(mu_log10 - t[0]) / T)
        p1 = torch.sigmoid(-(mu_log10 - t[1]) / T)
        p2 = torch.sigmoid(-(mu_log10 - t[2]) / T)

        p_C1 = p0
        p_C2 = (p1 - p0).clamp(min=0)
        p_C3 = (p2 - p1).clamp(min=0)
        p_NC = (1.0 - p2).clamp(min=0)

        probs = torch.stack([p_C1, p_C2, p_C3, p_NC], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.log(probs.clamp(min=1e-8))

    @staticmethod
    def _pcgrad_project(task_grads: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(task_grads)
        if n == 0:
            return {}

        ref_tensors: Dict[str, torch.Tensor] = {}
        for grads in task_grads:
            for k, v in grads.items():
                if k not in ref_tensors:
                    ref_tensors[k] = v
        param_names = list(ref_tensors.keys())

        def make_flat(grads_i: Dict[str, torch.Tensor]) -> torch.Tensor:
            parts = []
            for k in param_names:
                if k in grads_i:
                    parts.append(grads_i[k].float().flatten())
                else:
                    ref = ref_tensors[k]
                    parts.append(torch.zeros(ref.numel(), dtype=torch.float32, device=ref.device))
            return torch.cat(parts)

        flat = [make_flat(task_grads[i]) for i in range(n)]

        projected = [g.clone() for g in flat]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dot = torch.dot(projected[i], flat[j])
                if dot < 0:
                    norm_sq = flat[j].dot(flat[j])
                    if norm_sq > 1e-12:
                        projected[i] = projected[i] - (dot / norm_sq) * flat[j]

        combined = sum(projected) / n

        result, offset = {}, 0
        for k in param_names:
            ref = ref_tensors[k]
            size = ref.numel()
            result[k] = combined[offset:offset + size].reshape(ref.shape).to(ref.dtype)
            offset += size
        return result

    def _pcgrad_parameter_scopes(
        self,
    ) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
        grouped = getattr(self.model, "grouped_sharing", None)
        if grouped is None:
            raise ValueError(
                "hierarchical_shared PCGrad requires model.grouped_sharing"
            )

        named_parameters = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        scopes: Dict[str, List[str]] = {
            "global_shared": [],
            **{f"group:{group}": [] for group in grouped.blocks},
            "task_private": [],
        }
        fusion_sharing = getattr(self.model, "condition_fusion_sharing", "shared")
        molecular_fusion_sharing = getattr(
            self.model, "molecular_fusion_sharing", "shared"
        )
        graph_message_sharing = getattr(
            self.model, "graph_message_sharing", "shared"
        )

        def parameter_ids(module: nn.Module | None) -> set[int]:
            if module is None:
                return set()
            return {id(parameter) for parameter in module.parameters()}

        graph_encoder = self.model.molecular_encoder.graph_encoder
        graph_parameter_ids = parameter_ids(graph_encoder)
        if graph_message_sharing == "shared":
            global_parameter_ids = set(graph_parameter_ids)
        else:
            if graph_encoder is None:
                raise ValueError("Routed graph sharing requires an active graph encoder")
            global_parameter_ids = {
                id(parameter) for parameter in graph_encoder.shared_parameters()
            }
        group_parameter_ids = {
            group: parameter_ids(block) for group, block in grouped.blocks.items()
        }

        if graph_message_sharing == "group":
            routed_tail_ids: set[int] = set()
            for route, tail in graph_encoder.routed_tails.items():
                tail_ids = parameter_ids(tail)
                if route in group_parameter_ids:
                    group_parameter_ids[route].update(tail_ids)
                routed_tail_ids.update(tail_ids)
            primary_route = self.model.primary_graph_message_route
            if primary_route in group_parameter_ids:
                primary_tail_ids = (
                    graph_parameter_ids - global_parameter_ids - routed_tail_ids
                )
                group_parameter_ids[primary_route].update(primary_tail_ids)

        molecular_encoder = self.model.molecular_encoder
        if self.model.primary_molecular_fusion_route == "shared":
            global_parameter_ids.update(
                id(parameter)
                for parameter in molecular_encoder.parameters()
                if id(parameter) not in graph_parameter_ids
            )
        if self.model.primary_condition_fusion_route == "shared":
            global_parameter_ids.update(parameter_ids(self.model.feature_fusion))
        if fusion_sharing == "group":
            primary_route = self.model.primary_condition_fusion_route
            if primary_route in group_parameter_ids:
                group_parameter_ids[primary_route].update(
                    parameter_ids(self.model.feature_fusion)
                )
            for route, fusion in self.model.condition_fusions.items():
                if route in group_parameter_ids:
                    group_parameter_ids[route].update(parameter_ids(fusion))
        if molecular_fusion_sharing == "group":
            primary_route = self.model.primary_molecular_fusion_route
            if primary_route in group_parameter_ids:
                group_parameter_ids[primary_route].update(
                    parameter_ids(self.model.molecular_encoder.fp_proj)
                )
            for route, fusion in self.model.molecular_fusions.items():
                if route in group_parameter_ids:
                    group_parameter_ids[route].update(parameter_ids(fusion))

        overlapping_groups = {
            (left, right): group_parameter_ids[left] & group_parameter_ids[right]
            for index, left in enumerate(group_parameter_ids)
            for right in list(group_parameter_ids)[index + 1 :]
            if group_parameter_ids[left] & group_parameter_ids[right]
        }
        if global_parameter_ids & set().union(*group_parameter_ids.values()):
            raise AssertionError("Global and group PCGrad parameter scopes overlap")
        if overlapping_groups:
            raise AssertionError("Organism-group PCGrad parameter scopes overlap")

        for name, parameter in named_parameters.items():
            parameter_id = id(parameter)
            if parameter_id in global_parameter_ids:
                scopes["global_shared"].append(name)
                continue
            matches = [
                group
                for group, identifiers in group_parameter_ids.items()
                if parameter_id in identifiers
            ]
            if len(matches) > 1:
                raise AssertionError(f"PCGrad scope overlap for parameter {name!r}")
            if matches:
                scopes[f"group:{matches[0]}"].append(name)
            else:
                scopes["task_private"].append(name)

        empty = sorted(scope for scope, names in scopes.items() if not names)
        if empty:
            raise ValueError(f"Hierarchical PCGrad has empty parameter scopes: {empty}")
        assigned = [name for names in scopes.values() for name in names]
        if len(assigned) != len(set(assigned)) or set(assigned) != set(named_parameters):
            raise AssertionError("Hierarchical PCGrad parameter partition is not exact")

        audit: Dict[str, Any] = {
            "status": "passed",
            "scope": "hierarchical_shared",
            "total_trainable_parameter_tensors": len(named_parameters),
            "total_trainable_parameters": int(
                sum(parameter.numel() for parameter in named_parameters.values())
            ),
            "scopes": {},
        }
        for scope, names in scopes.items():
            if scope == "global_shared":
                scope_tasks = list(self.tasks)
            elif scope.startswith("group:"):
                group = scope.split(":", 1)[1]
                scope_tasks = [
                    task
                    for task in self.tasks
                    if grouped.task_to_group[task] == group
                ]
            else:
                scope_tasks = list(self.tasks)
            audit["scopes"][scope] = {
                "parameter_tensors": len(names),
                "parameters": int(sum(named_parameters[name].numel() for name in names)),
                "tasks": scope_tasks,
            }
        return scopes, audit

    @staticmethod
    def _gradient_vector_statistics(
        vectors: List[torch.Tensor],
    ) -> Dict[str, float]:
        cosines: List[float] = []
        conflicts = 0
        comparable = 0
        for left_index, left in enumerate(vectors):
            left_norm = torch.linalg.vector_norm(left)
            for right in vectors[left_index + 1:]:
                denominator = left_norm * torch.linalg.vector_norm(right)
                if float(denominator.detach().cpu()) <= 0.0:
                    continue
                dot = torch.dot(left, right)
                cosine = float((dot / denominator).detach().cpu())
                if math.isfinite(cosine):
                    cosines.append(cosine)
                    comparable += 1
                    conflicts += int(float(dot.detach().cpu()) < 0.0)
        return {
            "cosine_mean": float(np.mean(cosines)) if cosines else float("nan"),
            "conflict_fraction": (
                float(conflicts / comparable) if comparable else float("nan")
            ),
            "comparable_pairs": float(comparable),
        }

    @classmethod
    def _project_pcgrad_scope(
        cls,
        *,
        task_grads: Mapping[str, Dict[str, torch.Tensor]],
        parameter_refs: Mapping[str, torch.nn.Parameter],
        parameter_names: List[str],
        tasks: List[str],
        total_task_count: int,
        projection_seed: int,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        if len(tasks) < 2:
            raise ValueError("A projected PCGrad scope requires at least two tasks")
        if total_task_count < len(tasks):
            raise ValueError("total_task_count cannot be smaller than scope tasks")

        def flatten(task: str) -> torch.Tensor:
            parts = []
            for name in parameter_names:
                gradient = task_grads.get(task, {}).get(name)
                if gradient is None:
                    reference = parameter_refs[name]
                    parts.append(
                        torch.zeros(
                            reference.numel(),
                            dtype=torch.float32,
                            device=reference.device,
                        )
                    )
                else:
                    parts.append(gradient.detach().float().reshape(-1))
            return torch.cat(parts)

        raw = [flatten(task) for task in tasks]
        projected = [gradient.clone() for gradient in raw]
        for left_index in range(len(tasks)):
            other_indices = [
                index for index in range(len(tasks)) if index != left_index
            ]
            random.Random(
                int(projection_seed) + 104729 * (left_index + 1)
            ).shuffle(other_indices)
            for right_index in other_indices:
                dot = torch.dot(projected[left_index], raw[right_index])
                if float(dot.detach().cpu()) >= 0.0:
                    continue
                norm_squared = torch.dot(raw[right_index], raw[right_index])
                if float(norm_squared.detach().cpu()) > 1e-12:
                    projected[left_index] = projected[left_index] - (
                        dot / norm_squared
                    ) * raw[right_index]

        combined = sum(projected) / int(total_task_count)
        result: Dict[str, torch.Tensor] = {}
        offset = 0
        for name in parameter_names:
            reference = parameter_refs[name]
            size = reference.numel()
            result[name] = combined[offset:offset + size].reshape(
                reference.shape
            ).to(reference.dtype)
            offset += size
        if offset != combined.numel():
            raise AssertionError("PCGrad scope reconstruction consumed the wrong size")

        before = cls._gradient_vector_statistics(raw)
        after = cls._gradient_vector_statistics(projected)
        diagnostics = {
            "cosine_mean_pre": before["cosine_mean"],
            "cosine_mean_post": after["cosine_mean"],
            "conflict_fraction_pre": before["conflict_fraction"],
            "conflict_fraction_post": after["conflict_fraction"],
            "comparable_pairs": before["comparable_pairs"],
        }
        return result, diagnostics

    def _hierarchical_pcgrad_project(
        self,
        task_grads: Mapping[str, Dict[str, torch.Tensor]],
        *,
        optimizer_step: int,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        if set(task_grads) != set(self.tasks):
            raise ValueError("Hierarchical PCGrad requires one gradient map per task")
        scopes, _audit = self._pcgrad_parameter_scopes()
        parameter_refs = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        grouped = self.model.grouped_sharing
        total_task_count = len(self.tasks)
        projection_seed = self.seed + 1_000_003 * int(optimizer_step)
        combined: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, float] = {}

        projected_scope_tasks = {
            "global_shared": list(self.tasks),
            **{
                f"group:{group}": [
                    task
                    for task in self.tasks
                    if grouped.task_to_group[task] == group
                ]
                for group in grouped.blocks
            },
        }
        for scope_index, (scope, scope_tasks) in enumerate(
            projected_scope_tasks.items()
        ):
            scope_result, scope_diagnostics = self._project_pcgrad_scope(
                task_grads=task_grads,
                parameter_refs=parameter_refs,
                parameter_names=scopes[scope],
                tasks=scope_tasks,
                total_task_count=total_task_count,
                projection_seed=projection_seed + 8191 * scope_index,
            )
            overlap = set(combined).intersection(scope_result)
            if overlap:
                raise AssertionError(f"PCGrad projected duplicate parameters: {overlap}")
            combined.update(scope_result)
            label = scope.replace(":", "_")
            diagnostics.update(
                {
                    f"pcgrad_{label}_{name}": value
                    for name, value in scope_diagnostics.items()
                }
            )

        for name in scopes["task_private"]:
            reference = parameter_refs[name]
            values = [
                task_grads[task].get(name)
                for task in self.tasks
                if task_grads[task].get(name) is not None
            ]
            combined[name] = (
                sum(value.detach() for value in values) / total_task_count
                if values
                else torch.zeros_like(reference)
            )

        if set(combined) != set(parameter_refs):
            missing = sorted(set(parameter_refs) - set(combined))
            extra = sorted(set(combined) - set(parameter_refs))
            raise AssertionError(
                f"Hierarchical PCGrad output mismatch: missing={missing}, extra={extra}"
            )
        return combined, diagnostics

    def _pcgrad_optimizer_step(
        self,
        projected_grads: Mapping[str, torch.Tensor],
    ) -> tuple[float, float, float]:
        self.optimizer.zero_grad()
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad and name in projected_grads:
                parameter.grad = projected_grads[name]
        self.scaler.unscale_(self.optimizer.optimizer)
        grad_pre_tensor = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clipping_norm
        )
        grad_pre = float(grad_pre_tensor.detach().cpu())
        grad_post = (
            min(grad_pre, self.gradient_clipping_norm)
            if np.isfinite(grad_pre)
            else grad_pre
        )
        clipped = float(
            np.isfinite(grad_pre) and grad_pre > self.gradient_clipping_norm
        )
        self.scaler.step(self.optimizer.optimizer)
        self.scaler.update()
        return grad_pre, grad_post, clipped

    def l2_regularization(self) -> torch.Tensor:
        return self.lambda_reg * sum(p.norm(2).pow(2) for p in self.model.parameters() if p.requires_grad)

    def _regression_loss(
        self,
        task_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.regression_mode == "heteroscedastic":
            if "log_sigma" not in task_output:
                raise KeyError("Heteroscedastic model output is missing log_sigma")
            mu = task_output["mu"].float()
            target = labels.float()
            variance = torch.exp(2.0 * task_output["log_sigma"].float())
            return self.nll_loss_fn(mu, target, variance)
        if "log_sigma" in task_output:
            raise ValueError("Deterministic model must not emit log_sigma")
        return 0.5 * self.mse_loss_fn(task_output["mu"].float(), labels.float())

    def _mean_auxiliary_loss(
        self,
        task_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.huber_loss_fn(task_output["mu"].float(), labels.float())

    def _loss_balancing_state(self) -> Dict[str, Any]:
        return {
            "mode": self.task_loss_normalization,
            "ema_values": dict(self.task_loss_ema_values),
            "effective_weights": dict(self.current_task_loss_weights),
            "updates": int(self.task_loss_ema_updates),
            "beta": self.task_loss_ema_beta,
            "eps": self.task_loss_ema_eps,
            "warmup_epochs": self.task_loss_ema_warmup_epochs,
            "min_weight": self.task_loss_ema_min_weight,
            "max_weight": self.task_loss_ema_max_weight,
        }

    def _task_loss_weights_for_step(
        self,
        task_objectives: Mapping[str, torch.Tensor],
        *,
        epoch: int,
    ) -> Dict[str, float]:
        if set(task_objectives) != set(self.tasks):
            raise ValueError("Task objectives do not exactly match trainer tasks")
        if self.task_loss_normalization == "none":
            self.current_task_loss_weights = dict(self.loss_weights)
            return dict(self.current_task_loss_weights)

        detached = {
            task: float(task_objectives[task].detach().float().cpu())
            for task in self.tasks
        }
        if not all(math.isfinite(value) for value in detached.values()):
            return dict(self.current_task_loss_weights)

        for task, value in detached.items():
            magnitude = max(abs(value), self.task_loss_ema_eps)
            self.task_loss_ema_values[task] = (
                self.task_loss_ema_beta * self.task_loss_ema_values[task]
                + (1.0 - self.task_loss_ema_beta) * magnitude
            )
        self.task_loss_ema_updates += 1

        if epoch < self.task_loss_ema_warmup_epochs:
            weights = {task: 1.0 for task in self.tasks}
        else:
            mean_ema = float(np.mean(list(self.task_loss_ema_values.values())))
            weights = {
                task: float(
                    np.clip(
                        mean_ema / (value + self.task_loss_ema_eps),
                        self.task_loss_ema_min_weight,
                        self.task_loss_ema_max_weight,
                    )
                )
                for task, value in self.task_loss_ema_values.items()
            }
        self.current_task_loss_weights = weights
        return dict(weights)

    def _set_step_learning_rate(self, optimizer_step: int) -> None:
        if self.scheduler_mode != "step":
            return
        step = min(
            max(int(optimizer_step), 0),
            max(
                getattr(self, "stage_total_planned_steps", self.total_planned_steps)
                - 1,
                0,
            ),
        )
        for param_group, base_lr in zip(
            self.optimizer.optimizer.param_groups, self.base_lrs
        ):
            low_lr = base_lr / 1000.0
            if self.warmup_steps > 0 and step < self.warmup_steps:
                denominator = max(1, self.warmup_steps - 1)
                fraction = step / denominator
                lr = low_lr + (base_lr - low_lr) * fraction
            else:
                decay_steps = max(
                    1,
                    getattr(
                        self, "stage_total_planned_steps", self.total_planned_steps
                    )
                    - self.warmup_steps
                    - 1,
                )
                progress = min(
                    1.0,
                    max(0.0, (step - self.warmup_steps) / decay_steps),
                )
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                    1.0 + math.cos(math.pi * progress)
                )
            param_group["lr"] = float(lr)

    def _training_progress(self, optimizer_step: int) -> float:
        return min(1.0, max(0.0, optimizer_step / max(1, self.total_planned_steps)))

    def _shared_gradient_parameters(self) -> List[torch.nn.Parameter]:
        molecular_encoder = self.model.molecular_encoder
        graph_encoder = molecular_encoder.graph_encoder
        graph_parameter_ids = (
            {id(parameter) for parameter in graph_encoder.parameters()}
            if graph_encoder is not None
            else set()
        )
        if graph_encoder is None:
            graph_parameters = []
        elif getattr(self.model, "graph_message_sharing", "shared") == "shared":
            graph_parameters = list(graph_encoder.parameters())
        else:
            graph_parameters = list(graph_encoder.shared_parameters())

        parameters: List[torch.nn.Parameter] = []
        seen = set()

        def add_parameter(parameter: torch.nn.Parameter) -> None:
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))

        for parameter in graph_parameters:
            add_parameter(parameter)
        if getattr(
            self.model, "primary_molecular_fusion_route", "shared"
        ) == "shared":
            for parameter in molecular_encoder.parameters():
                if id(parameter) not in graph_parameter_ids:
                    add_parameter(parameter)
        if getattr(
            self.model, "primary_condition_fusion_route", "shared"
        ) == "shared":
            for parameter in self.model.feature_fusion.parameters():
                add_parameter(parameter)
        return parameters

    def _shared_gradient_cosines(
        self,
        task_objectives: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        parameters = self._shared_gradient_parameters()
        if len(task_objectives) < 2 or not parameters:
            return {}

        flattened: Dict[str, torch.Tensor] = {}
        for task, objective in task_objectives.items():
            gradients = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            parts = [
                gradient.detach().float().reshape(-1)
                if gradient is not None
                else torch.zeros(
                    parameter.numel(), device=parameter.device, dtype=torch.float32
                )
                for parameter, gradient in zip(parameters, gradients)
            ]
            flattened[task] = torch.cat(parts)

        result: Dict[str, float] = {}
        cosine_values = []
        ordered_tasks = [task for task in self.tasks if task in flattened]
        for left_index, left_task in enumerate(ordered_tasks):
            left = flattened[left_task]
            left_norm = torch.linalg.vector_norm(left)
            for right_task in ordered_tasks[left_index + 1:]:
                right = flattened[right_task]
                denominator = left_norm * torch.linalg.vector_norm(right)
                cosine = (
                    torch.dot(left, right) / denominator
                    if denominator > 0
                    else torch.tensor(float("nan"), device=left.device)
                )
                value = float(cosine.detach().cpu())
                result[f"grad_cosine__{left_task}__{right_task}"] = value
                if np.isfinite(value):
                    cosine_values.append(value)
        if cosine_values:
            result["grad_cosine_mean"] = float(np.mean(cosine_values))
            result["grad_cosine_negative_fraction"] = float(
                np.mean(np.asarray(cosine_values) < 0)
            )
        return result

    def _group_gradient_cosines(
        self,
        task_objectives: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        grouped = getattr(self.model, "grouped_sharing", None)
        if grouped is None:
            return {}
        results: Dict[str, float] = {}
        for group, block in grouped.blocks.items():
            group_tasks = [
                task
                for task in self.tasks
                if grouped.task_to_group[task] == group and task in task_objectives
            ]
            if len(group_tasks) != 2:
                continue
            parameters = [
                parameter for parameter in block.parameters() if parameter.requires_grad
            ]
            if not parameters:
                continue
            flattened: Dict[str, torch.Tensor] = {}
            for task in group_tasks:
                gradients = torch.autograd.grad(
                    task_objectives[task],
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                flattened[task] = torch.cat(
                    [
                        gradient.detach().float().reshape(-1)
                        if gradient is not None
                        else torch.zeros(
                            parameter.numel(),
                            device=parameter.device,
                            dtype=torch.float32,
                        )
                        for parameter, gradient in zip(parameters, gradients)
                    ]
                )
            left_task, right_task = group_tasks
            left = flattened[left_task]
            right = flattened[right_task]
            denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
            cosine = (
                torch.dot(left, right) / denominator
                if denominator > 0
                else torch.tensor(float("nan"), device=left.device)
            )
            results[
                f"group_grad_cosine__{group}__{left_task}__{right_task}"
            ] = float(cosine.detach().cpu())
        return results

    def _requested_training_tasks(self, task: str) -> List[str]:
        requested = [task]
        paired_task = self.ec_pairs.get(task)
        if (
            self.model_kind in {"mmoe", "grouped"}
            and paired_task in self.tasks
        ):
            requested.append(paired_task)
        return requested

    def _ordinal_loss(
        self,
        task: str,
        outputs: Dict[str, Dict[str, torch.Tensor]],
        *,
        training_progress: float,
    ) -> Optional[torch.Tensor]:
        paired_task = self.ec_pairs.get(task)
        if (
            self.lambda_ord <= 0
            or training_progress < self.ordering_start_fraction
            or paired_task not in outputs
            or task not in self.label_scaler_params
            or paired_task not in self.label_scaler_params
        ):
            return None

        mean_selected, scale_selected = self.label_scaler_params[task]
        mean_paired, scale_paired = self.label_scaler_params[paired_task]
        selected_raw = outputs[task]["mu"].float() * scale_selected + mean_selected
        paired_raw = outputs[paired_task]["mu"].float() * scale_paired + mean_paired
        if "EC50" in task:
            ec50_raw, ec10_raw = selected_raw, paired_raw
        else:
            ec50_raw, ec10_raw = paired_raw, selected_raw
        return torch.relu(ec10_raw - ec50_raw).mean()

    @torch.no_grad()
    def _evaluation_step(
        self,
        *,
        loaders: Dict[str, DataLoader],
        datasets: Dict[str, YieldDataset],
        split_role: str,
        evaluation_tasks: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        if set(loaders) != set(self.tasks) or set(datasets) != set(self.tasks):
            raise ValueError(f"Incomplete {split_role} loaders/datasets")
        selected_tasks = (
            list(self.tasks) if evaluation_tasks is None else list(evaluation_tasks)
        )
        if (
            not selected_tasks
            or len(selected_tasks) != len(set(selected_tasks))
            or not set(selected_tasks).issubset(self.tasks)
        ):
            raise ValueError(f"Invalid {split_role} evaluation task subset")

        self.model.eval()
        export_gate_weights = self.model_kind == "mmoe"
        export_ordering_diagnostics = (
            split_role == "outer_test" and self.model_kind in {"mmoe", "grouped"}
        )
        export_routing_diagnostics = bool(
            getattr(self.model, "supports_routing_diagnostics", False)
        )
        task_losses = {task: [] for task in selected_tasks}
        all_metrics: Dict[str, float] = {}
        prediction_frames = []
        metric_rows = []

        for task in selected_tasks:
            dataset = datasets[task]
            label_scaler = self.train_datasets[task].label_scaler
            paired_task = self.ec_pairs.get(task)
            requested_tasks = [task]
            if export_ordering_diagnostics and paired_task in self.tasks:
                requested_tasks.append(paired_task)

            preds_batches = []
            labels_batches = []
            uncertainty_batches = []
            paired_pred_batches = []
            gate_batches = []
            group_residual_batches = []
            adapter_residual_batches = []
            sample_ids = []

            for batch in tqdm(loaders[task], desc=f"Processing {split_role}/{task}", leave=False):
                (
                    graph,
                    duration_values,
                    effect_onehots,
                    media_onehots,
                    labels,
                    ghs_classes,
                    smiles_list,
                    batch_sample_ids,
                ) = batch

                graph = graph.to(self.device)
                duration_values = duration_values.to(self.device)
                effect_onehots = effect_onehots.to(self.device)
                media_onehots = media_onehots.to(self.device)
                labels = labels.to(self.device)
                ghs_classes = ghs_classes.to(self.device)

                forward_kwargs = {
                    "smiles_list": smiles_list,
                    "requested_tasks": requested_tasks,
                }
                if self.model_kind in {"mmoe", "grouped"}:
                    forward_kwargs["return_gate_weights"] = export_gate_weights
                if export_routing_diagnostics:
                    forward_kwargs["return_routing_diagnostics"] = True
                outputs = self.model(
                    graph,
                    duration_values,
                    effect_onehots,
                    media_onehots,
                    **forward_kwargs,
                )
                task_out = outputs[task]
                mu = task_out["mu"]
                regression_loss = self._regression_loss(task_out, labels)
                cls_log_probs = self._ghs_soft_logprobs(mu, task)
                cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                task_losses[task].append(
                    (regression_loss + self.lambda_cls * cls_loss).item()
                )

                preds_batches.append(
                    label_scaler.inverse_transform(
                        mu.detach().cpu().numpy().reshape(-1, 1)
                    ).reshape(-1)
                )
                labels_batches.append(
                    label_scaler.inverse_transform(
                        labels.detach().cpu().numpy().reshape(-1, 1)
                    ).reshape(-1)
                )
                if self.regression_mode == "heteroscedastic":
                    uncertainty_batches.append(
                        torch.exp(task_out["log_sigma"])
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        * float(label_scaler.scale_[0])
                    )
                else:
                    uncertainty_batches.append(
                        np.full(len(batch_sample_ids), np.nan, dtype=float)
                    )

                if paired_task in outputs:
                    paired_scaler = self.train_datasets[paired_task].label_scaler
                    paired_pred_batches.append(
                        paired_scaler.inverse_transform(
                            outputs[paired_task]["mu"]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, 1)
                        ).reshape(-1)
                    )
                if export_gate_weights:
                    gate_batches.append(
                        task_out["gate_weights"].detach().cpu().numpy()
                    )
                group_residual_batches.append(
                    task_out.get(
                        "group_residual_ratio",
                        torch.full_like(task_out["mu"], float("nan")),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                adapter_residual_batches.append(
                    task_out.get(
                        "adapter_residual_ratio",
                        torch.full_like(task_out["mu"], float("nan")),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )
                sample_ids.extend(batch_sample_ids)

            if not preds_batches:
                raise RuntimeError(f"No predictions produced for {split_role}/{task}")

            all_preds = np.concatenate(preds_batches)
            all_labels = np.concatenate(labels_batches)
            all_uncertainty = np.concatenate(uncertainty_batches)
            all_paired_preds = (
                np.concatenate(paired_pred_batches)
                if paired_pred_batches
                else np.full(len(sample_ids), np.nan, dtype=float)
            )
            all_gate_weights = np.concatenate(gate_batches) if gate_batches else None
            all_group_residual = np.concatenate(group_residual_batches)
            all_adapter_residual = np.concatenate(adapter_residual_batches)
            if sample_ids != dataset.sample_ids:
                raise AssertionError(
                    f"Prediction sample_id order does not match frozen {split_role} manifest "
                    f"for {task}"
                )
            if len(set(sample_ids)) != len(sample_ids):
                raise AssertionError(f"Duplicate prediction sample_id values for {split_role}/{task}")
            if not (
                len(all_preds)
                == len(all_labels)
                == len(all_uncertainty)
                == len(all_paired_preds)
                == len(sample_ids)
                == len(dataset.sample_ids)
            ):
                raise AssertionError(f"Prediction row-count mismatch for {split_role}/{task}")
            if not np.isfinite(all_preds).all() or not np.isfinite(all_labels).all():
                raise FloatingPointError(f"Non-finite prediction values for {split_role}/{task}")
            if self.regression_mode == "heteroscedastic" and (
                not np.isfinite(all_uncertainty).all() or (all_uncertainty <= 0).any()
            ):
                raise FloatingPointError(f"Invalid predictive uncertainty for {split_role}/{task}")
            if all_gate_weights is not None:
                if not np.isfinite(all_gate_weights).all():
                    raise FloatingPointError(f"Non-finite MMoE gates for {split_role}/{task}")
                if not np.allclose(all_gate_weights.sum(axis=1), 1.0, atol=1e-5):
                    raise FloatingPointError(f"MMoE gate rows do not sum to one for {split_role}/{task}")

            preds_tensor = torch.tensor(all_preds, device=self.device, dtype=torch.float32)
            labels_tensor = torch.tensor(all_labels, device=self.device, dtype=torch.float32)
            for metric_fn in self.valid_metrics[task].values():
                metric_fn.update(preds_tensor, labels_tensor)

            all_metrics[f"{task}_loss"] = float(np.mean(task_losses[task]))
            for metric_name, metric_fn in self.valid_metrics[task].items():
                try:
                    value = metric_fn.compute()
                    all_metrics[f"{task}_{metric_name}"] = (
                        value.item() if isinstance(value, torch.Tensor) else float(value)
                    )
                finally:
                    metric_fn.reset()

            metric_rows.append(
                {
                    "model": self.variant_name,
                    "model_kind": self.model_kind,
                    "regression_mode": self.regression_mode,
                    "dataset_layer": self.dataset_layer,
                    "task": task,
                    "protocol": self.cv_protocol,
                    "fold": self.outer_fold_id,
                    "split_role": split_role,
                    "seed": self.seed,
                    "n": len(sample_ids),
                    **{
                        {
                            "r2": "R2",
                            "rmse": "RMSE",
                            "mae": "MAE",
                            "mse": "MSE",
                            "pearson": "Pearson_r",
                        }.get(metric_name, metric_name): all_metrics.get(
                            f"{task}_{metric_name}"
                        )
                        for metric_name in self.valid_metrics[task]
                    },
                }
            )

            if paired_pred_batches:
                if "EC50" in task:
                    predicted_ec50, predicted_ec10 = all_preds, all_paired_preds
                else:
                    predicted_ec50, predicted_ec10 = all_paired_preds, all_preds
                predicted_gap = predicted_ec50 - predicted_ec10
                ordering_violation = predicted_gap < 0
                ordering_context = "same_covariate_counterfactual"
                exported_paired_task = paired_task
            else:
                predicted_ec50 = np.full(len(sample_ids), np.nan, dtype=float)
                predicted_ec10 = np.full(len(sample_ids), np.nan, dtype=float)
                predicted_gap = np.full(len(sample_ids), np.nan, dtype=float)
                ordering_violation = np.full(len(sample_ids), np.nan, dtype=float)
                ordering_context = None
                exported_paired_task = None

            molecule_by_sample = dict(zip(dataset.sample_ids, dataset.molecule_ids))
            split_parent_by_sample = dict(
                zip(dataset.sample_ids, dataset.split_parent_ids)
            )
            split_group_by_sample = dict(
                zip(dataset.sample_ids, dataset.split_group_ids)
            )
            prediction_data = {
                "sample_id": sample_ids,
                "standard_molecule_id": [
                    molecule_by_sample[sample_id] for sample_id in sample_ids
                ],
                "split_parent_id": [
                    split_parent_by_sample[sample_id] for sample_id in sample_ids
                ],
                "split_group_id": [
                    split_group_by_sample[sample_id] for sample_id in sample_ids
                ],
                "task": task,
                "dataset_layer": self.dataset_layer,
                "protocol": self.cv_protocol,
                "outer_fold_id": self.outer_fold_id,
                "split_role": split_role,
                "seed": self.seed,
                "model": self.variant_name,
                "model_kind": self.model_kind,
                "regression_mode": self.regression_mode,
                "y_true": all_labels,
                "y_pred": all_preds,
                "uncertainty": all_uncertainty,
                "paired_task": exported_paired_task,
                "paired_y_pred": all_paired_preds,
                "predicted_ec50": predicted_ec50,
                "predicted_ec10": predicted_ec10,
                "predicted_gap": predicted_gap,
                "ordering_violation": ordering_violation,
                "ordering_context": ordering_context,
                "group_residual_ratio": all_group_residual,
                "adapter_residual_ratio": all_adapter_residual,
            }
            if all_gate_weights is not None:
                for expert_index in range(all_gate_weights.shape[1]):
                    prediction_data[f"gate_expert_{expert_index}"] = all_gate_weights[
                        :, expert_index
                    ]
            prediction_frames.append(pd.DataFrame(prediction_data))

        all_metrics["loss"] = float(
            np.mean([all_metrics[f"{task}_loss"] for task in selected_tasks])
        )
        for source_suffix, aggregate_name in (
            ("_rmse", "avg_rmse"),
            ("_r2", "avg_r2"),
            ("_pearson", "avg_pearson"),
        ):
            values = [
                value
                for key, value in all_metrics.items()
                if key.endswith(source_suffix) and value is not None
            ]
            all_metrics[aggregate_name] = float(np.mean(values)) if values else None

        predictions = pd.concat(prediction_frames, ignore_index=True)
        if predictions["sample_id"].duplicated().any():
            raise AssertionError(f"Duplicate sample_id values across {split_role} predictions")
        expected_ids = {
            sample_id
            for task, dataset in datasets.items()
            if task in selected_tasks
            for sample_id in dataset.sample_ids
        }
        if set(predictions["sample_id"]) != expected_ids:
            raise AssertionError(f"{split_role} predictions do not exactly match the frozen manifest")

        if split_role == "outer_test":
            predictions_dir = self.run_dir / "predictions"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(predictions_dir / f"{split_role}_latest.csv", index=False)
            predictions.to_parquet(
                predictions_dir / f"{split_role}_latest.parquet", index=False
            )
            pd.DataFrame(metric_rows).to_csv(
                predictions_dir / f"{split_role}_metrics_latest.csv", index=False
            )
        if not hasattr(self, "last_prediction_tables"):
            self.last_prediction_tables = {}
        self.last_prediction_tables[split_role] = predictions

        logger_metrics = dict(all_metrics)
        if hasattr(self, "current_epoch"):
            logger_metrics["epoch"] = self.current_epoch
        self.logger.log_metrics(metrics=logger_metrics, prefix=split_role)
        self.model.train()
        return all_metrics

    def valid_step(self) -> Dict[str, float]:
        return self._evaluation_step(
            loaders=self.valid_loaders,
            datasets=self.valid_datasets,
            split_role="inner_val",
        )

    def valid_tasks_step(self, tasks: Sequence[str]) -> Dict[str, float]:
        return self._evaluation_step(
            loaders=self.valid_loaders,
            datasets=self.valid_datasets,
            split_role="inner_val",
            evaluation_tasks=tasks,
        )

    def test_step(self) -> Dict[str, float]:
        if not self.test_loaders:
            raise RuntimeError("No outer_test datasets were configured")
        return self._evaluation_step(
            loaders=self.test_loaders,
            datasets=self.test_datasets,
            split_role="outer_test",
        )

    def make_checkpoint(
        self,
        checkpoint_name: str,
        *,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        model_state: Optional[Dict[str, torch.Tensor]] = None,
        include_optimizer: bool = False,
        selected_task: str | None = None,
        loss_balancing_state_override: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        ckpt_dir = self.run_dir / "train" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_spec = getattr(self.model, "model_spec", None)
        if not isinstance(model_spec, dict):
            raise ValueError("Model must expose a serializable model_spec")
        if model_spec.get("regression_mode") != self.regression_mode:
            raise ValueError("model_spec regression_mode does not match trainer")
        if list(getattr(self.model, "tasks", [])) != list(self.tasks):
            raise ValueError("Checkpoint model tasks do not match trainer tasks")

        reference_dataset = self.train_datasets[self.tasks[0]]
        ckpt_dict = {
            "schema_version": 1,
            "model_class": type(self.model).__name__,
            "model_spec": model_spec,
            "model": model_state if model_state is not None else self.model.state_dict(),
            "variant_name": self.variant_name,
            "model_kind": self.model_kind,
            "regression_mode": self.regression_mode,
            "tasks": list(self.tasks),
            "metrics": metrics,
            "selected_task": selected_task,
            "seed": self.seed,
            "training_spec": {
                "lambda_cls": self.lambda_cls,
                "lambda_ord": self.lambda_ord,
                "lambda_mu": self.lambda_mu,
                "huber_beta": self.huber_beta,
                "ordering_start_fraction": self.ordering_start_fraction,
                "n_steps_per_epoch": self.n_steps_per_epoch,
                "total_planned_steps": self.total_planned_steps,
                "training_scope": self.training_scope,
                "staged_finetuning": self.staged_finetuning,
                "stage1_epochs": self.stage1_epochs,
                "stage2_epochs": self.stage2_epochs,
                "checkpoint_stage": self.current_stage,
                "checkpoint_stage_epoch": getattr(
                    self, "current_stage_local_epoch", int(epoch)
                ),
                "stage_learning_rates": dict(self.stage_lrs),
                "train_batch_size": self.train_batch_size,
                "valid_batch_size": self.valid_batch_size,
                "task_loss_reduction": (
                    "ema_normalized_mean"
                    if self.task_loss_normalization == "ema"
                    else (
                        "mean"
                        if all(
                            math.isclose(weight, 1.0)
                            for weight in self.loss_weights.values()
                        )
                        else "normalized_weighted_mean"
                    )
                ),
                "task_loss_weights": dict(self.loss_weights),
                "task_loss_normalization": self.task_loss_normalization,
                "task_loss_ema_beta": self.task_loss_ema_beta,
                "task_loss_ema_eps": self.task_loss_ema_eps,
                "task_loss_ema_warmup_epochs": self.task_loss_ema_warmup_epochs,
                "task_loss_ema_min_weight": self.task_loss_ema_min_weight,
                "task_loss_ema_max_weight": self.task_loss_ema_max_weight,
                "lambda_expert_diversity": self.lambda_expert_diversity,
                "sampler_seed_scheme": "fold_seed_plus_canonical_task_index",
                "use_pcgrad": self.use_pcgrad,
                "pcgrad_scope": self.pcgrad_scope,
                "use_swa": self.use_swa,
                "scheduler_mode": self.scheduler_mode,
                "warmup_epochs": self.warmup_epochs,
                "warmup_fraction": self.warmup_fraction,
                "warmup_steps": self.warmup_steps,
                "base_lrs": list(self.base_lrs),
                "min_lr": self.min_lr,
                "gradient_diagnostics_every_n_epochs": (
                    self.gradient_diagnostics_every_n_epochs
                ),
                "n_epochs": self.n_epochs,
                "valid_every_n_epochs": self.valid_every_n_epochs,
                "patience": self.patience,
                "best_metric": self.best_metric,
                "metric_direction": self.metric_direction,
                "checkpoint_selection": self.checkpoint_selection,
                "task_private_finetune": (
                    {
                        "base_selection": "inner_macro_rmse",
                        "base_checkpoint": "best_model.pt",
                        "private_modules": list(TASK_PRIVATE_MODULE_NAMES),
                        "frozen_modules_mode": "eval",
                        "base_epoch_zero_eligible": True,
                        "stage_b_max_epochs_per_task": self.n_epochs,
                        "stage_b_valid_every_n_epochs": self.valid_every_n_epochs,
                        "stage_b_patience_validations": self.patience,
                        "stage_b_steps_per_epoch": self.n_steps_per_epoch,
                        "stage_b_seed_formula": (
                            "fold_seed_plus_1000000_plus_canonical_task_index"
                        ),
                        "sampler": self.task_private_sampling_mode,
                    }
                    if self.checkpoint_selection == "task_private_finetune"
                    else None
                ),
                "checkpoint_epoch": int(epoch),
                "finetune_source": (
                    {
                        "path": self.transfer_audit["source_checkpoint"],
                        "sha256": self.transfer_audit["source_checkpoint_sha256"],
                        "variant": self.transfer_audit["source_variant"],
                    }
                    if self.transfer_audit is not None
                    else None
                ),
            },
            "data_contract": {
                "dataset_layer": self.dataset_layer,
                "cv_protocol": self.cv_protocol,
                "outer_fold_id": self.outer_fold_id,
                "model_csv_path": str(reference_dataset.file_path),
                "dataset_sha256": reference_dataset.dataset_sha256,
                "manifest_path": str(reference_dataset.manifest_path),
                "manifest_sha256": reference_dataset.manifest_sha256,
                "feature_cache_version": FEATURE_CACHE_VERSION,
                "training_scope": self.training_scope,
            },
            "label_scalers": {
                task: {"mean": mean, "scale": scale}
                for task, (mean, scale) in self.label_scaler_params.items()
            },
            "feature_preprocessing": self.feature_preprocessing,
            "pcgrad_parameter_scope_audit": self.pcgrad_parameter_scope_audit,
            "loss_balancing_state": dict(
                loss_balancing_state_override
                if loss_balancing_state_override is not None
                else self._loss_balancing_state()
            ),
        }
        if include_optimizer:
            ckpt_dict["optimizer"] = self.optimizer.optimizer.state_dict()
        target = ckpt_dir / f"{checkpoint_name}.pt"
        atomic_torch_save(ckpt_dict, target)
        return target

    def _update_taskwise_selection(
        self,
        valid_metrics: Mapping[str, float],
        *,
        epoch: int,
    ) -> None:
        if self.checkpoint_selection not in {"taskwise", "task_soup"}:
            return
        predictions = self.last_prediction_tables.get("inner_val")
        if predictions is None or predictions.empty:
            raise RuntimeError("Task-wise selection requires inner predictions")

        improved_tasks = []
        for task in self.tasks:
            metric_name = f"{task}_rmse"
            value = valid_metrics.get(metric_name)
            if value is None or not np.isfinite(value):
                raise FloatingPointError(
                    f"Task-wise metric {metric_name!r} is missing or non-finite"
                )
            if float(value) < self.best_task_valid_metrics[task]:
                improved_tasks.append(task)
        if not improved_tasks:
            return

        snapshot = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.model.state_dict().items()
        }
        for task in improved_tasks:
            metric_name = f"{task}_rmse"
            selected = predictions.loc[predictions["task"].eq(task)].copy()
            expected_ids = self.valid_datasets[task].sample_ids
            if selected["sample_id"].astype(str).tolist() != list(expected_ids):
                raise AssertionError(
                    f"Task-wise prediction order changed for inner_val/{task}"
                )
            self.best_task_valid_metrics[task] = float(valid_metrics[metric_name])
            self.best_task_records[task] = {
                "selected_task": task,
                "selection_metric": metric_name,
                "selection_value": float(valid_metrics[metric_name]),
                "epoch": int(epoch),
                "stage": self.current_stage,
                "stage_epoch": int(self.current_stage_local_epoch),
            }
            self.best_task_model_states[task] = snapshot
            if self.checkpoint_selection == "task_soup":
                self.best_task_loss_balancing_states[task] = (
                    self._loss_balancing_state()
                )
            self.best_task_prediction_frames[task] = selected

    def _update_balanced_selection(
        self,
        valid_metrics: Mapping[str, float],
        *,
        epoch: int,
    ) -> None:
        if self.checkpoint_selection != "balanced":
            return
        required = ["avg_rmse", *(f"{task}_rmse" for task in self.tasks)]
        values = {name: valid_metrics.get(name) for name in required}
        if any(value is None or not np.isfinite(value) for value in values.values()):
            raise FloatingPointError(
                "Balanced checkpoint selection requires finite Macro and task RMSE values"
            )
        self.balanced_validation_records.append(
            {
                "epoch": int(epoch),
                "stage": self.current_stage,
                "stage_epoch": int(self.current_stage_local_epoch),
                **{name: float(value) for name, value in values.items()},
            }
        )
        self.balanced_model_states.append(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.model.state_dict().items()
            }
        )
        self.balanced_loss_balancing_states.append(self._loss_balancing_state())

    def _finalize_taskwise_selection(self) -> None:
        if self.checkpoint_selection != "taskwise":
            return
        missing = sorted(
            task
            for task in self.tasks
            if (
                task not in self.best_task_records
                or task not in self.best_task_model_states
                or task not in self.best_task_prediction_frames
            )
        )
        if missing:
            raise RuntimeError(f"Task-wise checkpoints are incomplete: {missing}")

        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        selector_checkpoint = checkpoint_dir / "best_model.pt"
        if not selector_checkpoint.is_file():
            raise FileNotFoundError(selector_checkpoint)
        inventory: Dict[str, Dict[str, Any]] = {}
        prediction_parts = []
        for task in self.tasks:
            record = self.best_task_records[task]
            checkpoint = self.make_checkpoint(
                f"best_model_{task}",
                epoch=int(record["epoch"]),
                metrics=record,
                model_state=self.best_task_model_states[task],
                selected_task=task,
            )
            selected_predictions = self.best_task_prediction_frames[task]
            prediction_parts.append(selected_predictions)
            inventory[task] = {
                **record,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": _sha256_file(checkpoint),
                "prediction_rows": int(len(selected_predictions)),
            }

        predictions = pd.concat(prediction_parts, ignore_index=True)
        expected_ids = {
            str(sample_id)
            for task in self.tasks
            for sample_id in self.valid_datasets[task].sample_ids
        }
        if (
            predictions["sample_id"].duplicated().any()
            or set(predictions["sample_id"].astype(str)) != expected_ids
        ):
            raise AssertionError("Task-wise inner predictions are not exactly aligned")
        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        selected_csv = predictions_dir / "inner_val_best.csv"
        selected_parquet = predictions_dir / "inner_val_best.parquet"
        predictions.to_csv(selected_csv, index=False)
        predictions.to_parquet(selected_parquet, index=False)
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "selection": "taskwise_inner_rmse",
            "variant_name": self.variant_name,
            "seed": self.seed,
            "outer_fold_id": self.outer_fold_id,
            "selector_checkpoint": selector_checkpoint.name,
            "selector_checkpoint_sha256": _sha256_file(selector_checkpoint),
            "inner_predictions": selected_csv.name,
            "inner_predictions_sha256": _sha256_file(selected_csv),
            "tasks": inventory,
        }
        atomic_write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            checkpoint_dir / "taskwise_selection.json",
        )

    def _finalize_task_soup_selection(self) -> None:
        if self.checkpoint_selection != "task_soup":
            return
        missing = sorted(
            task
            for task in self.tasks
            if (
                task not in self.best_task_records
                or task not in self.best_task_model_states
                or task not in self.best_task_loss_balancing_states
            )
        )
        if missing:
            raise RuntimeError(f"Task-soup snapshots are incomplete: {missing}")

        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        macro_checkpoint_path = checkpoint_dir / "best_model.pt"
        if not macro_checkpoint_path.is_file():
            raise FileNotFoundError(macro_checkpoint_path)
        macro_checkpoint = torch.load(macro_checkpoint_path, map_location="cpu")
        macro_state = macro_checkpoint.get("model")
        if not isinstance(macro_state, dict):
            raise ValueError("Macro-best checkpoint has no model state")

        source_states = [self.best_task_model_states[task] for task in self.tasks]
        averaged_state = _uniform_task_soup_state(source_states, macro_state)

        source_inventory: Dict[str, Dict[str, Any]] = {}
        for task in self.tasks:
            record = self.best_task_records[task]
            checkpoint = self.make_checkpoint(
                f"task_soup_source_{task}",
                epoch=int(record["epoch"]),
                metrics=record,
                model_state=self.best_task_model_states[task],
                selected_task=task,
                loss_balancing_state_override=(
                    self.best_task_loss_balancing_states[task]
                ),
            )
            source_inventory[task] = {
                **record,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": _sha256_file(checkpoint),
            }

        self.model.load_state_dict(averaged_state, strict=True)
        soup_metrics = self.valid_step()
        soup_record: Dict[str, Any] = {
            **soup_metrics,
            "epoch": int(self.current_epoch),
            "stage": self.current_stage,
            "stage_epoch": int(self.current_stage_local_epoch),
            "checkpoint_role": "task_soup",
            "source_epochs": {
                task: int(self.best_task_records[task]["epoch"])
                for task in self.tasks
            },
        }
        soup_checkpoint = self.make_checkpoint(
            "best_model_task_soup",
            epoch=int(self.current_epoch),
            metrics=soup_record,
            model_state=averaged_state,
        )
        predictions = self.last_prediction_tables.get("inner_val")
        if predictions is None or predictions.empty:
            raise RuntimeError("Task-soup validation produced no inner predictions")
        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        selected_csv = predictions_dir / "inner_val_best.csv"
        selected_parquet = predictions_dir / "inner_val_best.parquet"
        predictions.to_csv(selected_csv, index=False)
        predictions.to_parquet(selected_parquet, index=False)

        manifest = {
            "schema_version": 1,
            "status": "completed",
            "selection": "task_best_uniform_soup",
            "variant_name": self.variant_name,
            "seed": self.seed,
            "outer_fold_id": self.outer_fold_id,
            "macro_checkpoint": macro_checkpoint_path.name,
            "macro_checkpoint_sha256": _sha256_file(macro_checkpoint_path),
            "soup_checkpoint": soup_checkpoint.name,
            "soup_checkpoint_sha256": _sha256_file(soup_checkpoint),
            "inner_predictions": selected_csv.name,
            "inner_predictions_sha256": _sha256_file(selected_csv),
            "averaging": {
                "source_count": len(self.tasks),
                "floating_state": "uniform_mean_fp64_accumulator",
                "non_floating_state": "macro_best",
            },
            "metrics": soup_metrics,
            "tasks": source_inventory,
        }
        atomic_write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            checkpoint_dir / "task_soup_selection.json",
        )
        self.best_valid_metrics = soup_record
        self.best_valid_metric = float(soup_metrics[self.best_metric])

    def _finalize_balanced_selection(self) -> None:
        if self.checkpoint_selection != "balanced":
            return
        if not (
            len(self.balanced_validation_records)
            == len(self.balanced_model_states)
            == len(self.balanced_loss_balancing_states)
        ):
            raise RuntimeError("Balanced validation snapshots are incomplete")

        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        macro_checkpoint_path = checkpoint_dir / "best_model.pt"
        if not macro_checkpoint_path.is_file():
            raise FileNotFoundError(macro_checkpoint_path)
        selection = _select_balanced_checkpoint_record(
            self.balanced_validation_records,
            self.tasks,
            macro_tolerance=0.01,
        )
        selected_index = int(selection["selected_record_index"])
        selected_record = self.balanced_validation_records[selected_index]
        selected_state = self.balanced_model_states[selected_index]
        selected_loss_state = self.balanced_loss_balancing_states[selected_index]

        self.model.load_state_dict(selected_state, strict=True)
        balanced_metrics = self.valid_step()
        metric_names = ["avg_rmse", *(f"{task}_rmse" for task in self.tasks)]
        mismatches = [
            name
            for name in metric_names
            if not np.isclose(
                float(balanced_metrics[name]),
                float(selected_record[name]),
                rtol=1e-7,
                atol=1e-9,
            )
        ]
        if mismatches:
            raise RuntimeError(
                f"Balanced checkpoint re-evaluation changed metrics: {mismatches}"
            )

        checkpoint_record: Dict[str, Any] = {
            **balanced_metrics,
            "epoch": int(selected_record["epoch"]),
            "stage": selected_record["stage"],
            "stage_epoch": int(selected_record["stage_epoch"]),
            "checkpoint_role": "balanced",
            "macro_tolerance": float(selection["macro_tolerance"]),
            "maximum_relative_regret": float(
                selection["selected_maximum_relative_regret"]
            ),
        }
        balanced_checkpoint = self.make_checkpoint(
            "best_model_balanced",
            epoch=int(selected_record["epoch"]),
            metrics=checkpoint_record,
            model_state=selected_state,
            loss_balancing_state_override=selected_loss_state,
        )
        predictions = self.last_prediction_tables.get("inner_val")
        if predictions is None or predictions.empty:
            raise RuntimeError("Balanced checkpoint validation produced no inner predictions")
        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        selected_csv = predictions_dir / "inner_val_best.csv"
        selected_parquet = predictions_dir / "inner_val_best.parquet"
        predictions.to_csv(selected_csv, index=False)
        predictions.to_parquet(selected_parquet, index=False)

        manifest_records = []
        for record in selection["validation_records"]:
            original = self.balanced_validation_records[int(record["record_index"])]
            manifest_records.append(
                {
                    "epoch": int(record["epoch"]),
                    "stage": original["stage"],
                    "stage_epoch": int(original["stage_epoch"]),
                    "avg_rmse": float(record["avg_rmse"]),
                    "task_rmse": {
                        task: float(value)
                        for task, value in record["task_rmse"].items()
                    },
                    "eligible": bool(record["eligible"]),
                    "relative_regret": {
                        task: float(value)
                        for task, value in record["relative_regret"].items()
                    },
                    "maximum_relative_regret": float(
                        record["maximum_relative_regret"]
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "selection": "macro_within_1pct_minimax_task_regret",
            "variant_name": self.variant_name,
            "seed": self.seed,
            "outer_fold_id": self.outer_fold_id,
            "macro_checkpoint": macro_checkpoint_path.name,
            "macro_checkpoint_sha256": _sha256_file(macro_checkpoint_path),
            "balanced_checkpoint": balanced_checkpoint.name,
            "balanced_checkpoint_sha256": _sha256_file(balanced_checkpoint),
            "inner_predictions": selected_csv.name,
            "inner_predictions_sha256": _sha256_file(selected_csv),
            "macro_tolerance": float(selection["macro_tolerance"]),
            "macro_best_rmse": float(selection["macro_best_rmse"]),
            "macro_limit_rmse": float(selection["macro_limit_rmse"]),
            "eligible_count": int(selection["eligible_count"]),
            "selected_epoch": int(selection["selected_epoch"]),
            "selected_macro_rmse": float(selection["selected_macro_rmse"]),
            "selected_maximum_relative_regret": float(
                selection["selected_maximum_relative_regret"]
            ),
            "task_best_rmse": {
                task: float(value)
                for task, value in selection["task_best_rmse"].items()
            },
            "validation_records": manifest_records,
        }
        atomic_write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            checkpoint_dir / "balanced_checkpoint_selection.json",
        )
        self.best_valid_metrics = checkpoint_record
        self.best_valid_metric = float(balanced_metrics[self.best_metric])

    def _set_task_private_mode(self, task: str) -> List[str]:
        state_keys = _task_private_state_keys(self.model.state_dict(), self.tasks)
        expected = set(state_keys[task])
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        trainable_names = []
        for name, parameter in self.model.named_parameters():
            if name in expected:
                parameter.requires_grad_(True)
                trainable_names.append(name)
        if set(trainable_names) != expected:
            missing = sorted(expected - set(trainable_names))
            extra = sorted(set(trainable_names) - expected)
            raise ValueError(
                f"Task-private parameter inventory mismatch for {task}: "
                f"missing={missing}, extra={extra}"
            )

        self.model.eval()
        for module_name in TASK_PRIVATE_MODULE_NAMES:
            module_dict = getattr(self.model, module_name, None)
            if not isinstance(module_dict, nn.ModuleDict) or task not in module_dict:
                raise ValueError(f"Missing task-private module {module_name}.{task}")
            module_dict[task].train()
        return sorted(trainable_names)

    def _reset_task_private_random_stream(self, task: str) -> int:
        task_index = self.canonical_tasks.index(task)
        stage_seed = int(self.seed + 1_000_000 + task_index)
        random.seed(stage_seed)
        np.random.seed(stage_seed % (2**32))
        torch.manual_seed(stage_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(stage_seed)

        dataset = self.train_datasets[task]
        if self.task_private_sampling_mode == "weighted_replacement":
            loader = self.train_loaders[task]
            sampler = loader.sampler
            if (
                not isinstance(sampler, WeightedRandomSampler)
                or not sampler.replacement
                or sampler.generator is None
                or int(sampler.num_samples) != len(dataset)
            ):
                raise ValueError(f"Sampler contract changed for {task}")
            sampler.generator.manual_seed(stage_seed)
            if loader.generator is None:
                raise ValueError(f"Training loader generator is missing for {task}")
            loader.generator.manual_seed(stage_seed + 100_000)
        else:
            sampler_generator = torch.Generator().manual_seed(stage_seed)
            loader_generator = torch.Generator().manual_seed(stage_seed + 100_000)
            sampler = RandomSampler(
                dataset,
                replacement=False,
                generator=sampler_generator,
            )
            loader = DataLoader(
                dataset,
                batch_size=self.train_batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=dataset.collate,
                pin_memory=False,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
                generator=loader_generator,
            )
            self.train_loaders[task] = loader
        self.train_iterators[task] = iter(loader)
        return stage_seed

    @staticmethod
    def _assert_task_prediction_equivalence(
        expected: pd.DataFrame,
        actual: pd.DataFrame,
        *,
        task: str,
    ) -> None:
        keys = ["task", "sample_id"]
        left = expected.sort_values(keys).reset_index(drop=True)
        right = actual.sort_values(keys).reset_index(drop=True)
        if (
            len(left) != len(right)
            or not left[keys].astype(str).equals(right[keys].astype(str))
        ):
            raise AssertionError(f"Task-private prediction IDs changed for {task}")
        for column in (
            "y_true",
            "y_pred",
            "uncertainty",
            "group_residual_ratio",
            "adapter_residual_ratio",
        ):
            if not np.array_equal(
                left[column].to_numpy(),
                right[column].to_numpy(),
                equal_nan=True,
            ):
                raise AssertionError(
                    f"Task-private merged prediction changed for {task}/{column}"
                )

    def _train_one_task_private_stage(
        self,
        *,
        task: str,
        base_state: Mapping[str, torch.Tensor],
        base_checkpoint_sha256: str,
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, Any], pd.DataFrame]:
        self.model.load_state_dict(base_state, strict=True)
        stage_seed = self._reset_task_private_random_stream(task)
        task_private_loader = self.train_loaders[task]
        trainable_names = self._set_task_private_mode(task)
        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        trainable_count = int(sum(parameter.numel() for parameter in trainable_parameters))
        if trainable_count <= 0:
            raise ValueError(f"No task-private parameters are trainable for {task}")

        base_lr = self.optimizer.kwargs.get("lr")
        if base_lr is None:
            raise ValueError("Task-private optimizer requires an explicit base learning rate")
        parameter_groups = [
            {
                "params": trainable_parameters,
                "lr": float(base_lr),
                "name": f"task_private_{task}",
            }
        ]
        self._validate_parameter_groups(parameter_groups)
        self.optimizer.initialize(parameter_groups=parameter_groups)
        self.current_stage = f"task_private:{task}"
        self.current_stage_local_epoch = 0
        self.current_epoch = 0
        self.stage_optimizer_step = 0
        self._initialize_scheduler(stage_epochs=self.n_epochs)
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.optimizer_group_audit[self.current_stage] = [
            {
                "name": f"task_private_{task}",
                "lr": float(base_lr),
                "parameter_tensors": len(trainable_names),
                "parameters": trainable_count,
                "parameter_names": trainable_names,
            }
        ]

        stage_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stage_started = time.perf_counter()
        initial_metrics = self.valid_tasks_step([task])
        initial_predictions = self.last_prediction_tables["inner_val"].copy()
        best_value = float(initial_metrics[f"{task}_rmse"])
        if not math.isfinite(best_value):
            raise FloatingPointError(f"Non-finite base RMSE for {task}")
        private_keys = _task_private_state_keys(base_state, self.tasks)[task]
        best_private_state = {
            name: base_state[name].detach().cpu().clone() for name in private_keys
        }
        best_predictions = initial_predictions
        best_stage_epoch = 0
        best_optimizer_steps = 0
        selection_origin = "base_epoch_zero"
        validations_without_improvement = 0
        epochs_completed = 0
        early_stopped = False
        loader_resets = 0
        exposure_counts: Counter[str] = Counter()

        for stage_epoch in range(1, self.n_epochs + 1):
            self.current_stage_local_epoch = stage_epoch
            self.current_epoch = stage_epoch
            self._set_task_private_mode(task)
            losses = []
            gradient_norms = []
            for _step in range(self.n_steps_per_epoch):
                self._set_step_learning_rate(self.stage_optimizer_step)
                try:
                    batch = next(self.train_iterators[task])
                except StopIteration:
                    loader_resets += 1
                    self.train_iterators[task] = iter(task_private_loader)
                    batch = next(self.train_iterators[task])
                (
                    graph,
                    duration_values,
                    effect_onehots,
                    media_onehots,
                    labels,
                    ghs_classes,
                    smiles_list,
                    batch_sample_ids,
                ) = batch
                exposure_counts.update(str(value) for value in batch_sample_ids)
                graph = graph.to(self.device)
                duration_values = duration_values.to(self.device)
                effect_onehots = effect_onehots.to(self.device)
                media_onehots = media_onehots.to(self.device)
                labels = labels.to(self.device)
                ghs_classes = ghs_classes.to(self.device)
                training_progress = min(
                    1.0,
                    self.stage_optimizer_step / max(1, self.stage_total_planned_steps),
                )

                with autocast(enabled=self.amp_enabled):
                    outputs = self.model(
                        graph,
                        duration_values,
                        effect_onehots,
                        media_onehots,
                        smiles_list=smiles_list,
                        requested_tasks=self._requested_training_tasks(task),
                    )
                    task_out = outputs[task]
                    mu = task_out["mu"]
                    regression_loss = self._regression_loss(task_out, labels)
                    mean_loss = self._mean_auxiliary_loss(task_out, labels)
                    cls_log_probs = self._ghs_soft_logprobs(mu, task)
                    cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                    task_loss = (
                        regression_loss
                        + self.lambda_mu * mean_loss
                        + self.lambda_cls * cls_loss
                    )
                    ordinal_loss = self._ordinal_loss(
                        task,
                        outputs,
                        training_progress=training_progress,
                    )
                    if ordinal_loss is not None:
                        task_loss = task_loss + self.lambda_ord * ordinal_loss
                    total_loss = task_loss + self.l2_regularization()
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite task-private loss for {task} at epoch {stage_epoch}"
                    )
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters, self.gradient_clipping_norm
                )
                self.scaler.step(self.optimizer.optimizer)
                self.scaler.update()
                self.stage_optimizer_step += 1
                losses.append(float(total_loss.detach().cpu()))
                gradient_norms.append(float(grad_norm.detach().cpu()))

            if self.scheduler_mode == "epoch":
                if stage_epoch <= self.warmup_epochs and self.warmup_epochs > 0:
                    scale = stage_epoch / self.warmup_epochs
                    for group, learning_rate in zip(
                        self.optimizer.optimizer.param_groups, self.base_lrs
                    ):
                        group["lr"] = learning_rate * scale
                else:
                    assert self.lr_scheduler is not None
                    self.lr_scheduler.step()

            epochs_completed = stage_epoch
            self.logger.log_metrics(
                metrics={
                    "task": task,
                    "stage_epoch": stage_epoch,
                    "optimizer_steps": self.stage_optimizer_step,
                    "loss": float(np.mean(losses)),
                    "gradient_norm": float(np.mean(gradient_norms)),
                    "lr": float(self.optimizer.optimizer.param_groups[0]["lr"]),
                },
                prefix="task_private_train",
            )
            should_validate = (
                stage_epoch % self.valid_every_n_epochs == 0
                or stage_epoch == self.n_epochs
            )
            if not should_validate:
                continue
            metrics = self.valid_tasks_step([task])
            predictions = self.last_prediction_tables["inner_val"].copy()
            value = float(metrics[f"{task}_rmse"])
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite task-private RMSE for {task}")
            if value < best_value:
                best_value = value
                best_private_state = {
                    name: self.model.state_dict()[name].detach().cpu().clone()
                    for name in private_keys
                }
                best_predictions = predictions
                best_stage_epoch = stage_epoch
                best_optimizer_steps = self.stage_optimizer_step
                selection_origin = "task_private_finetune"
                validations_without_improvement = 0
            else:
                validations_without_improvement += 1
            if validations_without_improvement >= self.patience:
                early_stopped = True
                break

        source_state = {
            name: tensor.detach().cpu().clone() for name, tensor in base_state.items()
        }
        source_state.update(best_private_state)
        self.model.load_state_dict(source_state, strict=True)
        source_metrics = self.valid_tasks_step([task])
        source_predictions = self.last_prediction_tables["inner_val"].copy()
        self._assert_task_prediction_equivalence(
            best_predictions, source_predictions, task=task
        )
        if not math.isclose(
            float(source_metrics[f"{task}_rmse"]),
            best_value,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise AssertionError(f"Task-private RMSE changed after restoring {task}")

        checkpoint_record: Dict[str, Any] = {
            **source_metrics,
            "selected_task": task,
            "selection_metric": f"{task}_rmse",
            "selection_value": best_value,
            "selection_origin": selection_origin,
            "stage_epoch": best_stage_epoch,
            "optimizer_steps": best_optimizer_steps,
            "base_checkpoint_sha256": base_checkpoint_sha256,
        }
        source_checkpoint = self.make_checkpoint(
            f"task_private_source_{task}",
            epoch=best_stage_epoch,
            metrics=checkpoint_record,
            model_state=source_state,
            selected_task=task,
        )
        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_csv = predictions_dir / f"task_private_best_{task}.csv"
        prediction_parquet = predictions_dir / f"task_private_best_{task}.parquet"
        source_predictions.to_csv(prediction_csv, index=False)
        source_predictions.to_parquet(prediction_parquet, index=False)
        exposure_frame = pd.DataFrame(
            {
                "sample_id": self.train_datasets[task].sample_ids,
                "molecule_id": self.train_datasets[task].molecule_ids,
                "exposure_count": [
                    int(exposure_counts.get(sample_id, 0))
                    for sample_id in self.train_datasets[task].sample_ids
                ],
            }
        )
        exposure_path = predictions_dir / f"task_private_exposure_{task}.csv"
        exposure_frame.to_csv(exposure_path, index=False)
        exposure_values = exposure_frame["exposure_count"].to_numpy(dtype=float)
        exposure_total = float(exposure_values.sum())
        exposure_ess = (
            float(exposure_total**2 / np.square(exposure_values).sum())
            if exposure_total > 0.0
            else 0.0
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        else:
            peak_allocated = 0
            peak_reserved = 0
        self.stage_resource_summaries.append(
            {
                "schema_version": 1,
                "stage": self.current_stage,
                "task": task,
                "started_at_utc": stage_started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "wall_seconds": float(time.perf_counter() - stage_started),
                "epochs_completed": epochs_completed,
                "optimizer_steps": self.stage_optimizer_step,
                "early_stopped": early_stopped,
                "sampling_mode": self.task_private_sampling_mode,
                "loader_resets": loader_resets,
                "trainable_parameters": trainable_count,
                "peak_cuda_memory_allocated_bytes_cumulative": peak_allocated,
                "peak_cuda_memory_reserved_bytes_cumulative": peak_reserved,
                "optimizer_groups": [
                    {
                        "name": f"task_private_{task}",
                        "lr": float(base_lr),
                        "parameter_tensors": len(trainable_names),
                        "parameters": trainable_count,
                    }
                ],
            }
        )
        record = {
            "stage_seed": stage_seed,
            "trainable_keys": trainable_names,
            "trainable_parameter_tensors": len(trainable_names),
            "trainable_parameters": trainable_count,
            "best_stage_epoch": best_stage_epoch,
            "best_optimizer_steps": best_optimizer_steps,
            "selection_metric": f"{task}_rmse",
            "selection_value": best_value,
            "selection_origin": selection_origin,
            "epochs_completed": epochs_completed,
            "early_stopped": early_stopped,
            "sampling_mode": self.task_private_sampling_mode,
            "loader_resets": loader_resets,
            "exposure_draws": int(exposure_total),
            "unique_samples_exposed": int((exposure_values > 0).sum()),
            "sample_exposure_ess": exposure_ess,
            "sample_exposure_ess_ratio": exposure_ess / len(exposure_values),
            "exposure_audit": exposure_path.name,
            "exposure_audit_sha256": _sha256_file(exposure_path),
            "source_checkpoint": source_checkpoint.name,
            "source_checkpoint_sha256": _sha256_file(source_checkpoint),
            "source_state_sha256": _state_dict_digest(source_state),
            "private_state_sha256": _state_dict_digest(source_state, private_keys),
            "inner_predictions": prediction_csv.name,
            "inner_predictions_sha256": _sha256_file(prediction_csv),
            "inner_predictions_parquet": prediction_parquet.name,
            "inner_predictions_parquet_sha256": _sha256_file(prediction_parquet),
        }
        return source_state, record, source_predictions

    def _finalize_task_private_finetuning(self) -> None:
        if self.checkpoint_selection != "task_private_finetune":
            return
        checkpoint_dir = self.run_dir / "train" / "checkpoints"
        base_checkpoint_path = checkpoint_dir / "best_model.pt"
        if not base_checkpoint_path.is_file():
            raise FileNotFoundError(base_checkpoint_path)
        base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu")
        base_state = base_checkpoint.get("model")
        if not isinstance(base_state, dict):
            raise ValueError("Task-private base checkpoint has no model state")
        base_checkpoint_sha256 = _sha256_file(base_checkpoint_path)
        private_keys = _task_private_state_keys(base_state, self.tasks)
        private_key_union = {
            name for task_keys in private_keys.values() for name in task_keys
        }
        shared_keys = sorted(set(base_state) - private_key_union)
        shared_sha256 = _state_dict_digest(base_state, shared_keys)

        task_states: Dict[str, Dict[str, torch.Tensor]] = {}
        task_records: Dict[str, Dict[str, Any]] = {}
        task_prediction_parts = []
        for task in self.tasks:
            source_state, record, predictions = self._train_one_task_private_stage(
                task=task,
                base_state=base_state,
                base_checkpoint_sha256=base_checkpoint_sha256,
            )
            if _state_dict_digest(source_state, shared_keys) != shared_sha256:
                raise AssertionError(f"Shared state changed while fine-tuning {task}")
            task_states[task] = source_state
            task_records[task] = record
            task_prediction_parts.append(predictions)

        merged_state, merged_private_keys = _merge_task_private_states(
            base_state, task_states, self.tasks
        )
        if merged_private_keys != private_keys:
            raise AssertionError("Task-private key inventory changed during merge")
        if _state_dict_digest(merged_state, shared_keys) != shared_sha256:
            raise AssertionError("Merged checkpoint changed shared state")
        for task in self.tasks:
            if _state_dict_digest(
                merged_state, private_keys[task]
            ) != _state_dict_digest(task_states[task], private_keys[task]):
                raise AssertionError(f"Merged private state changed for {task}")

        self.model.load_state_dict(merged_state, strict=True)
        self.current_stage = "task_private_merge"
        self.current_stage_local_epoch = 0
        self.current_epoch = int(
            base_checkpoint.get("training_spec", {}).get("checkpoint_epoch", 0)
        )
        merged_metrics = self.valid_step()
        merged_predictions = self.last_prediction_tables["inner_val"].copy()
        expected_predictions = pd.concat(task_prediction_parts, ignore_index=True)
        for task in self.tasks:
            self._assert_task_prediction_equivalence(
                expected_predictions.loc[expected_predictions["task"].eq(task)],
                merged_predictions.loc[merged_predictions["task"].eq(task)],
                task=task,
            )

        predictions_dir = self.run_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        selected_csv = predictions_dir / "inner_val_best.csv"
        selected_parquet = predictions_dir / "inner_val_best.parquet"
        merged_predictions.to_csv(selected_csv, index=False)
        merged_predictions.to_parquet(selected_parquet, index=False)
        merged_record: Dict[str, Any] = {
            **merged_metrics,
            "epoch": self.current_epoch,
            "stage": self.current_stage,
            "checkpoint_role": "task_private_merged",
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "shared_state_sha256": shared_sha256,
            "task_best_stage_epochs": {
                task: int(task_records[task]["best_stage_epoch"])
                for task in self.tasks
            },
        }
        merged_checkpoint = self.make_checkpoint(
            "best_model_task_private_finetune",
            epoch=self.current_epoch,
            metrics=merged_record,
            model_state=merged_state,
        )
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "selection": "shared_base_task_private_adapter_head_finetune",
            "variant_name": self.variant_name,
            "seed": self.seed,
            "outer_fold_id": self.outer_fold_id,
            "base_checkpoint": base_checkpoint_path.name,
            "base_checkpoint_sha256": base_checkpoint_sha256,
            "merged_checkpoint": merged_checkpoint.name,
            "merged_checkpoint_sha256": _sha256_file(merged_checkpoint),
            "inner_predictions": selected_csv.name,
            "inner_predictions_sha256": _sha256_file(selected_csv),
            "inner_predictions_parquet": selected_parquet.name,
            "inner_predictions_parquet_sha256": _sha256_file(selected_parquet),
            "private_modules": list(TASK_PRIVATE_MODULE_NAMES),
            "task_private_sampling_mode": self.task_private_sampling_mode,
            "private_key_sets_pairwise_disjoint": True,
            "shared_tensor_count": len(shared_keys),
            "shared_state_sha256": shared_sha256,
            "base_state_sha256": _state_dict_digest(base_state),
            "merged_state_sha256": _state_dict_digest(merged_state),
            "merged_metrics": merged_metrics,
            "prediction_equivalence": "exact_with_equal_nan",
            "tasks": task_records,
        }
        atomic_write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            checkpoint_dir / "task_private_finetune_selection.json",
        )
        self.task_private_finetune_audit = manifest
        self.best_valid_metrics = merged_record
        self.best_valid_metric = float(merged_metrics[self.best_metric])
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self) -> Dict[str, float]:
        stage_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stage_started = time.perf_counter()
        stage_completed_epochs = 0

        def record_stage_resources(*, early_stopped: bool) -> None:
            if any(
                row.get("stage") == self.current_stage
                for row in self.stage_resource_summaries
            ):
                raise AssertionError(f"Duplicate stage resource record: {self.current_stage}")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
                peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
            else:
                peak_allocated = 0
                peak_reserved = 0
            self.stage_resource_summaries.append(
                {
                    "schema_version": 1,
                    "stage": self.current_stage,
                    "started_at_utc": stage_started_at,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "wall_seconds": float(time.perf_counter() - stage_started),
                    "epochs_completed": int(stage_completed_epochs),
                    "optimizer_steps": int(self.stage_optimizer_step),
                    "early_stopped": bool(early_stopped),
                    "trainable_parameters": int(
                        sum(
                            parameter.numel()
                            for parameter in self.model.parameters()
                            if parameter.requires_grad
                        )
                    ),
                    "peak_cuda_memory_allocated_bytes_cumulative": peak_allocated,
                    "peak_cuda_memory_reserved_bytes_cumulative": peak_reserved,
                    "optimizer_groups": [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "parameter_names"
                        }
                        for row in self.optimizer_group_audit.get(
                            self.current_stage, []
                        )
                    ],
                }
            )

        batch_no = 0
        acc_nll = []
        acc_mu = []
        acc_cls = []
        acc_ord = []
        acc_l2  = []
        acc_grad_pre = []
        acc_grad_post = []
        acc_clip = []
        acc_scaler_scale = []
        acc_pcgrad_metrics: Dict[str, List[float]] = {}
        acc_task_reg = {task: [] for task in self.tasks}
        acc_task_mu = {task: [] for task in self.tasks}
        acc_expert_diversity: List[float] = []
        acc_gate_entropy = {task: [] for task in self.tasks}
        acc_pair_cosines = {name: [] for name in self.expert_pair_names}
        self.model.train()

        last_epoch = -1
        for epoch in range(self.n_epochs):
            if self.staged_finetuning and epoch == self.stage1_epochs:
                record_stage_resources(early_stopped=False)
                self._configure_training_stage("stage2")
                stage_started_at = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                stage_started = time.perf_counter()
                stage_completed_epochs = 0
            self.current_epoch = epoch
            self.current_stage_local_epoch = (
                epoch - self.stage1_epochs
                if self.staged_finetuning and self.current_stage == "stage2"
                else epoch
            )
            if self.early_stop:
                print(f"[EarlyStop] at epoch={epoch}")
                break
            last_epoch = epoch

            epoch_pbar = tqdm(
                range(self.n_steps_per_epoch),
                desc=(
                    f"{self.current_stage} epoch "
                    f"{self.current_stage_local_epoch + 1}/"
                    f"{self.stage1_epochs if self.current_stage == 'stage1' else self.stage2_epochs if self.current_stage == 'stage2' else self.n_epochs}"
                ),
                leave=True,
            )
            for step in epoch_pbar:
                self._set_step_learning_rate(self.stage_optimizer_step)
                training_progress = self._training_progress(batch_no)
                step_nll, step_mu, step_cls, step_ord = [], [], [], []
                step_task_reg: Dict[str, float] = {}
                step_task_mu: Dict[str, float] = {}
                step_pcgrad_metrics: Dict[str, float] = {}
                step_diversity_losses: List[torch.Tensor] = []
                step_gate_entropy: Dict[str, float] = {}
                step_pair_cosines = {
                    name: [] for name in self.expert_pair_names
                }
                diversity_loss_val = 0.0
                grad_pre = float("nan")
                grad_post = float("nan")
                clipped = float("nan")
                scaler_scale = float(self.scaler.get_scale())

                if self.use_pcgrad:
                    all_task_grads: Dict[str, Dict[str, torch.Tensor]] = {}
                    nan_in_step = False

                    for task in self.tasks:
                        try:
                            batch = next(self.train_iterators[task])
                        except StopIteration:
                            self.train_iterators[task] = iter(self.train_loaders[task])
                            batch = next(self.train_iterators[task])

                        (
                            graph,
                            duration_values,
                            effect_onehots,
                            media_onehots,
                            labels,
                            ghs_classes,
                            smiles_list,
                            _sample_ids,
                        ) = batch
                        graph = graph.to(self.device)
                        duration_values = duration_values.to(self.device)
                        effect_onehots = effect_onehots.to(self.device)
                        media_onehots = media_onehots.to(self.device)
                        labels = labels.to(self.device)
                        ghs_classes = ghs_classes.to(self.device)

                        self.optimizer.zero_grad()
                        with autocast(enabled=self.amp_enabled):
                            outputs = self.model(
                                graph, duration_values,
                                effect_onehots, media_onehots,
                                smiles_list=smiles_list,
                                requested_tasks=self._requested_training_tasks(task),
                            )
                            task_out = outputs[task]
                            mu = task_out["mu"]
                            regression_loss = self._regression_loss(task_out, labels)
                            mean_loss = self._mean_auxiliary_loss(task_out, labels)
                            cls_log_probs = self._ghs_soft_logprobs(mu, task)
                            cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                            task_loss = (
                                regression_loss
                                + self.lambda_mu * mean_loss
                                + self.lambda_cls * cls_loss
                            )

                            ordinal_loss = self._ordinal_loss(
                                task,
                                outputs,
                                training_progress=training_progress,
                            )
                            ordinal_loss_val = 0.0
                            if ordinal_loss is not None:
                                ordinal_loss_val = ordinal_loss.item()
                                task_loss = task_loss + self.lambda_ord * ordinal_loss

                        if not torch.isfinite(task_loss):
                            print(f"Non-finite task loss ({task}) at epoch={epoch}, step={step}, skip step")
                            nan_in_step = True
                            break

                        self.scaler.scale(task_loss).backward()

                        all_task_grads[task] = {
                            name: param.grad.clone()
                            for name, param in self.model.named_parameters()
                            if param.requires_grad and param.grad is not None
                        }
                        step_nll.append(regression_loss.item())
                        step_mu.append(mean_loss.item())
                        step_cls.append(cls_loss.item())
                        step_task_reg[task] = regression_loss.item()
                        step_task_mu[task] = mean_loss.item()
                        if ordinal_loss_val > 0:
                            step_ord.append(ordinal_loss_val)

                    if nan_in_step:
                        continue

                    if self.pcgrad_scope == "hierarchical_shared":
                        projected_grads, step_pcgrad_metrics = (
                            self._hierarchical_pcgrad_project(
                                all_task_grads,
                                optimizer_step=self.stage_optimizer_step,
                            )
                        )
                    else:
                        projected_grads = self._pcgrad_project(
                            [all_task_grads[task] for task in self.tasks]
                        )
                    grad_pre, grad_post, clipped = (
                        self._pcgrad_optimizer_step(projected_grads)
                    )
                    l2_val = self.l2_regularization().item()

                else:
                    with autocast(enabled=self.amp_enabled):
                        total_task_loss = torch.tensor(0.0, device=self.device)
                        task_objectives: Dict[str, torch.Tensor] = {}

                        for task in self.tasks:
                            try:
                                batch = next(self.train_iterators[task])
                            except StopIteration:
                                self.train_iterators[task] = iter(self.train_loaders[task])
                                batch = next(self.train_iterators[task])

                            (
                                graph,
                                duration_values,
                                effect_onehots,
                                media_onehots,
                                labels,
                                ghs_classes,
                                smiles_list,
                                _sample_ids,
                            ) = batch
                            graph = graph.to(self.device)
                            duration_values = duration_values.to(self.device)
                            effect_onehots = effect_onehots.to(self.device)
                            media_onehots = media_onehots.to(self.device)
                            labels = labels.to(self.device)
                            ghs_classes = ghs_classes.to(self.device)

                            outputs = self.model(
                                graph, duration_values,
                                effect_onehots, media_onehots,
                                smiles_list=smiles_list,
                                requested_tasks=self._requested_training_tasks(task),
                                return_mmoe_diagnostics=(
                                    self.lambda_expert_diversity > 0.0
                                ),
                            )
                            task_out = outputs[task]
                            mu = task_out["mu"]
                            regression_loss = self._regression_loss(task_out, labels)
                            mean_loss = self._mean_auxiliary_loss(task_out, labels)
                            cls_log_probs = self._ghs_soft_logprobs(mu, task)
                            cls_loss = self.cls_loss_fn(cls_log_probs, ghs_classes)
                            task_loss = (
                                regression_loss
                                + self.lambda_mu * mean_loss
                                + self.lambda_cls * cls_loss
                            )

                            ordinal_loss = self._ordinal_loss(
                                task,
                                outputs,
                                training_progress=training_progress,
                            )
                            ordinal_loss_val = 0.0
                            if ordinal_loss is not None:
                                ordinal_loss_val = ordinal_loss.item()
                                task_loss = task_loss + self.lambda_ord * ordinal_loss

                            task_objectives[task] = task_loss
                            if self.lambda_expert_diversity > 0.0:
                                diversity = task_out.get("expert_diversity_loss")
                                pair_cosines = task_out.get("expert_pair_cosines")
                                gate_entropy = task_out.get("gate_entropy")
                                if (
                                    diversity is None
                                    or pair_cosines is None
                                    or gate_entropy is None
                                    or int(pair_cosines.numel())
                                    != len(self.expert_pair_names)
                                ):
                                    raise RuntimeError(
                                        f"Incomplete MMoE diagnostics for task {task}"
                                    )
                                step_diversity_losses.append(diversity)
                                step_gate_entropy[task] = float(
                                    gate_entropy.detach().float().cpu()
                                )
                                pair_values = (
                                    pair_cosines.detach().float().cpu().tolist()
                                )
                                for name, value in zip(
                                    self.expert_pair_names, pair_values
                                ):
                                    step_pair_cosines[name].append(float(value))
                            step_nll.append(regression_loss.item())
                            step_mu.append(mean_loss.item())
                            step_cls.append(cls_loss.item())
                            step_task_reg[task] = regression_loss.item()
                            step_task_mu[task] = mean_loss.item()
                            if ordinal_loss_val > 0:
                                step_ord.append(ordinal_loss_val)

                    effective_weights = self._task_loss_weights_for_step(
                        task_objectives, epoch=epoch
                    )
                    effective_weight_sum = float(sum(effective_weights.values()))
                    total_task_loss = sum(
                        task_objectives[task] * effective_weights[task]
                        for task in self.tasks
                    ) / effective_weight_sum
                    diversity_loss = (
                        torch.stack(step_diversity_losses).mean()
                        if step_diversity_losses
                        else torch.tensor(0.0, device=self.device)
                    )
                    diversity_loss_val = float(
                        diversity_loss.detach().float().cpu()
                    )

                    if (
                        self.gradient_diagnostics_every_n_epochs > 0
                        and step == 0
                        and epoch % self.gradient_diagnostics_every_n_epochs == 0
                    ):
                        cosine_metrics = {
                            **self._shared_gradient_cosines(task_objectives),
                            **self._group_gradient_cosines(task_objectives),
                        }
                        if cosine_metrics:
                            self.logger.log_metrics(
                                metrics={
                                    "epoch": epoch,
                                    "stage": self.current_stage,
                                    "stage_epoch": self.current_stage_local_epoch,
                                    "global_step": batch_no,
                                    **cosine_metrics,
                                },
                                prefix="gradient_diagnostics",
                            )

                    l2_reg = self.l2_regularization()
                    total_loss = (
                        total_task_loss
                        + self.lambda_expert_diversity * diversity_loss
                        + l2_reg
                    )

                    if not torch.isfinite(total_loss):
                        print(f"Non-finite loss at epoch={epoch}, step={step}, skip this batch")
                        continue

                    self.optimizer.zero_grad()
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer.optimizer)
                    grad_pre_tensor = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping_norm
                    )
                    grad_pre = float(grad_pre_tensor.detach().cpu())
                    grad_post = (
                        min(grad_pre, self.gradient_clipping_norm)
                        if np.isfinite(grad_pre)
                        else grad_pre
                    )
                    clipped = float(
                        np.isfinite(grad_pre)
                        and grad_pre > self.gradient_clipping_norm
                    )
                    self.scaler.step(self.optimizer.optimizer)
                    self.scaler.update()
                    l2_val = l2_reg.item()

                acc_nll.extend(step_nll)
                acc_mu.extend(step_mu)
                acc_cls.extend(step_cls)
                acc_ord.extend(step_ord)
                acc_l2.append(l2_val)
                acc_grad_pre.append(grad_pre)
                acc_grad_post.append(grad_post)
                acc_clip.append(clipped)
                acc_scaler_scale.append(scaler_scale)
                for name, value in step_pcgrad_metrics.items():
                    acc_pcgrad_metrics.setdefault(name, []).append(value)
                for task in self.tasks:
                    if task in step_task_reg:
                        acc_task_reg[task].append(step_task_reg[task])
                    if task in step_task_mu:
                        acc_task_mu[task].append(step_task_mu[task])
                    if task in step_gate_entropy:
                        acc_gate_entropy[task].append(step_gate_entropy[task])
                if self.lambda_expert_diversity > 0.0:
                    acc_expert_diversity.append(diversity_loss_val)
                    for name, values in step_pair_cosines.items():
                        acc_pair_cosines[name].extend(values)
                batch_no += 1
                self.stage_optimizer_step += 1

                if batch_no % self.log_train_every_n_batches == 0:
                    avg_nll = np.mean(acc_nll) if acc_nll else 0.0
                    avg_mu = np.mean(acc_mu) if acc_mu else 0.0
                    avg_cls = np.mean(acc_cls) if acc_cls else 0.0
                    avg_ord = np.mean(acc_ord) if acc_ord else 0.0
                    avg_l2  = np.mean(acc_l2)  if acc_l2  else 0.0
                    avg_div = (
                        float(np.mean(acc_expert_diversity))
                        if acc_expert_diversity
                        else 0.0
                    )
                    log_info = {
                        "epoch": epoch,
                        "stage": self.current_stage,
                        "stage_epoch": self.current_stage_local_epoch,
                        "global_step": batch_no,
                        "nll_loss":   avg_nll,
                        "mu_huber_loss": avg_mu,
                        "cls_loss":   avg_cls,
                        "ord_loss":   avg_ord,
                        "l2_reg":     avg_l2,
                        "expert_diversity_loss": avg_div,
                        "total_loss": avg_nll + self.lambda_mu * avg_mu + self.lambda_cls * avg_cls + self.lambda_ord * avg_ord + self.lambda_expert_diversity * avg_div + avg_l2,
                        "lr": self.optimizer.optimizer.param_groups[0]['lr'],
                        "gradient_norm_pre_clip": float(np.nanmean(acc_grad_pre)),
                        "gradient_norm_post_clip": float(np.nanmean(acc_grad_post)),
                        "gradient_clipped_fraction": float(np.nanmean(acc_clip)),
                        "grad_scaler_scale": float(np.nanmean(acc_scaler_scale)),
                        "ordering_active": training_progress >= self.ordering_start_fraction,
                    }
                    for param_group in self.optimizer.optimizer.param_groups:
                        group_name = param_group.get("name")
                        if group_name:
                            log_info[f"lr_{group_name}"] = float(param_group["lr"])
                    for task in self.tasks:
                        if acc_task_reg[task]:
                            log_info[f"{task}_train_regression_loss"] = float(
                                np.mean(acc_task_reg[task])
                            )
                        if acc_task_mu[task]:
                            log_info[f"{task}_train_mu_huber_loss"] = float(
                                np.mean(acc_task_mu[task])
                            )
                        log_info[f"{task}_task_loss_ema"] = float(
                            self.task_loss_ema_values[task]
                        )
                        log_info[f"{task}_task_loss_weight"] = float(
                            self.current_task_loss_weights[task]
                        )
                        if acc_gate_entropy[task]:
                            log_info[f"{task}_gate_entropy"] = float(
                                np.mean(acc_gate_entropy[task])
                            )
                    for name, values in acc_pair_cosines.items():
                        if values:
                            log_info[name] = float(np.mean(values))
                    for name, values in acc_pcgrad_metrics.items():
                        finite_values = [value for value in values if math.isfinite(value)]
                        if finite_values:
                            log_info[name] = float(np.mean(finite_values))
                    self.logger.log_metrics(metrics=log_info, prefix="train")

                    acc_nll.clear(); acc_mu.clear(); acc_cls.clear(); acc_ord.clear(); acc_l2.clear()
                    acc_grad_pre.clear(); acc_grad_post.clear(); acc_clip.clear(); acc_scaler_scale.clear()
                    acc_pcgrad_metrics.clear()
                    acc_expert_diversity.clear()
                    for values in acc_pair_cosines.values():
                        values.clear()
                    for task in self.tasks:
                        acc_task_reg[task].clear(); acc_task_mu[task].clear()
                        acc_gate_entropy[task].clear()
                    epoch_pbar.set_postfix(nll=f"{avg_nll:.4f}", cls=f"{avg_cls:.4f}")

            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                self.swa_active = True
            elif self.scheduler_mode == "step":
                pass
            elif self.current_stage_local_epoch < self.warmup_epochs:
                scale = (self.current_stage_local_epoch + 1) / self.warmup_epochs
                for pg, base_lr in zip(self.optimizer.optimizer.param_groups, self.base_lrs):
                    pg['lr'] = base_lr * scale
            else:
                assert self.lr_scheduler is not None
                self.lr_scheduler.step()

            current_lr = self.optimizer.optimizer.param_groups[0]['lr']

            stage_final_epoch = (
                self.current_stage == "stage2"
                and self.current_stage_local_epoch == self.stage2_epochs - 1
            ) or (
                self.current_stage == "standard" and epoch == self.n_epochs - 1
            )
            should_validate = (
                bool(self.valid_loaders)
                and self.current_stage != "stage1"
                and (
                    (
                        self.current_stage_local_epoch + 1
                    ) % self.valid_every_n_epochs == 0
                    or stage_final_epoch
                )
            )
            if should_validate:
                valid_metrics = self.valid_step()
                self._update_taskwise_selection(valid_metrics, epoch=epoch)
                self._update_balanced_selection(valid_metrics, epoch=epoch)

                if self.checkpoint_best:
                    current_val = valid_metrics.get(self.best_metric, None)
                    if current_val is None or not np.isfinite(current_val):
                        raise FloatingPointError(
                            f"Selection metric {self.best_metric!r} is missing or non-finite: "
                            f"{current_val!r}"
                        )

                    if ((self.metric_direction == "min" and current_val < self.best_valid_metric)
                            or (self.metric_direction == "max" and current_val > self.best_valid_metric)):
                        self.logger.log_metrics(
                            metrics={**valid_metrics, "epoch": epoch},
                            prefix="best_valid",
                        )
                        self.best_valid_metrics = dict(
                            valid_metrics,
                            epoch=epoch,
                            stage=self.current_stage,
                            stage_epoch=self.current_stage_local_epoch,
                        )
                        self.best_valid_metric = current_val
                        self.make_checkpoint(
                            "best_model", epoch=epoch, metrics=self.best_valid_metrics
                        )
                        best_predictions = self.last_prediction_tables.get("inner_val")
                        if best_predictions is not None:
                            predictions_dir = self.run_dir / "predictions"
                            predictions_dir.mkdir(parents=True, exist_ok=True)
                            prediction_name = (
                                "inner_val_macro_best.csv"
                                if self.checkpoint_selection
                                in {
                                    "taskwise",
                                    "task_soup",
                                    "balanced",
                                    "task_private_finetune",
                                }
                                else "inner_val_best.csv"
                            )
                            best_predictions.to_csv(
                                predictions_dir / prediction_name, index=False
                            )
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1

                    if self.epochs_no_improve >= self.patience:
                        print(f"No improvement after {self.patience} epochs, early stopping.")
                        self.early_stop = True
                else:
                    self.best_valid_metrics = dict(
                        valid_metrics,
                        epoch=epoch,
                        stage=self.current_stage,
                        stage_epoch=self.current_stage_local_epoch,
                    )

            stage_completed_epochs = self.current_stage_local_epoch + 1

        record_stage_resources(early_stopped=self.early_stop)
        self._finalize_stage_audit()
        self.make_checkpoint(
            "last_model",
            epoch=last_epoch,
            metrics=self.best_valid_metrics,
        )
        self._finalize_taskwise_selection()
        self._finalize_task_soup_selection()
        self._finalize_balanced_selection()
        self._finalize_task_private_finetuning()
        if self.training_scope == "full_data":
            self.make_checkpoint(
                "final_model",
                epoch=last_epoch,
                metrics=self.best_valid_metrics,
            )

        if self.use_swa and self.swa_active:
            print("[SWA] 使用平均模型进行最终验证...")
            _orig_model = self.model
            self.model = self.swa_model
            swa_metrics = self.valid_step()
            self.model = _orig_model

            swa_state = {
                k[len("module."):]: v
                for k, v in self.swa_model.state_dict().items()
                if k.startswith("module.")
            }
            ckpt_dir = self.run_dir / "train" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.make_checkpoint(
                "swa_final",
                epoch=last_epoch,
                metrics=swa_metrics,
                model_state=swa_state,
            )
            print(f"[SWA] swa_final.pt saved. avg_r2={swa_metrics.get('avg_r2', 'N/A'):.4f}")
            self.logger.log_metrics(metrics=swa_metrics, prefix="swa_valid")

        if self.test_loaders:
            checkpoint_name = (
                "best_model_task_soup.pt"
                if self.checkpoint_selection == "task_soup"
                else (
                    "best_model_balanced.pt"
                    if self.checkpoint_selection == "balanced"
                    else (
                        "best_model_task_private_finetune.pt"
                        if self.checkpoint_selection == "task_private_finetune"
                        else (
                            "best_model.pt"
                            if self.checkpoint_best
                            else "last_model.pt"
                        )
                    )
                )
            )
            checkpoint_path = self.run_dir / "train" / "checkpoints" / checkpoint_name
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Selected checkpoint is missing: {checkpoint_path}")
            selected = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(selected["model"])
            outer_metrics = self.test_step()
            outer_predictions = self.last_prediction_tables["outer_test"]
            outer_predictions.to_csv(
                self.run_dir / "predictions" / "outer_test_predictions.csv",
                index=False,
            )
            self.logger.log_metrics(metrics=outer_metrics, prefix="outer_test_final")

            success_payload = {
                "schema_version": 1,
                "status": "completed",
                "variant_name": self.variant_name,
                "model_kind": self.model_kind,
                "regression_mode": self.regression_mode,
                "cv_protocol": self.cv_protocol,
                "outer_fold_id": self.outer_fold_id,
                "seed": self.seed,
                "selected_checkpoint": checkpoint_name,
                "selection_metric": self.best_metric,
                "selection_direction": self.metric_direction,
                "outer_metrics": outer_metrics,
            }
            atomic_write_text(
                json.dumps(success_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                self.run_dir / "_SUCCESS.json",
            )

        return self.best_valid_metrics

    def close(self):
        self.logger.close()
