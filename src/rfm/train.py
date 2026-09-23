import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

dir_path = Path(__file__).resolve().parent
project_root = dir_path.parent
for import_path in (project_root, dir_path):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

import gin
import torch
from gin_config import get_time_stamp
from rfm.artifacts import atomic_write_text
from rfm.project_layout import AQUATOX_DATA_ROOT
from torch_geometric import seed_everything
from trainer import *


FULL_TRAIN_DATASET_SCOPES = (
    "build_train_dataset_fish_ec50",
    "build_train_dataset_fish_ec10",
    "build_train_dataset_aquatic_invertebrates_ec50",
    "build_train_dataset_aquatic_invertebrates_ec10",
    "build_train_dataset_algae_ec50",
    "build_train_dataset_algae_ec10",
)


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_resource_usage(device: torch.device) -> tuple[int, int, str | None]:
    if device.type != "cuda":
        return 0, 0, None
    torch.cuda.synchronize(device)
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
        torch.cuda.get_device_name(device),
    )


if __name__ == "__main__":
    training_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    training_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=str(project_root / "rfm" / "configs" / "rfm_train.gin"),
    )
    parser.add_argument("--cv-protocol", choices=("scaffold", "molecule"), default="scaffold")
    parser.add_argument("--outer-fold-id", type=int, default=0)
    parser.add_argument(
        "--training-scope", choices=("cv", "full_data"), default="cv"
    )
    parser.add_argument(
        "--dataset-layer",
        choices=("common_intersection", "curated_full"),
        default="common_intersection",
    )
    parser.add_argument(
        "--data-root",
        dest="aquatox_data_root",
        type=Path,
        default=AQUATOX_DATA_ROOT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed; defaults to 20260825 + outer_fold_id.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument("--experiment-signature", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--n-steps-per-epoch", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--preprocessed-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--finetune-checkpoint", type=Path, default=None)
    parser.add_argument("--finetune-checkpoint-sha256", default=None)
    parser.add_argument(
        "--finetune-source-variant", default="pretrained_reference"
    )

    args = parser.parse_args()
    if args.training_scope == "cv" and args.outer_fold_id not in range(5):
        parser.error("--outer-fold-id must be in [0, 4] for --training-scope=cv")
    if args.training_scope == "full_data" and args.outer_fold_id != -1:
        parser.error("--outer-fold-id must be -1 for --training-scope=full_data")
    if (args.finetune_checkpoint is None) != (
        args.finetune_checkpoint_sha256 is None
    ):
        parser.error(
            "--finetune-checkpoint and --finetune-checkpoint-sha256 must be provided together"
        )
    if args.finetune_checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", args.finetune_checkpoint_sha256
    ):
        parser.error("--finetune-checkpoint-sha256 must be 64 lowercase hex characters")
    for option_name, value in (
        ("--n-epochs", args.n_epochs),
        ("--stage1-epochs", args.stage1_epochs),
    ):
        if value is not None and value < 1:
            parser.error(f"{option_name} must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")

    seed = (
        args.seed
        if args.seed is not None
        else (20260825 if args.training_scope == "full_data" else 20260825 + args.outer_fold_id)
    )
    config = str(Path(args.cfg).expanduser().resolve())
    aquatox_data_entry = args.aquatox_data_root.expanduser().absolute()
    aquatox_root = args.aquatox_data_root.expanduser().resolve()
    model_csv = aquatox_root / "data" / f"{args.dataset_layer}_model.csv"
    role_manifest = aquatox_root / "manifests" / "cv_roles.csv.gz"
    for required_path in (Path(config), model_csv, role_manifest):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    seed_everything(seed)
    config_name = Path(config).stem
    variant_name = args.variant_name or config_name
    run_name = (
        f"{variant_name}/{args.dataset_layer}/{args.cv_protocol}/"
        f"fold_{args.outer_fold_id}"
    )
    if args.run_dir is None:
        run_name = f"{run_name}/{get_time_stamp()}"
    bindings = [
        f"run_name={json.dumps(run_name)}",
        f"YieldDataset.cv_protocol={json.dumps(args.cv_protocol)}",
        f"YieldDataset.outer_fold_id={args.outer_fold_id}",
        f"YieldDataset.dataset_layer={json.dumps(args.dataset_layer)}",
        f"YieldDataset.file_path={json.dumps(str(model_csv))}",
        f"YieldDataset.manifest_path={json.dumps(str(role_manifest))}",
        f"YieldTrainer.seed={seed}",
        f"YieldTrainer.variant_name={json.dumps(variant_name)}",
        f"YieldTrainer.device={json.dumps(args.device)}",
        f"YieldTrainer.training_scope={json.dumps(args.training_scope)}",
        f"YieldGNN.initialization_seed={seed}",
        f"SingleTaskYieldGNN.initialization_seed={seed}",
    ]
    if args.training_scope == "full_data":
        bindings.extend(
            [
                *(f"{scope}/YieldDataset.split_role='full_train'" for scope in FULL_TRAIN_DATASET_SCOPES),
                "YieldTrainer.valid_datasets={}",
                "YieldTrainer.test_datasets={}",
                "YieldTrainer.checkpoint_best=False",
            ]
        )
    if args.run_dir is not None:
        bindings.append(f"run_dir={json.dumps(str(args.run_dir.expanduser().resolve()))}")
    if args.task is not None:
        bindings.extend(
            [
                f"YieldTrainer.tasks={json.dumps([args.task])}",
                f"SingleTaskYieldGNN.task={json.dumps(args.task)}",
            ]
        )
    if args.n_steps_per_epoch is not None:
        if args.n_steps_per_epoch < 1:
            raise ValueError("--n-steps-per-epoch must be positive")
        bindings.append(f"YieldTrainer.n_steps_per_epoch={args.n_steps_per_epoch}")
    if args.n_epochs is not None:
        bindings.append(f"YieldTrainer.n_epochs={args.n_epochs}")
    if args.stage1_epochs is not None:
        bindings.append(f"YieldTrainer.stage1_epochs={args.stage1_epochs}")
    if args.num_workers is not None:
        bindings.append(f"YieldTrainer.num_workers={args.num_workers}")
    if args.finetune_checkpoint is not None:
        bindings.extend(
            [
                "YieldTrainer.staged_finetuning=True",
                (
                    "YieldTrainer.finetune_checkpoint_path="
                    f"{json.dumps(str(args.finetune_checkpoint.expanduser().resolve()))}"
                ),
                (
                    "YieldTrainer.finetune_checkpoint_sha256="
                    f"{json.dumps(args.finetune_checkpoint_sha256)}"
                ),
                (
                    "YieldTrainer.finetune_source_variant="
                    f"{json.dumps(args.finetune_source_variant)}"
                ),
            ]
        )
    if args.preprocessed_dir is not None:
        preprocessed_dir = str(args.preprocessed_dir.expanduser().resolve())
        bindings.extend(
            [
                f"YieldDataset.preprocessed_dir={json.dumps(preprocessed_dir)}",
                f"YieldTrainer.preprocessed_dir={json.dumps(preprocessed_dir)}",
            ]
        )
    gin.parse_config_files_and_bindings([config], bindings=bindings)
    trainer = YieldTrainer()
    _reset_cuda_peak_memory(trainer.device)
    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    initial_trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "status": "training",
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "project_root": str(project_root),
        "run_dir": str(trainer.run_dir.resolve()),
        "variant_name": variant_name,
        "experiment_signature": args.experiment_signature,
        "task": args.task,
        "training_scope": args.training_scope,
        "dataset_layer": args.dataset_layer,
        "cv_protocol": args.cv_protocol,
        "outer_fold_id": args.outer_fold_id,
        "seed": seed,
        "n_steps_per_epoch": trainer.n_steps_per_epoch,
        "num_workers": trainer.num_workers,
        "config_path": config,
        "config_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        "aquatox_data_entry": str(aquatox_data_entry),
        "aquatox_data_root": str(aquatox_root),
        "model_csv_path": str(model_csv),
        "cv_roles_path": str(role_manifest),
        "preprocessed_dir": str(trainer.preprocessed_dir.resolve()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "dataset_sha256": trainer.train_datasets[trainer.tasks[0]].dataset_sha256,
        "manifest_sha256": trainer.train_datasets[trainer.tasks[0]].manifest_sha256,
        "finetune_checkpoint_path": (
            str(args.finetune_checkpoint.expanduser().resolve())
            if args.finetune_checkpoint is not None
            else None
        ),
        "finetune_checkpoint_sha256": args.finetune_checkpoint_sha256,
    }
    atomic_write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "run_manifest.json",
    )
    atomic_write_text(
        gin.operative_config_str(), trainer.run_dir / "operative_config.gin"
    )
    atomic_write_text(gin.config_str(), trainer.run_dir / "config.gin")
    if trainer.transfer_audit is not None:
        atomic_write_text(
            json.dumps(trainer.transfer_audit, indent=2, sort_keys=True) + "\n",
            trainer.run_dir / "transfer_audit.json",
        )
    atomic_write_text(
        json.dumps(trainer.optimizer_group_audit, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "optimizer_group_audit.json",
    )
    trainer.logger.log_code(str(project_root))
    trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
    trainer.logger.log_to_file(gin.config_str(), "config")
    seed_everything(seed)
    training_metrics = trainer.train()
    atomic_write_text(
        json.dumps(trainer.optimizer_group_audit, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "optimizer_group_audit.json",
    )
    atomic_write_text(
        json.dumps(trainer.stage_audit, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "stage_audit.json",
    )
    if trainer.task_private_finetune_audit:
        atomic_write_text(
            json.dumps(
                trainer.task_private_finetune_audit, indent=2, sort_keys=True
            )
            + "\n",
            trainer.run_dir / "task_private_finetune_audit.json",
        )
    peak_allocated, peak_reserved, cuda_device_name = _cuda_resource_usage(
        trainer.device
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    training_resources = {
        "schema_version": 1,
        "status": "completed",
        "stage": "training",
        "started_at_utc": training_started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": float(time.perf_counter() - training_started),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "initial_trainable_parameters": int(initial_trainable_parameters),
        "stage_resources": trainer.stage_resource_summaries,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "cuda_device_name": cuda_device_name,
    }
    atomic_write_text(
        json.dumps(training_resources, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "resources" / "training_resources.json",
    )
    run_manifest.update(
        {
            "status": "completed",
            "finished_at_utc": training_resources["finished_at_utc"],
            "training_metrics": training_metrics,
            "training_resources_path": str(
                (trainer.run_dir / "resources" / "training_resources.json").resolve()
            ),
        }
    )
    atomic_write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        trainer.run_dir / "run_manifest.json",
    )
    trainer.close()
