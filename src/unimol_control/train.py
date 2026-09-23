import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

dir_path = Path(__file__).parent.absolute()
sys.path.append(str(dir_path))

import gin
from gin_config import get_time_stamp
from torch_geometric import seed_everything
from trainer import *


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
    os.replace(temporary, path)

if __name__ == "__main__":
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-fold", type=int, choices=range(5), default=0)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Optional path below experiments/. Use this to make a CV run "
            "directory deterministic and easy to audit."
        ),
    )
    parser.add_argument(
        "--split-protocol",
        choices=("scaffold", "molecule"),
        default="scaffold",
    )

    args = parser.parse_args()
    seed = args.seed
    config = args.cfg
    if config is None:
        parser.error("--cfg is required")
    config_path = Path(config).expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")

    seed_everything(seed)
    config_name = Path(config).stem
    if args.run_name is None:
        run_name = (
            f"{config_name}/{get_time_stamp()}_"
            f"{args.split_protocol}_fold_{args.outer_fold}"
        )
    else:
        requested_run = Path(args.run_name)
        if (
            requested_run == Path(".")
            or not requested_run.parts
            or requested_run.is_absolute()
            or "." in requested_run.parts
            or ".." in requested_run.parts
        ):
            parser.error("--run-name must be a relative path below experiments/")
        run_name = requested_run.as_posix().strip("/")
        if not run_name:
            parser.error("--run-name must not be empty")
    gin.parse_config_files_and_bindings(
        [config],
        bindings=[
            f'run_name="{run_name}"',
            f'YieldDataset.outer_fold_id={args.outer_fold}',
            f'YieldDataset.split_protocol="{args.split_protocol}"',
            f'YieldTrainer.seed={seed}',
            f'YieldTrainer.config_path="{config_path.as_posix()}"',
        ],
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    trainer = None
    result = None
    status = "failed"
    error_record = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer = YieldTrainer()
        trainer.logger.log_code(dir_path)
        trainer.logger.log_to_file(gin.operative_config_str(), "operative_config")
        trainer.logger.log_to_file(gin.config_str(), "config")
        result = trainer.train()
        status = "completed"
    except BaseException as error:
        error_record = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        wall_seconds = time.perf_counter() - started
        if trainer is not None:
            try:
                trainer.close()
            finally:
                checkpoint_dir = trainer.run_dir / "train" / "checkpoints"
                checkpoints = {
                    path.stem: _file_record(path)
                    for path in sorted(checkpoint_dir.glob("*.pt"))
                    if path.is_file()
                }
                peak_allocated = (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available()
                    else 0
                )
                peak_reserved = (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available()
                    else 0
                )
                metadata = {
                    "schema_version": 1,
                    "status": status,
                    "variant_name": trainer.variant_name,
                    "protocol": args.split_protocol,
                    "outer_fold_id": args.outer_fold,
                    "seed": seed,
                    "started_at_utc": started_at.isoformat(),
                    "finished_at_utc": finished_at.isoformat(),
                    "training_wall_seconds": wall_seconds,
                    "training_peak_cuda_memory_allocated_bytes": peak_allocated,
                    "training_peak_cuda_memory_reserved_bytes": peak_reserved,
                    "total_parameters": trainer.total_parameters,
                    "trainable_parameters": trainer.trainable_parameters,
                    "selection_metric": trainer.best_metric,
                    "metric_direction": trainer.metric_direction,
                    "validation_granularity": trainer.validation_granularity,
                    "best_valid_metrics": result,
                    "data_contract": trainer.data_contract,
                    "pretrained_checkpoint": trainer.pretrained_checkpoint,
                    "config": _file_record(config_path),
                    "checkpoints": checkpoints,
                    "error": error_record,
                }
                _write_json_atomic(trainer.run_dir / "run_metadata.json", metadata)
