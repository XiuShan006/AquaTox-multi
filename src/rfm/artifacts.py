
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import torch


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    return Path(temporary)


def atomic_joblib_dump(value: Any, target: str | Path) -> None:
    target_path = Path(target)
    temporary = _temporary_path(target_path)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(value: Any, target: str | Path) -> None:
    target_path = Path(target)
    temporary = _temporary_path(target_path)
    try:
        torch.save(value, temporary)
        os.replace(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(text: str, target: str | Path) -> None:
    target_path = Path(target)
    temporary = _temporary_path(target_path)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target_path)
    finally:
        temporary.unlink(missing_ok=True)
