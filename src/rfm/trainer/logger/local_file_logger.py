
from __future__ import annotations

import csv
import json
import math
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import gin

from rfm.artifacts import atomic_write_text

from .logger_base import LoggerBase


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "item"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@gin.configurable()
class LocalFileLogger(LoggerBase):

    def __init__(self, logdir: str | Path):
        super().__init__(logdir)
        self.metrics_path = self.logdir / "metrics.jsonl"
        self._handle = self.metrics_path.open("a", encoding="utf-8")
        with self.metrics_path.open("r", encoding="utf-8") as handle:
            self._event_index = sum(1 for _ in handle)
        self._closed = False

    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        record = {
            "event_index": self._event_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prefix": str(prefix),
            **{str(key): _json_value(value) for key, value in metrics.items()},
        }
        self._handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
        self._handle.flush()
        self._event_index += 1

    def log_code(self, source_path: str | Path):
        atomic_write_text(str(Path(source_path).resolve()) + "\n", self.logdir / "source_root.txt")

    def log_to_file(self, content: Any, name: str, type: str = "txt"):
        if type == "json":
            value = content if isinstance(content, str) else json.dumps(content, indent=2)
            atomic_write_text(value, self.logdir / f"{name}.json")
        elif type == "txt":
            atomic_write_text(str(content), self.logdir / f"{name}.txt")
        elif type == "to_pickle":
            target = self.logdir / f"{name}.pkl"
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=self.logdir
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                with temporary.open("wb") as handle:
                    pickle.dump(content, handle)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            raise ValueError(f"Unknown log artifact type: {type}")

    def log_config(self, config: Dict[str, Any]):
        self.log_to_file(config, "runtime_config", type="json")

    def _publish_csv(self) -> None:
        records = [
            json.loads(line)
            for line in self.metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            return
        fieldnames: list[str] = []
        seen: set[str] = set()
        for record in records:
            for key in record:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        target = self.logdir / "metrics.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.logdir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self):
        if self._closed:
            return
        self._handle.close()
        self._publish_csv()
        self._closed = True

    def restart(self):
        self.close()
        self._handle = self.metrics_path.open("a", encoding="utf-8")
        self._closed = False
