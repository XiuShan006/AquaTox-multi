from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class LoggerBase(ABC):

    def __init__(self, logdir: str | Path):
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], prefix: str):
        ...

    @abstractmethod
    def log_code(self, source_path: str | Path):
        ...

    @abstractmethod
    def log_to_file(self, content: Any, name: str, type: str = "txt"):
        ...

    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def restart(self):
        ...
