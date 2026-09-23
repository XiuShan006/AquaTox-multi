
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
AQUATOX_DATA_ROOT = DATA_ROOT / "aquatox"
PREPROCESSED_ROOT = PROJECT_ROOT / "preprocessed_graphs_local"
AQUATOX_PREPROCESSED_ROOT = PREPROCESSED_ROOT / "aquatox"
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "rfm_train"
OUTPUT_ROOT = PROJECT_ROOT / "output"


def require_child_path(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    parent_resolved = parent.expanduser().resolve()
    try:
        relative = resolved.relative_to(parent_resolved)
    except ValueError as error:
        raise ValueError(f"{label} must be under {parent_resolved}: {resolved}") from error
    if relative == Path("."):
        raise ValueError(f"{label} must name a run below {parent_resolved}")
    return resolved
