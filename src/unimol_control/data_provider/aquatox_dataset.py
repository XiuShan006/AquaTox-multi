from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import pandas as pd
import torch


SPLIT_TO_ROLE = {
    "train": "outer_train",
    "valid": "inner_val",
    "test": "outer_test",
    "outer_train": "outer_train",
    "inner_val": "inner_val",
    "outer_test": "outer_test",
}
VALID_PROTOCOLS = frozenset({"scaffold", "molecule"})
VALID_ROLES = frozenset({"outer_train", "inner_val", "outer_test"})

DATA_COLUMNS = frozenset(
    {
        "sample_id",
        "Standardized_SMILES",
        "log10_mgperL",
        "Duration_Value",
        "effect",
        "endpoint",
        "task",
        "molecule_id",
        "split_parent_id",
    }
)
ROLE_COLUMNS = frozenset(
    {"protocol", "outer_fold_id", "sample_id", "split_group_id", "role"}
)
CONFORMER_AUDIT_COLUMNS = frozenset(
    {
        "molecule_id",
        "conformer_status",
        "candidate_count",
        "converged_count",
        "mmff_energy",
        "cache_relpath",
    }
)


def _resolved_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


@lru_cache(maxsize=8)
def _read_csv(path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def load_aquatox_fold(
    file_path: str | Path,
    roles_path: str | Path,
    *,
    protocol: str,
    outer_fold_id: int,
    sep: str = ",",
) -> pd.DataFrame:
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(
            f"Unsupported protocol {protocol!r}; expected one of {sorted(VALID_PROTOCOLS)}"
        )
    if not isinstance(outer_fold_id, (int, np.integer)) or not 0 <= int(outer_fold_id) <= 4:
        raise ValueError("outer_fold_id must be an integer in [0, 4]")

    data_file = _resolved_file(file_path, "release model table")
    role_file = _resolved_file(roles_path, "nested-CV role manifest")
    data = _read_csv(str(data_file), sep)
    roles = _read_csv(str(role_file), ",")

    missing_data = sorted(DATA_COLUMNS - set(data.columns))
    missing_roles = sorted(ROLE_COLUMNS - set(roles.columns))
    if missing_data:
        raise ValueError(f"release model table is missing columns: {missing_data}")
    if missing_roles:
        raise ValueError(f"Role manifest is missing columns: {missing_roles}")
    if data["sample_id"].isna().any() or data["sample_id"].duplicated().any():
        raise ValueError("release model table must contain unique, non-null sample_id values")
    null_counts = data[list(DATA_COLUMNS)].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(
            f"release model table contains required-field nulls: {null_counts.to_dict()}"
        )

    selected_roles = roles[
        (roles["protocol"] == protocol)
        & (roles["outer_fold_id"] == int(outer_fold_id))
    ][["sample_id", "split_group_id", "role"]].copy()
    if selected_roles.empty:
        raise ValueError(
            f"No role rows for protocol={protocol!r}, outer_fold_id={outer_fold_id}"
        )
    if (
        selected_roles["sample_id"].isna().any()
        or selected_roles["sample_id"].duplicated().any()
    ):
        raise ValueError("Selected role manifest rows must be one-to-one by sample_id")
    unexpected_roles = sorted(set(selected_roles["role"].dropna()) - VALID_ROLES)
    if unexpected_roles:
        raise ValueError(f"Unexpected nested-CV roles: {unexpected_roles}")

    fold = data.merge(
        selected_roles,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    missing_role_ids = fold.loc[fold["role"].isna(), "sample_id"]
    if not missing_role_ids.empty:
        preview = missing_role_ids.head(5).tolist()
        raise ValueError(
            f"Role manifest does not cover {len(missing_role_ids)} model samples; examples={preview}"
        )
    if set(fold["role"]) != VALID_ROLES:
        raise ValueError(
            "Selected model table must contain outer_train, inner_val, and outer_test rows"
        )

    isolation_columns = {
        "molecule_id": "molecules",
        "split_parent_id": "split parents",
        "split_group_id": "split groups",
    }
    for column, description in isolation_columns.items():
        role_counts = fold.groupby(column, dropna=False)["role"].nunique()
        leaking = role_counts[role_counts > 1]
        if not leaking.empty:
            raise ValueError(
                f"Found {len(leaking)} {description} assigned to multiple roles"
            )
    return fold


def select_aquatox_split(
    file_path: str | Path,
    roles_path: str | Path,
    *,
    task: str,
    split: str,
    protocol: str,
    outer_fold_id: int,
    sep: str = ",",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if split not in SPLIT_TO_ROLE:
        raise ValueError(
            f"Unsupported split {split!r}; expected one of {sorted(SPLIT_TO_ROLE)}"
        )
    fold = load_aquatox_fold(
        file_path,
        roles_path,
        protocol=protocol,
        outer_fold_id=outer_fold_id,
        sep=sep,
    )
    task_rows = fold[fold["task"] == task]
    if task_rows.empty:
        available = sorted(fold["task"].unique().tolist())
        raise ValueError(f"No data for task {task!r}; available tasks={available}")

    role = SPLIT_TO_ROLE[split]
    selected = task_rows[task_rows["role"] == role].reset_index(drop=True)
    fit_rows = task_rows[task_rows["role"] == "outer_train"].reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"No rows for task={task!r}, protocol={protocol!r}, "
            f"outer_fold_id={outer_fold_id}, role={role!r}"
        )
    if fit_rows.empty:
        raise ValueError(f"No outer_train preprocessing rows for task {task!r}")
    return selected, fit_rows


def conformer_path(conformer_root: str | Path, molecule_id: str) -> Path:
    root = Path(conformer_root).expanduser().resolve()
    if not molecule_id.startswith("mol_") or len(molecule_id) < 6:
        raise ValueError(f"Invalid molecule_id: {molecule_id!r}")
    return root / f"mol_{molecule_id[4:6]}" / f"{molecule_id}.npz"


def validate_aquatox_mmff94_audit(
    audit_path: str | Path,
    molecule_ids: Iterable[str],
    *,
    conformer_root: str | Path,
) -> dict[str, float]:
    audit_file = _resolved_file(audit_path, "release MMFF94 conformer audit")
    root = Path(conformer_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"release conformer directory not found: {root}")

    audit = _read_csv(str(audit_file), ",")
    missing_columns = sorted(CONFORMER_AUDIT_COLUMNS - set(audit.columns))
    if missing_columns:
        raise ValueError(f"Conformer audit is missing columns: {missing_columns}")
    if audit["molecule_id"].isna().any() or audit["molecule_id"].duplicated().any():
        raise ValueError("Conformer audit must contain unique, non-null molecule_id values")

    requested_ids = sorted({str(molecule_id) for molecule_id in molecule_ids})
    if not requested_ids:
        raise ValueError("At least one molecule_id is required for MMFF94 audit validation")
    indexed = audit.set_index("molecule_id", drop=False)
    missing_ids = sorted(set(requested_ids) - set(indexed.index))
    if missing_ids:
        raise ValueError(
            f"Conformer audit is missing {len(missing_ids)} requested molecules; "
            f"examples={missing_ids[:5]}"
        )

    selected = indexed.loc[requested_ids].copy()
    unsupported = selected[selected["conformer_status"] != "success"]
    if not unsupported.empty:
        counts = unsupported["conformer_status"].value_counts().to_dict()
        raise ValueError(
            "Requested release molecules are not all converged MMFF94 conformers: "
            f"{counts}"
        )

    energies = pd.to_numeric(selected["mmff_energy"], errors="coerce")
    candidate_counts = pd.to_numeric(selected["candidate_count"], errors="coerce")
    converged_counts = pd.to_numeric(selected["converged_count"], errors="coerce")
    energy_values = energies.to_numpy(dtype=float)
    candidate_values = candidate_counts.to_numpy(dtype=float)
    converged_values = converged_counts.to_numpy(dtype=float)
    invalid_numeric = (
        ~np.isfinite(energy_values)
        | ~np.isfinite(candidate_values)
        | ~np.isfinite(converged_values)
        | (candidate_values <= 0)
        | (converged_values <= 0)
        | (converged_values > candidate_values)
    )
    if invalid_numeric.any():
        bad_ids = selected.index[invalid_numeric].tolist()
        raise ValueError(
            "MMFF94 audit contains invalid energy/candidate metadata for "
            f"{len(bad_ids)} molecules; examples={bad_ids[:5]}"
        )

    bad_paths = []
    for molecule_id, relative_path in selected["cache_relpath"].items():
        expected = conformer_path(root, molecule_id)
        recorded = (root / str(relative_path)).resolve()
        if recorded != expected or not expected.is_file():
            bad_paths.append(molecule_id)
    if bad_paths:
        raise ValueError(
            "MMFF94 audit/cache path mismatch for "
            f"{len(bad_paths)} molecules; examples={bad_paths[:5]}"
        )

    return {molecule_id: float(energies.loc[molecule_id]) for molecule_id in requested_ids}


def load_aquatox_unimol_input(
    path: str | Path,
    dictionary: Any,
    *,
    expected_smiles: str | None = None,
    expected_mmff_energy: float | None = None,
    max_atoms: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conformer_file = _resolved_file(path, "release conformer")
    with np.load(conformer_file, allow_pickle=False) as payload:
        required = {
            "atomic_numbers",
            "coordinates",
            "energy",
            "model_smiles",
            "rdkit_version",
            "seed",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"{conformer_file} is missing arrays: {missing}")
        atomic_numbers = np.asarray(payload["atomic_numbers"], dtype=np.int64)
        coordinates = np.asarray(payload["coordinates"], dtype=np.float32)
        model_smiles = str(np.asarray(payload["model_smiles"]).reshape(-1)[0])
        energy_values = np.asarray(payload["energy"], dtype=np.float64).reshape(-1)
        rdkit_versions = np.asarray(payload["rdkit_version"]).reshape(-1)
        seed_values = np.asarray(payload["seed"]).reshape(-1)

    if atomic_numbers.ndim != 1 or coordinates.shape != (len(atomic_numbers), 3):
        raise ValueError(
            f"Invalid atom/coordinate shapes in {conformer_file}: "
            f"{atomic_numbers.shape}, {coordinates.shape}"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"Non-finite coordinates in {conformer_file}")
    if energy_values.size != 1 or not np.isfinite(energy_values[0]):
        raise ValueError(f"Invalid MMFF94 energy in {conformer_file}")
    if rdkit_versions.size != 1 or not str(rdkit_versions[0]).strip():
        raise ValueError(f"Invalid RDKit version metadata in {conformer_file}")
    if (
        seed_values.size != 1
        or not np.issubdtype(seed_values.dtype, np.integer)
        or int(seed_values[0]) < 0
    ):
        raise ValueError(f"Invalid conformer seed metadata in {conformer_file}")
    if expected_mmff_energy is not None and not np.isclose(
        energy_values[0], expected_mmff_energy, rtol=1e-10, atol=1e-10
    ):
        raise ValueError(
            f"MMFF94 energy mismatch for {conformer_file.name}: "
            f"audit={expected_mmff_energy}, npz={energy_values[0]}"
        )
    if expected_smiles is not None and model_smiles != expected_smiles:
        raise ValueError(
            f"Conformer SMILES mismatch for {conformer_file.name}: "
            f"expected {expected_smiles!r}, found {model_smiles!r}"
        )

    keep = atomic_numbers != 1
    atomic_numbers = atomic_numbers[keep]
    coordinates = coordinates[keep]
    if not 0 < len(atomic_numbers) <= max_atoms:
        raise ValueError(
            f"Heavy-atom count {len(atomic_numbers)} outside [1, {max_atoms}] "
            f"for {conformer_file.name}"
        )

    from rdkit import Chem

    periodic_table = Chem.GetPeriodicTable()
    atom_symbols = np.asarray(
        [periodic_table.GetElementSymbol(int(number)) for number in atomic_numbers]
    )
    atom_vec = torch.as_tensor(dictionary.vec_index(atom_symbols), dtype=torch.long)
    atom_vec = torch.cat(
        [
            torch.tensor([dictionary.bos()], dtype=torch.long),
            atom_vec,
            torch.tensor([dictionary.eos()], dtype=torch.long),
        ]
    )

    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float32)
    coordinate_tensor = torch.cat(
        [torch.zeros((1, 3)), coordinate_tensor, torch.zeros((1, 3))], dim=0
    )
    dist = torch.cdist(coordinate_tensor, coordinate_tensor)
    edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
    return atom_vec, dist, edge_type
