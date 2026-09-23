from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dgl
import gin
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from rfm.featurizers import ReactionFeaturizer
from rfm.artifacts import atomic_joblib_dump


ALLOWED_PROTOCOLS = frozenset({"scaffold", "molecule"})
CV_SPLIT_ROLES = frozenset({"outer_train", "inner_val", "outer_test"})
ALLOWED_SPLIT_ROLES = frozenset({*CV_SPLIT_ROLES, "full_train"})
ALLOWED_DATASET_LAYERS = frozenset({"curated_full", "common_intersection"})
FEATURE_CACHE_VERSION = "fusion_graph_v1"


@lru_cache(maxsize=16)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_token(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


@gin.configurable()
class YieldDataset(Dataset):

    ALL_EFFECTS = ("DVP", "GRO", "ITX", "MOR", "MPH", "POP", "REP")
    REQUIRED_DATA_COLUMNS = (
        "sample_id",
        "Standardized_SMILES",
        "log10_mgperL",
        "Duration_Value",
        "effect",
        "media_type",
        "endpoint",
        "task",
        "molecule_id",
        "split_parent_id",
        "aggregation_n",
    )
    REQUIRED_ROLE_COLUMNS = (
        "protocol",
        "outer_fold_id",
        "sample_id",
        "split_group_id",
        "role",
    )

    def __init__(
        self,
        task: str,
        split_role: str,
        preprocessed_dir: str,
        file_path: str,
        manifest_path: str,
        cv_protocol: str,
        outer_fold_id: int,
        dataset_layer: str,
        sep: str = ",",
        contrastive: bool = False,
        unknown_token: str = "<UNK>",
    ):
        self.task = str(task)
        self.split_role = str(split_role)
        self.split_name = self.split_role
        self.cv_protocol = str(cv_protocol)
        self.outer_fold_id = int(outer_fold_id)
        self.dataset_layer = str(dataset_layer)
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.file_path = Path(file_path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.contrastive = bool(contrastive)
        self.group_k_consecutive = 2 if self.contrastive else 1
        self.unknown_token = unknown_token

        self._validate_configuration()
        self.dataset_sha256 = _sha256(self.file_path)
        self.manifest_sha256 = _sha256(self.manifest_path)
        data_df = self._load_frozen_slice(sep=sep)

        self.sample_ids = data_df["sample_id"].astype(str).tolist()
        self.molecule_ids = data_df["molecule_id"].astype(str).tolist()
        self.split_parent_ids = data_df["split_parent_id"].astype(str).tolist()
        self.split_group_ids = data_df["split_group_id"].astype(str).tolist()
        self.smiles_list = data_df["Standardized_SMILES"].astype(str).tolist()
        self.labels_list = data_df["log10_mgperL"].astype(float).tolist()
        self.duration_list = data_df["Duration_Value"].astype(float).tolist()
        self.effect_list = data_df["effect"].astype(str).tolist()
        self.media_list = data_df["media_type"].astype(str).tolist()

        self._validate_structures()
        self._build_condition_features()
        self._fit_or_load_scalers()

        self.ghs_classes = np.asarray(
            [self._ghs_class(value, self.task) for value in self.labels_list],
            dtype=np.int64,
        )

        self.cache_namespace = (
            f"{self.dataset_layer}_{self.dataset_sha256[:12]}_{FEATURE_CACHE_VERSION}"
        )
        self._graph_dir = (
            self.preprocessed_dir / "graphs" / self.cache_namespace
            if self.preprocessed_dir
            else None
        )
        slice_token = _stable_token(
            f"{self.cache_namespace}|{self.manifest_sha256}|{self.cv_protocol}|"
            f"{self.outer_fold_id}|{self.split_role}|{self.task}"
        )
        self._mapping_path = (
            self.preprocessed_dir / f"smiles_to_path_{slice_token}.json"
            if self.preprocessed_dir
            else None
        )

        self.smiles_to_path: Optional[Dict[str, str]] = None
        self.graph_mapping_source: Optional[str] = None
        self._initialize_preprocessed_mapping()

        self.featurizer = None if self.is_preprocessed() else ReactionFeaturizer()
        self._graph_cache: Dict[str, dgl.DGLGraph] = {}

        print(
            f"[Dataset] layer={self.dataset_layer} protocol={self.cv_protocol} "
            f"outer_fold={self.outer_fold_id} role={self.split_role} "
            f"task={self.task} samples={len(self.sample_ids)} "
            f"molecules={len(set(self.molecule_ids))} "
            f"graph_source={self.graph_mapping_source or 'runtime'}"
        )

    def _validate_configuration(self) -> None:
        if self.cv_protocol not in ALLOWED_PROTOCOLS:
            raise ValueError(
                f"cv_protocol must be one of {sorted(ALLOWED_PROTOCOLS)}, "
                f"got {self.cv_protocol!r}"
            )
        if self.split_role not in ALLOWED_SPLIT_ROLES:
            raise ValueError(
                f"split_role must be one of {sorted(ALLOWED_SPLIT_ROLES)}, "
                f"got {self.split_role!r}"
            )
        if self.dataset_layer not in ALLOWED_DATASET_LAYERS:
            raise ValueError(
                f"dataset_layer must be one of {sorted(ALLOWED_DATASET_LAYERS)}, "
                f"got {self.dataset_layer!r}"
            )
        if self.split_role == "full_train":
            if self.outer_fold_id != -1:
                raise ValueError(
                    "outer_fold_id must be -1 when split_role='full_train', "
                    f"got {self.outer_fold_id}"
                )
        elif self.outer_fold_id not in range(5):
            raise ValueError(
                "outer_fold_id must be in [0, 4] for CV split roles, "
                f"got {self.outer_fold_id}"
            )
        if not self.file_path.is_file():
            raise FileNotFoundError(f"Aggregated model CSV not found: {self.file_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Frozen CV role manifest not found: {self.manifest_path}")

        recognized_name = self.file_path.name.removesuffix("_model.csv")
        if recognized_name in ALLOWED_DATASET_LAYERS and recognized_name != self.dataset_layer:
            raise ValueError(
                f"dataset_layer={self.dataset_layer!r} does not match {self.file_path.name!r}"
            )

    def _load_frozen_slice(self, *, sep: str) -> pd.DataFrame:
        data = pd.read_csv(self.file_path, sep=sep)
        missing_data_columns = sorted(set(self.REQUIRED_DATA_COLUMNS) - set(data.columns))
        if missing_data_columns:
            raise ValueError(
                f"Aggregated model CSV is missing columns: {missing_data_columns}"
            )
        if data["sample_id"].duplicated().any():
            duplicates = data.loc[data["sample_id"].duplicated(), "sample_id"].head().tolist()
            raise ValueError(f"Aggregated model CSV has duplicate sample_id values: {duplicates}")
        self.dataset_unique_structure_count = int(
            data["Standardized_SMILES"].astype(str).nunique()
        )

        task_data = data.loc[data["task"].eq(self.task), list(self.REQUIRED_DATA_COLUMNS)].copy()
        if task_data.empty:
            raise ValueError(f"No aggregated rows found for task {self.task!r}")
        if task_data[list(self.REQUIRED_DATA_COLUMNS)].isna().any().any():
            bad_columns = task_data.columns[task_data.isna().any()].tolist()
            raise ValueError(
                f"Task {self.task!r} contains missing required values in {bad_columns}"
            )
        if task_data["sample_id"].duplicated().any():
            raise ValueError(f"Task {self.task!r} contains duplicate sample_id values")

        roles = pd.read_csv(self.manifest_path)
        missing_role_columns = sorted(set(self.REQUIRED_ROLE_COLUMNS) - set(roles.columns))
        if missing_role_columns:
            raise ValueError(f"CV role manifest is missing columns: {missing_role_columns}")
        protocol_roles = roles.loc[
            roles["protocol"].eq(self.cv_protocol),
            ["outer_fold_id", "sample_id", "role", "split_group_id"],
        ].copy()
        unexpected_roles = sorted(set(protocol_roles["role"]) - CV_SPLIT_ROLES)
        if unexpected_roles:
            raise ValueError(f"CV role manifest contains unexpected roles: {unexpected_roles}")

        if self.split_role == "full_train":
            scoped_roles = self._collapse_full_train_roles(protocol_roles)
        else:
            scoped_roles = protocol_roles.loc[
                protocol_roles["outer_fold_id"].eq(self.outer_fold_id),
                ["sample_id", "role", "split_group_id"],
            ].copy()
        if scoped_roles.empty:
            scope = (
                "all five folds"
                if self.split_role == "full_train"
                else f"outer_fold_id={self.outer_fold_id}"
            )
            raise ValueError(f"No CV roles for protocol={self.cv_protocol!r}, {scope}")
        if scoped_roles["sample_id"].duplicated().any():
            raise ValueError("CV role manifest has duplicate sample_id values in the requested fold")

        merged = task_data.merge(
            scoped_roles,
            on="sample_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing_ids = merged.loc[merged["_merge"].ne("both"), "sample_id"].tolist()
        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} task rows are absent from the requested CV manifest; "
                f"examples={missing_ids[:5]}"
            )

        scaler_role = "full_train" if self.split_role == "full_train" else "outer_train"
        self._outer_train_scaler_frame = merged.loc[
            merged["role"].eq(scaler_role),
            ["Duration_Value", "log10_mgperL"],
        ].copy()
        if self._outer_train_scaler_frame.empty:
            raise ValueError(
                f"No {scaler_role} rows available to fit scalers for {self.task!r}"
            )

        selected = merged.loc[merged["role"].eq(self.split_role)].copy()
        if selected.empty:
            raise ValueError(
                f"No samples for task={self.task!r}, protocol={self.cv_protocol!r}, "
                f"outer_fold_id={self.outer_fold_id}, role={self.split_role!r}"
            )
        if selected["sample_id"].duplicated().any():
            raise ValueError("Frozen slice contains duplicate sample_id values")

        labels = selected["log10_mgperL"].to_numpy(dtype=float)
        durations = selected["Duration_Value"].to_numpy(dtype=float)
        if not np.isfinite(labels).all():
            raise ValueError("Frozen slice contains non-finite labels")
        if not np.isfinite(durations).all() or (durations < 0).any():
            raise ValueError("Frozen slice contains invalid exposure durations")
        if (selected["aggregation_n"].astype(int) < 1).any():
            raise ValueError("Frozen slice contains aggregation_n < 1")

        unexpected_effects = sorted(set(selected["effect"].astype(str)) - set(self.ALL_EFFECTS))
        if unexpected_effects:
            raise ValueError(f"Frozen slice contains unknown effects: {unexpected_effects}")
        unexpected_media = sorted(
            set(selected["media_type"].astype(str)) - {"FW", "SW", "OTHER"}
        )
        if unexpected_media:
            raise ValueError(f"Frozen slice contains unknown media classes: {unexpected_media}")

        return selected.sort_values("sample_id").reset_index(drop=True)

    @staticmethod
    def _collapse_full_train_roles(protocol_roles: pd.DataFrame) -> pd.DataFrame:
        if protocol_roles.empty:
            return pd.DataFrame(columns=["sample_id", "role", "split_group_id"])

        invalid_folds = sorted(
            set(protocol_roles["outer_fold_id"].astype(int)) - set(range(5))
        )
        if invalid_folds:
            raise ValueError(
                f"CV role manifest contains unexpected outer_fold_id values: {invalid_folds}"
            )
        if protocol_roles.duplicated(["sample_id", "outer_fold_id"]).any():
            raise ValueError(
                "CV role manifest has duplicate sample_id values within an outer fold"
            )

        grouped = protocol_roles.groupby("sample_id", sort=False, observed=True)
        fold_counts = grouped["outer_fold_id"].nunique()
        incomplete = fold_counts.loc[fold_counts.ne(5)]
        if not incomplete.empty:
            examples = incomplete.index.astype(str).tolist()[:5]
            raise ValueError(
                f"CV role manifest does not cover all five folds for "
                f"{len(incomplete)} samples; examples={examples}"
            )

        split_group_counts = grouped["split_group_id"].nunique(dropna=False)
        inconsistent = split_group_counts.loc[split_group_counts.ne(1)]
        if not inconsistent.empty:
            examples = inconsistent.index.astype(str).tolist()[:5]
            raise ValueError(
                f"CV role manifest assigns inconsistent split_group_id values to "
                f"{len(inconsistent)} samples across folds; examples={examples}"
            )

        collapsed = grouped["split_group_id"].first().reset_index()
        collapsed["role"] = "full_train"
        return collapsed[["sample_id", "role", "split_group_id"]]

    def _validate_structures(self) -> None:
        invalid = []
        for smiles in dict.fromkeys(self.smiles_list):
            if Chem.MolFromSmiles(smiles) is None:
                invalid.append(smiles)
        if invalid:
            raise ValueError(
                f"Frozen slice contains {len(invalid)} invalid structures; examples={invalid[:5]}"
            )

    def _build_condition_features(self) -> None:
        effect_index = {effect: idx for idx, effect in enumerate(self.ALL_EFFECTS)}
        indices = np.asarray([effect_index[effect] for effect in self.effect_list], dtype=np.int64)
        self.encoded_effects = np.eye(len(self.ALL_EFFECTS), dtype=np.float32)[indices]
        self.num_effects = len(self.ALL_EFFECTS)

        media_index = {"FW": 0, "SW": 1, "OTHER": 2}
        media_indices = np.asarray(
            [media_index[value] for value in self.media_list], dtype=np.int64
        )
        self.media_onehots = np.eye(3, dtype=np.float32)[media_indices]

    @property
    def artifact_stem(self) -> str:
        fold_token = (
            "fold-1_full_train"
            if self.split_role == "full_train"
            else f"fold{self.outer_fold_id}"
        )
        return (
            f"{self.dataset_layer}_{self.dataset_sha256[:12]}_"
            f"{self.manifest_sha256[:12]}_{self.cv_protocol}_"
            f"{fold_token}_{self.task}"
        )

    @property
    def label_scaler_path(self) -> Path:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required")
        return self.preprocessed_dir / f"label_scaler_{self.artifact_stem}.pkl"

    def _fit_or_load_scalers(self) -> None:
        if self.preprocessed_dir is None:
            raise ValueError("preprocessed_dir is required for fold-specific preprocessing artifacts")
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        duration_path = self.preprocessed_dir / f"duration_scaler_{self.artifact_stem}.pkl"
        label_path = self.label_scaler_path

        self.duration_values = np.asarray(self.duration_list, dtype=float).reshape(-1, 1)
        log_duration = np.log1p(self.duration_values)
        self.labels = np.asarray(self.labels_list, dtype=float).reshape(-1, 1)

        if self.split_role in {"outer_train", "full_train"}:
            self.duration_scaler = StandardScaler().fit(log_duration)
            self.label_scaler = StandardScaler().fit(self.labels)
            atomic_joblib_dump(self.duration_scaler, duration_path)
            atomic_joblib_dump(self.label_scaler, label_path)
        else:
            if duration_path.is_file() and label_path.is_file():
                self.duration_scaler = joblib.load(duration_path)
                self.label_scaler = joblib.load(label_path)
            else:
                train_duration = np.log1p(
                    self._outer_train_scaler_frame["Duration_Value"]
                    .to_numpy(dtype=float)
                    .reshape(-1, 1)
                )
                train_labels = (
                    self._outer_train_scaler_frame["log10_mgperL"]
                    .to_numpy(dtype=float)
                    .reshape(-1, 1)
                )
                self.duration_scaler = StandardScaler().fit(train_duration)
                self.label_scaler = StandardScaler().fit(train_labels)
                atomic_joblib_dump(self.duration_scaler, duration_path)
                atomic_joblib_dump(self.label_scaler, label_path)

        self.scaled_duration_values = self.duration_scaler.transform(log_duration)
        self.scaled_labels = self.label_scaler.transform(self.labels).reshape(-1).tolist()

    @staticmethod
    def _ghs_class(log10_val: float, task: str) -> int:
        thresholds = (-1.0, 0.0, 1.0) if "EC10" in task else (0.0, 1.0, 2.0)
        if log10_val <= thresholds[0]:
            return 0
        if log10_val <= thresholds[1]:
            return 1
        if log10_val <= thresholds[2]:
            return 2
        return 3

    def __len__(self) -> int:
        return len(self.smiles_list) // self.group_k_consecutive

    def _initialize_preprocessed_mapping(self) -> None:
        if self.smiles_to_path is not None or self.preprocessed_dir is None:
            return
        if self._graph_dir is not None:
            marker_path = self._graph_dir / "_GRAPH_STORE_SUCCESS.json"
            if marker_path.is_file():
                with marker_path.open("r", encoding="utf-8") as handle:
                    marker = json.load(handle)
                expected_marker = {
                    "status": "completed",
                    "dataset_layer": self.dataset_layer,
                    "model_csv_sha256": self.dataset_sha256,
                    "feature_cache_version": FEATURE_CACHE_VERSION,
                    "cache_namespace": self.cache_namespace,
                    "unique_structures": self.dataset_unique_structure_count,
                }
                mismatches = {
                    key: {"actual": marker.get(key), "expected": expected}
                    for key, expected in expected_marker.items()
                    if marker.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"Versioned graph-store marker does not match the dataset: {mismatches}"
                    )

                mapping: Dict[str, str] = {}
                missing_paths = []
                for smiles in dict.fromkeys(self.smiles_list):
                    relative_path = (
                        Path("graphs")
                        / self.cache_namespace
                        / f"{_stable_token(smiles)}.bin"
                    )
                    if not (self.preprocessed_dir / relative_path).is_file():
                        missing_paths.append(str(relative_path))
                    mapping[smiles] = relative_path.as_posix()
                if missing_paths:
                    raise FileNotFoundError(
                        f"Versioned graph store is incomplete ({len(missing_paths)} missing); "
                        f"examples={missing_paths[:5]}"
                    )
                self.smiles_to_path = mapping
                self.graph_mapping_source = "versioned_global_store"
                return

        if self._mapping_path and self._mapping_path.is_file():
            with self._mapping_path.open("r", encoding="utf-8") as handle:
                self.smiles_to_path = json.load(handle)
            self.graph_mapping_source = "slice_mapping"

    def is_preprocessed(self) -> bool:
        self._initialize_preprocessed_mapping()
        return self.smiles_to_path is not None

    def _preload_graphs(self) -> None:
        if self.preprocessed_dir is None or self.smiles_to_path is None:
            raise RuntimeError("Preprocessed graph mapping is not initialized")
        unique_smiles = list(dict.fromkeys(self.smiles_list))
        missing = [smiles for smiles in unique_smiles if smiles not in self.smiles_to_path]
        if missing:
            raise KeyError(
                f"Preprocessed mapping is incomplete ({len(missing)} missing); examples={missing[:5]}"
            )
        print(f"[{self.task}/{self.split_role}] preloading {len(unique_smiles)} molecular graphs")
        for smiles in tqdm(unique_smiles, desc=f"Preloading {self.split_role} graphs", leave=False):
            if smiles in self._graph_cache:
                continue
            graph_path = self.preprocessed_dir / self.smiles_to_path[smiles]
            if not graph_path.is_file():
                raise FileNotFoundError(f"Preprocessed graph is missing: {graph_path}")
            graphs, _ = dgl.load_graphs(str(graph_path))
            if len(graphs) != 1:
                raise ValueError(f"Expected one graph in {graph_path}, found {len(graphs)}")
            self._graph_cache[smiles] = graphs[0]

    def load_graph(self, smiles: str) -> dgl.DGLGraph:
        if smiles in self._graph_cache:
            return self._graph_cache[smiles]
        if self.preprocessed_dir is None or self.smiles_to_path is None:
            raise RuntimeError("Preprocessed graph cache is not available")
        if smiles not in self.smiles_to_path:
            raise KeyError(f"SMILES {smiles!r} is absent from the preprocessed graph mapping")
        graph_path = self.preprocessed_dir / self.smiles_to_path[smiles]
        if not graph_path.is_file():
            raise FileNotFoundError(f"Preprocessed graph is missing: {graph_path}")
        graphs, _ = dgl.load_graphs(str(graph_path))
        if len(graphs) != 1:
            raise ValueError(f"Expected one graph in {graph_path}, found {len(graphs)}")
        graph = graphs[0]
        self._graph_cache[smiles] = graph
        return graph

    def featurize_graph(self, smiles: str) -> dgl.DGLGraph:
        if self.featurizer is None:
            raise RuntimeError("Runtime featurizer is not initialized")
        graph = self.featurizer.featurize_smiles_single(smiles)
        if graph.idtype != torch.int32:
            graph = graph.int()
        return graph

    def _get_underlying_item(
        self, index: int
    ) -> tuple[dgl.DGLGraph, float, np.ndarray, np.ndarray, float, int, str, str]:
        if index >= len(self.smiles_list):
            raise IndexError(
                f"Index {index} out of range for dataset with {len(self.smiles_list)} items"
            )
        smiles = self.smiles_list[index]
        graph = self.load_graph(smiles) if self.is_preprocessed() else self.featurize_graph(smiles)
        return (
            graph,
            float(self.scaled_duration_values[index].item()),
            self.encoded_effects[index],
            self.media_onehots[index],
            float(self.scaled_labels[index]),
            int(self.ghs_classes[index]),
            smiles,
            self.sample_ids[index],
        )

    def __getitem__(
        self, index: int
    ) -> list[tuple[dgl.DGLGraph, float, np.ndarray, np.ndarray, float, int, str, str]]:
        items = []
        start = index * self.group_k_consecutive
        end = min((index + 1) * self.group_k_consecutive, len(self.smiles_list))
        for item_index in range(start, end):
            items.append(self._get_underlying_item(item_index))
        return items

    def collate(
        self,
        items: List[
            List[Tuple[dgl.DGLGraph, float, np.ndarray, np.ndarray, float, int, str, str]]
        ],
    ) -> Tuple[
        dgl.DGLGraph,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[str],
        List[str],
    ]:
        flattened = [item for chunk in items for item in chunk]
        if not flattened:
            raise ValueError("Cannot collate an empty batch")
        graphs, durations, effects, media, labels, classes, smiles, sample_ids = zip(*flattened)
        return (
            dgl.batch(list(graphs)),
            torch.tensor(durations, dtype=torch.float32),
            torch.tensor(np.asarray(effects), dtype=torch.float32),
            torch.tensor(np.asarray(media), dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.long),
            list(smiles),
            list(sample_ids),
        )

    def get_sample_weights(self, temperature: float = 2.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        counts = Counter(self.molecule_ids)
        return torch.tensor(
            [
                1.0 / (counts[molecule_id] ** (1.0 / temperature))
                for molecule_id in self.molecule_ids
            ],
            dtype=torch.float32,
        )

    def preprocess(self) -> None:
        if self.is_preprocessed():
            return
        if self.preprocessed_dir is None or self._mapping_path is None:
            raise ValueError("preprocessed_dir is required to persist graph features")

        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        graph_dir = self.preprocessed_dir / "graphs" / self.cache_namespace
        graph_dir.mkdir(parents=True, exist_ok=True)
        smiles_to_path: Dict[str, str] = {}

        for smiles in tqdm(dict.fromkeys(self.smiles_list), desc=f"Preprocessing {self.split_role}"):
            graph_name = f"{_stable_token(smiles)}.bin"
            graph_path = graph_dir / graph_name
            if graph_path.is_file():
                graphs, _ = dgl.load_graphs(str(graph_path))
                if len(graphs) != 1:
                    raise ValueError(f"Invalid graph cache entry: {graph_path}")
            else:
                graph = self.featurize_graph(smiles)
                temporary_path = graph_path.with_name(
                    f".{graph_path.name}.{os.getpid()}.tmp"
                )
                dgl.save_graphs(str(temporary_path), [graph])
                os.replace(temporary_path, graph_path)
            smiles_to_path[smiles] = str(graph_path.relative_to(self.preprocessed_dir))

        temporary_mapping = self._mapping_path.with_name(
            f".{self._mapping_path.name}.{os.getpid()}.tmp"
        )
        with temporary_mapping.open("w", encoding="utf-8") as handle:
            json.dump(smiles_to_path, handle, sort_keys=True)
        os.replace(temporary_mapping, self._mapping_path)
        self.smiles_to_path = smiles_to_path
        self.graph_mapping_source = "slice_mapping"
        self.featurizer = None
        self._preload_graphs()

    def get_feature_dimensions(self) -> Dict[str, int]:
        return {"num_effects": self.num_effects}
