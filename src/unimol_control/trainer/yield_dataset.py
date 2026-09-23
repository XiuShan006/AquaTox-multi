from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gin
import json
import hashlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from tqdm import tqdm

from unicore.data import Dictionary as UnicoreDictionary

from data_provider.conformer import (
    ConformerSettings,
    DEFAULT_EXCLUDED_SMILES,
    build_unimol_input_isolated,
    summarize_conformer_metadata,
)
from data_provider.aquatox_dataset import (
    conformer_path,
    load_aquatox_unimol_input,
    select_aquatox_split,
    validate_aquatox_mmff94_audit,
)

try:
    import dgl
except Exception:
    dgl = None


@gin.configurable()
class YieldDataset(Dataset):
    def __init__(
        self,
        task: str,
        split: str,
        preprocessed_dir: str,
        split_ratio: float = 0.8,
        sep: str = ',',
        contrastive: bool = False,
        unknown_token: str = "<UNK>",
        file_path: str = 'data/aquatox/data/common_intersection_model.csv',
        unimol_precompute: bool = False,
        split_type: str = 'random',
        roles_path: Optional[str] = None,
        split_protocol: str = 'scaffold',
        outer_fold_id: int = 0,
        conformer_root: Optional[str] = None,
        conformer_audit_path: Optional[str] = None,
        unimol_random_seed: int = 42,
        unimol_mmff_variant: str = 'MMFF94',
        unimol_mmff_max_iters: int = 1000,
        unimol_use_uff_fallback: bool = True,
        unimol_uff_max_iters: int = 1000,
        unimol_molecule_timeout_seconds: float = 60.0,
    ):
        self.task = task
        self.split_name = split
        self.file_path = str(Path(file_path).expanduser().resolve())
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.contrastive = contrastive
        self.group_k_consecutive = 2 if contrastive else 1
        self.unknown_token = unknown_token
        self.unimol_precompute = unimol_precompute
        if split_type not in {'random', 'unified', 'manifest'}:
            raise ValueError("split_type must be 'random', 'unified', or 'manifest'")
        self.split_type = split_type
        self.manifest_split = split_type == 'manifest'
        self.roles_path = roles_path
        self.split_protocol = split_protocol
        self.outer_fold_id = int(outer_fold_id)
        self.source_conformer_root = (
            Path(conformer_root).expanduser().resolve() if conformer_root else None
        )
        self.source_conformer_audit_path = (
            Path(conformer_audit_path).expanduser().resolve()
            if conformer_audit_path
            else None
        )
        if self.manifest_split and not self.roles_path:
            raise ValueError("roles_path is required when split_type='manifest'")
        if self.manifest_split and self.source_conformer_root is None:
            raise ValueError("conformer_root is required when split_type='manifest'")
        if self.manifest_split and self.source_conformer_audit_path is None:
            raise ValueError(
                "conformer_audit_path is required when split_type='manifest'"
            )
        if self.source_conformer_root is not None and not self.source_conformer_root.is_dir():
            raise FileNotFoundError(
                f"release conformer directory not found: {self.source_conformer_root}"
            )
        self.unimol_conformer_settings = ConformerSettings(
            random_seed=unimol_random_seed,
            mmff_variant=unimol_mmff_variant,
            mmff_max_iters=unimol_mmff_max_iters,
            use_uff_fallback=unimol_use_uff_fallback,
            uff_max_iters=unimol_uff_max_iters,
            molecule_timeout_seconds=unimol_molecule_timeout_seconds,
        )
        self.unimol_cache_tag = self.unimol_conformer_settings.cache_tag

        self.ALL_EFFECTS = ['DVP', 'GRO', 'ITX', 'MOR', 'MPH', 'POP', 'REP']

        self.SMILES_TO_FILTER = (
            [] if self.source_conformer_root is not None else list(DEFAULT_EXCLUDED_SMILES)
        )
        self.failed_smiles_file = (
            self.preprocessed_dir / f"unimol_failed_smiles_{self.unimol_cache_tag}.json"
            if self.preprocessed_dir and self.source_conformer_root is None
            else None
        )
        self.dynamic_failed_smiles = set()
        if self.failed_smiles_file is not None and self.failed_smiles_file.exists():
            try:
                with open(self.failed_smiles_file, 'r') as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    self.dynamic_failed_smiles = set(data)
                elif isinstance(data, dict) and 'failed' in data:
                    self.dynamic_failed_smiles = set(data['failed'])
                print(f"加载到 {len(self.dynamic_failed_smiles)} 个历史失败的SMILES用于过滤")
            except Exception as e:
                print(f"[WARN] 读取unimol失败过滤表出错: {e}")

        if self.manifest_split:
            data_df, preprocessing_fit_df = select_aquatox_split(
                file_path,
                self.roles_path,
                task=self.task,
                split=split,
                protocol=self.split_protocol,
                outer_fold_id=self.outer_fold_id,
                sep=sep,
            )
            print(
                f"[Split] Manifest: protocol={self.split_protocol}, "
                f"outer_fold={self.outer_fold_id}, role={data_df['role'].iloc[0]}, "
                f"task={self.task}, samples={len(data_df)}"
            )
        else:
            df = pd.read_csv(file_path, sep=sep)
            df = df.dropna(
                subset=[
                    'Standardized_SMILES',
                    'log10_mgperL',
                    'Duration_Value',
                    'effect',
                    'endpoint',
                    'task',
                ]
            )
            df = df[df['task'] == self.task]
            if len(df) == 0:
                raise ValueError(f"No data found for task '{self.task}'")

            unique_smiles = df['Standardized_SMILES'].unique()
            if split_type == 'unified':
                all_tasks_df = pd.read_csv(file_path, sep=sep).dropna(
                    subset=['Standardized_SMILES']
                )
                union_smiles = sorted(all_tasks_df['Standardized_SMILES'].unique())
                rng = np.random.RandomState(42)
                shuffled = rng.permutation(len(union_smiles))
                n_valid_union = int(len(union_smiles) * (1.0 - split_ratio))
                valid_smiles_union = set(
                    np.array(union_smiles)[shuffled[:n_valid_union]]
                )
                train_smiles_set = set(unique_smiles) - valid_smiles_union
                print(
                    f"[Split] Unified: train_mol={len(train_smiles_set)}, "
                    f"valid_mol={len(set(unique_smiles) & valid_smiles_union)} "
                    f"/ {len(unique_smiles)} total"
                )
            else:
                rng = np.random.RandomState(42)
                shuffled = rng.permutation(len(unique_smiles))
                n_train = int(len(unique_smiles) * split_ratio)
                train_smiles_set = set(unique_smiles[shuffled[:n_train]])
                print(
                    f"[Split] Random: train_mol={len(train_smiles_set)}/"
                    f"{len(unique_smiles)}"
                )

            train_df = df[
                df['Standardized_SMILES'].isin(train_smiles_set)
            ].reset_index(drop=True)
            valid_df = df[
                ~df['Standardized_SMILES'].isin(train_smiles_set)
            ].reset_index(drop=True)
            data_df = train_df if split == "train" else valid_df
            preprocessing_fit_df = train_df

        excluded_smiles = set(self.SMILES_TO_FILTER) | self.dynamic_failed_smiles
        keep_mask = ~data_df['Standardized_SMILES'].isin(excluded_smiles)
        filtered_df = data_df.loc[keep_mask].reset_index(drop=True)
        for smiles in data_df.loc[~keep_mask, 'Standardized_SMILES']:
            print(f"过滤掉SMILES: {smiles}")

        self.smiles_list = filtered_df['Standardized_SMILES'].tolist()
        self.labels_list = filtered_df['log10_mgperL'].tolist()
        self.duration_list = filtered_df['Duration_Value'].tolist()
        self.effect_list = filtered_df['effect'].tolist()
        self.sample_ids = (
            filtered_df['sample_id'].tolist() if 'sample_id' in filtered_df else []
        )
        self.molecule_ids = (
            filtered_df['molecule_id'].tolist() if 'molecule_id' in filtered_df else []
        )
        self.split_parent_ids = (
            filtered_df['split_parent_id'].tolist()
            if 'split_parent_id' in filtered_df
            else []
        )
        self.split_group_ids = (
            filtered_df['split_group_id'].tolist()
            if 'split_group_id' in filtered_df
            else []
        )
        self.smiles_to_molecule_id = dict(zip(self.smiles_list, self.molecule_ids))
        self.aquatox_mmff_energy_by_molecule: Dict[str, float] = {}
        if self.source_conformer_root is not None:
            molecule_ids_per_smiles = filtered_df.groupby('Standardized_SMILES')[
                'molecule_id'
            ].nunique()
            if (molecule_ids_per_smiles != 1).any():
                raise ValueError("Each release SMILES must map to exactly one molecule_id")
            if self.source_conformer_audit_path is None:
                raise ValueError(
                    "conformer_audit_path is required with release conformers"
                )
            self.aquatox_mmff_energy_by_molecule = validate_aquatox_mmff94_audit(
                self.source_conformer_audit_path,
                self.molecule_ids,
                conformer_root=self.source_conformer_root,
            )

        fit_keep = ~preprocessing_fit_df['Standardized_SMILES'].isin(excluded_smiles)
        preprocessing_fit_df = preprocessing_fit_df.loc[fit_keep].reset_index(drop=True)
        if preprocessing_fit_df.empty:
            raise ValueError(f"No preprocessing fit rows remain for task {self.task!r}")
        self.preprocessing_fit_sample_ids = (
            preprocessing_fit_df['sample_id'].tolist()
            if 'sample_id' in preprocessing_fit_df
            else []
        )
        self._unimol_cache: Dict[str, tuple] = {}
        self._source_conformers_ready: Optional[bool] = None

        self.ghs_classes = np.array(
            [self._ghs_class(v, self.task) for v in self.labels_list],
            dtype=np.int64,
        )

        self.unimol_smiles_to_path: Optional[Dict[str, str]] = None
        self.unimol_conformer_metadata: Optional[Dict[str, Dict[str, Any]]] = None
        if self.preprocessed_dir and self.source_conformer_root is None:
            unimol_json = self._unimol_mapping_path()
            if unimol_json.exists():
                with open(unimol_json, "r") as fp:
                    self.unimol_smiles_to_path = json.load(fp)
            metadata_json = self._unimol_metadata_path()
            if metadata_json.exists():
                with open(metadata_json, "r") as fp:
                    self.unimol_conformer_metadata = json.load(fp)
                self.dynamic_failed_smiles.update(
                    smi
                    for smi, record in self.unimol_conformer_metadata.items()
                    if not record.get('success')
                )
        if (
            self.preprocessed_dir
            and self.source_conformer_root is None
            and self.unimol_precompute
            and not self.is_unimol_preprocessed()
        ):
            try:
                self.preprocess_unimol()
            except Exception as e:
                raise RuntimeError(
                    f"Uni-Mol preprocessing failed for {self.task}/{self.split_name}: {e}"
                ) from e
        self._drop_new_unimol_failures()

        if self.manifest_split:
            fit_effects = preprocessing_fit_df['effect'].to_numpy().reshape(-1, 1)
            unexpected_effects = sorted(
                set(preprocessing_fit_df['effect']) - set(self.ALL_EFFECTS)
            )
            if unexpected_effects:
                raise ValueError(f"Unexpected effect values: {unexpected_effects}")
            self.effect_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                categories=[self.ALL_EFFECTS],
            )
            self.effect_encoder.fit(fit_effects)
            self.encoded_effects = self.effect_encoder.transform(
                np.array(self.effect_list).reshape(-1, 1)
            )
        elif split == "train":
            all_effects = self.ALL_EFFECTS

            self.effect_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                categories=[all_effects]
            )

            self.encoded_effects = self.effect_encoder.fit_transform(
                np.array(self.effect_list).reshape(-1, 1)
            )

            if self.preprocessed_dir:
                self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.effect_encoder, self.preprocessed_dir / f"effect_encoder_{self.task}.pkl")
                with open(self.preprocessed_dir / f"effect_categories_{self.task}.json", "w") as fp:
                    json.dump(self.ALL_EFFECTS, fp)
        else:
            if self.preprocessed_dir:
                effect_encoder_path = self.preprocessed_dir / f"effect_encoder_{self.task}.pkl"
                effect_categories_path = self.preprocessed_dir / f"effect_categories_{self.task}.json"
            else:
                effect_encoder_path = Path(f"effect_encoder_{self.task}.pkl")
                effect_categories_path = Path(f"effect_categories_{self.task}.json")

            if not effect_encoder_path.exists() or not effect_categories_path.exists():
                raise FileNotFoundError(f"Missing effect encoder or categories for task '{self.task}'")

            self.effect_encoder = joblib.load(effect_encoder_path)

            with open(effect_categories_path, "r") as fp:
                all_effects = json.load(fp)

            self.encoded_effects = self.effect_encoder.transform(
                np.array(self.effect_list).reshape(-1, 1)
            )

        self.num_effects = len(self.ALL_EFFECTS)

        if self.encoded_effects.shape[1] < self.num_effects:
            padding = np.zeros((len(self.encoded_effects), self.num_effects - self.encoded_effects.shape[1]))
            self.encoded_effects = np.hstack([self.encoded_effects, padding])

        print(f"[DEBUG] Effect onehots 维度: {self.encoded_effects.shape}")
        print(f"[DEBUG] 预期effect类别数: {self.num_effects}")

        self.duration_values = np.array(self.duration_list).reshape(-1, 1)
        if self.manifest_split:
            fit_duration = preprocessing_fit_df['Duration_Value'].to_numpy().reshape(-1, 1)
            self.duration_scaler = StandardScaler().fit(fit_duration)
            self.scaled_duration_values = self.duration_scaler.transform(
                self.duration_values
            )
        elif split == "train":
            self.duration_scaler = StandardScaler()
            self.scaled_duration_values = self.duration_scaler.fit_transform(self.duration_values)

            if self.preprocessed_dir:
                self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.duration_scaler, self.preprocessed_dir / f"duration_scaler_{self.task}_train.pkl")
        else:
            if self.preprocessed_dir:
                duration_scaler_path = self.preprocessed_dir / f"duration_scaler_{self.task}_train.pkl"
            else:
                duration_scaler_path = Path(f"duration_scaler_{self.task}_train.pkl")

            if not duration_scaler_path.exists():
                raise FileNotFoundError(f"Missing duration scaler for task '{self.task}'")

            self.duration_scaler = joblib.load(duration_scaler_path)
            self.scaled_duration_values = self.duration_scaler.transform(self.duration_values)

        self.labels = np.array(self.labels_list).reshape(-1, 1)
        if self.manifest_split:
            fit_labels = preprocessing_fit_df['log10_mgperL'].to_numpy().reshape(-1, 1)
            self.label_scaler = StandardScaler().fit(fit_labels)
            scaled_labels = self.label_scaler.transform(self.labels).reshape(-1)
            self.scaled_labels = scaled_labels.tolist()
        elif split == "train":
            self.label_scaler = StandardScaler()
            self.scaled_labels = self.label_scaler.fit_transform(self.labels).squeeze().tolist()

            if self.preprocessed_dir:
                self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(self.label_scaler, self.preprocessed_dir / f"label_scaler_{self.task}_train.pkl")
        else:
            if self.preprocessed_dir:
                label_scaler_path = self.preprocessed_dir / f"label_scaler_{self.task}_train.pkl"
            else:
                label_scaler_path = Path(f"label_scaler_{self.task}_train.pkl")

            if not label_scaler_path.exists():
                raise FileNotFoundError(f"Missing label scaler for task '{self.task}'")

            self.label_scaler = joblib.load(label_scaler_path)
            self.scaled_labels = self.label_scaler.transform(self.labels).squeeze().tolist()

        self.smiles_to_path: Optional[Dict[str, str]] = None
        if self.preprocessed_dir:
            path_json = self.preprocessed_dir / f"smiles_to_path_{self.split_name}.json"
            if path_json.exists():
                with open(path_json, "r") as fp:
                    self.smiles_to_path = json.load(fp)

        if not self.is_preprocessed() and not self.is_unimol_preprocessed():
            from featurizers import ReactionFeaturizer
            self.featurizer = ReactionFeaturizer()
        else:
            self.featurizer = None

        if (
            self.is_unimol_preprocessed()
            and self.preprocessed_dir
            and self.source_conformer_root is None
        ):
            unique_smiles = list(dict.fromkeys(self.smiles_list))
            print(f"[{task}/{split}] 预加载 {len(unique_smiles)} 个唯一分子 Uni-Mol 张量到内存...")
            for smi in tqdm(unique_smiles, desc=f"Preloading Uni-Mol {split}", leave=False):
                if smi in self.unimol_smiles_to_path:
                    try:
                        data = torch.load(
                            self.preprocessed_dir / self.unimol_smiles_to_path[smi],
                            map_location='cpu',
                        )
                        self._unimol_cache[smi] = (data['atom_vec'], data['dist'], data['edge_type'])
                    except Exception:
                        pass
            print(f"[{task}/{split}] 已缓存 {len(self._unimol_cache)} 个 Uni-Mol 张量（RAM）")

    @staticmethod
    def _ghs_class(log10_val: float, task: str) -> int:
        if 'EC10' in task:
            thresholds = [-1.0, 0.0, 1.0]
        else:
            thresholds = [0.0, 1.0, 2.0]
        for cls_idx, thr in enumerate(thresholds):
            if log10_val <= thr:
                return cls_idx
        return len(thresholds)

    def __len__(self):
        return len(self.smiles_list) // self.group_k_consecutive

    def _drop_new_unimol_failures(self) -> None:
        if not self.dynamic_failed_smiles:
            return
        keep = [
            idx
            for idx, smi in enumerate(self.smiles_list)
            if smi not in self.dynamic_failed_smiles
        ]
        if len(keep) == len(self.smiles_list):
            return
        dropped = len(self.smiles_list) - len(keep)
        if not keep:
            raise ValueError(
                f"All samples in {self.task}/{self.split_name} failed Uni-Mol preprocessing"
            )

        self.smiles_list = [self.smiles_list[idx] for idx in keep]
        self.labels_list = [self.labels_list[idx] for idx in keep]
        self.duration_list = [self.duration_list[idx] for idx in keep]
        self.effect_list = [self.effect_list[idx] for idx in keep]
        for attribute in (
            "sample_ids",
            "molecule_ids",
            "split_parent_ids",
            "split_group_ids",
        ):
            values = getattr(self, attribute)
            if values:
                setattr(self, attribute, [values[idx] for idx in keep])
        self.ghs_classes = self.ghs_classes[keep]
        print(
            f"[Uni-Mol preprocess] removed {dropped} failed records from "
            f"{self.task}/{self.split_name}"
        )

    def is_preprocessed(self) -> bool:
        if self.smiles_to_path is None and self.preprocessed_dir:
            path_json = self.preprocessed_dir / f"smiles_to_path_{self.split_name}.json"
            if path_json.exists():
                with open(path_json, "r") as fp:
                    self.smiles_to_path = json.load(fp)
        return (self.smiles_to_path is not None)

    def _unimol_mapping_path(self) -> Path:
        assert self.preprocessed_dir is not None
        return self.preprocessed_dir / f"unimol_smiles_to_path_{self.unimol_cache_tag}.json"

    def _unimol_metadata_path(self) -> Path:
        assert self.preprocessed_dir is not None
        return self.preprocessed_dir / f"unimol_conformer_metadata_{self.unimol_cache_tag}.json"

    def _unimol_report_path(self) -> Path:
        assert self.preprocessed_dir is not None
        return self.preprocessed_dir / f"unimol_conformer_report_{self.unimol_cache_tag}.json"

    def is_unimol_preprocessed(self) -> bool:
        if self.source_conformer_root is not None:
            if self._source_conformers_ready is None:
                self._source_conformers_ready = all(
                    conformer_path(self.source_conformer_root, molecule_id).is_file()
                    for molecule_id in set(self.molecule_ids)
                )
            return self._source_conformers_ready
        if self.unimol_smiles_to_path is None and self.preprocessed_dir:
            unimol_json = self._unimol_mapping_path()
            if unimol_json.exists():
                with open(unimol_json, "r") as fp:
                    self.unimol_smiles_to_path = json.load(fp)
        if self.unimol_conformer_metadata is None and self.preprocessed_dir:
            metadata_json = self._unimol_metadata_path()
            if metadata_json.exists():
                with open(metadata_json, "r") as fp:
                    self.unimol_conformer_metadata = json.load(fp)
        if self.unimol_smiles_to_path is None or self.unimol_conformer_metadata is None:
            return False
        missing = [
            smi
            for smi in self.smiles_list
            if smi not in self.unimol_smiles_to_path
            or smi not in self.unimol_conformer_metadata
            or not (self.preprocessed_dir / self.unimol_smiles_to_path[smi]).is_file()
        ]
        return len(missing) == 0

    def load_graph(self, smiles: str):
        if dgl is None:
            raise ImportError("DGL is required for graph fallback loading. Use Uni-Mol precomputed inputs instead.")
        assert self.preprocessed_dir is not None and self.smiles_to_path is not None
        if smiles not in self.smiles_to_path:
            raise KeyError(f"SMILES '{smiles}' not found in preprocessed graphs")

        path_pkl = self.preprocessed_dir / self.smiles_to_path[smiles]
        graphs, _ = dgl.load_graphs(str(path_pkl))
        graph = graphs[0]

        return graph

    def featurize_graph(self, smiles: str):
        if dgl is None:
            raise ImportError("DGL is required for graph fallback featurization. Use Uni-Mol precomputed inputs instead.")
        try:
            graph = self.featurizer.featurize_smiles_single(smiles)
            if graph.idtype != torch.int32:
                graph = graph.int()
            return graph
        except Exception as e:
            print(f"Error featurizing SMILES '{smiles}': {e}")
            g = dgl.graph(([0], [0]), idtype=torch.int32)
            g.ndata['h'] = torch.zeros(1, 1271)
            g.edata['e'] = torch.zeros(1, 13)
            return g

    def _get_underling_item(self, index: int) -> tuple[Any, float, np.ndarray, float, int, str]:
        if index >= len(self.smiles_list):
            raise IndexError(f"Index {index} out of range for dataset with {len(self.smiles_list)} items")

        smiles = self.smiles_list[index]

        duration_value = self.scaled_duration_values[index].item()
        effect_onehot = self.encoded_effects[index]
        label = self.scaled_labels[index]
        ghs_class = int(self.ghs_classes[index])

        if self.is_unimol_preprocessed():
            g = None
        else:
            try:
                if self.is_preprocessed():
                    g = self.load_graph(smiles)
                else:
                    g = self.featurize_graph(smiles)
            except Exception as e:
                print(f"Error loading/featurizing graph for SMILES {smiles} at index {index}: {e}")
                g = dgl.graph(([0], [0]))
                g.ndata['h'] = torch.zeros(1, 1271)
                g.edata['e'] = torch.zeros(1, 13)

        return g, duration_value, effect_onehot, label, ghs_class, smiles

    def __getitem__(self, index: int) -> list[tuple[Any, float, np.ndarray, float, int, str]]:
        items = []
        start = index * self.group_k_consecutive
        end = min((index + 1) * self.group_k_consecutive, len(self.smiles_list))

        for i in range(start, end):
            graph, duration_value, effect_onehot, label, ghs_class, smiles = self._get_underling_item(i)
            items.append((graph, duration_value, effect_onehot, label, ghs_class, smiles))
        return items

    def collate(
        self,
        items: List[List[Tuple[Any, float, np.ndarray, float, int, str]]],
    ) -> Tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        graphs_list = []
        duration_values = []
        effect_onehots = []
        labels = []
        ghs_classes = []
        smiles_list = []

        for chunk in items:
            for (g, duration_value, effect_onehot, label, ghs_class, smiles) in chunk:
                if self.is_unimol_preprocessed() and smiles in self.dynamic_failed_smiles:
                    continue
                graphs_list.append(g)
                duration_values.append(duration_value)
                effect_onehots.append(effect_onehot)
                labels.append(label)
                ghs_classes.append(ghs_class)
                smiles_list.append(smiles)

        if self.is_unimol_preprocessed():
            if not hasattr(self, 'unimol_dict'):
                dictionary_path = Path(__file__).resolve().parents[1] / 'data_provider' / 'unimol_dict.txt'
                self.unimol_dict = UnicoreDictionary.load(str(dictionary_path))
                self.unimol_dict.add_symbol("[MASK]", is_special=True)
            atom_vecs, dists, etypes = [], [], []
            for smi in smiles_list:
                av, dist, et = self.load_unimol(smi)
                atom_vecs.append(av)
                dists.append(dist)
                etypes.append(et)
            max_len = max(av.shape[0] for av in atom_vecs)
            pad_idx = self.unimol_dict.pad()
            bsz = len(atom_vecs)
            av_b = torch.full((bsz, max_len), pad_idx, dtype=torch.long)
            dist_b = torch.zeros((bsz, max_len, max_len), dtype=torch.float32)
            et_b = torch.zeros((bsz, max_len, max_len), dtype=torch.long)
            for i in range(bsz):
                n = atom_vecs[i].shape[0]
                av_b[i, :n] = atom_vecs[i]
                dist_b[i, :n, :n] = dists[i]
                et_b[i, :n, :n] = etypes[i]
            batched_graph = (av_b, dist_b, et_b)
        else:
            if dgl is None:
                raise ImportError("DGL is required for graph fallback batching. Use Uni-Mol precomputed inputs instead.")
            batched_graph = dgl.batch(graphs_list)

        duration_values_tensor = torch.tensor(duration_values, dtype=torch.float)
        effect_onehots_tensor = torch.tensor(np.array(effect_onehots), dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        ghs_classes_tensor = torch.tensor(ghs_classes, dtype=torch.long)

        return batched_graph, duration_values_tensor, effect_onehots_tensor, labels_tensor, ghs_classes_tensor, smiles_list

    def preprocess(self):
        if dgl is None:
            raise ImportError("DGL is required for graph preprocessing. Use Uni-Mol precomputed inputs instead.")
        if self.is_preprocessed() or (self.preprocessed_dir is None):
            return

        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        smiles_to_path = {}

        for i in tqdm(range(len(self.smiles_list)), desc=f"Preprocessing {self.split_name}"):
            try:
                smiles = self.smiles_list[i]
                if smiles not in smiles_to_path:
                    g = self.featurize_graph(smiles)
                    pkl_name = f"{self.task}_{len(smiles_to_path)}.pkl"
                    smiles_to_path[smiles] = pkl_name
                    dgl.save_graphs(str(self.preprocessed_dir / pkl_name), [g])
            except Exception as e:
                print(f"Error preprocessing sample {i}: {str(e)}")
                continue

        with open(self.preprocessed_dir / f"smiles_to_path_{self.split_name}.json", "w") as fp:
            json.dump(smiles_to_path, fp)

        self.smiles_to_path = smiles_to_path
        self.featurizer = None

    def get_feature_dimensions(self) -> Dict[str, int]:
        return {
            "num_effects": self.num_effects
        }

    @staticmethod
    def _smi_hash(smi: str) -> str:
        return hashlib.md5(smi.encode()).hexdigest()[:16]

    def _write_unimol_manifests(
        self,
        smiles_to_path: Dict[str, str],
        conformer_metadata: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        report = summarize_conformer_metadata(
            conformer_metadata.values(),
            self.unimol_conformer_settings,
        )
        report["pre_conformer_exclusions"] = {
            "reason": "incompatible with the matched 2D graph featurization pipeline",
            "count": len(DEFAULT_EXCLUDED_SMILES),
            "smiles": list(DEFAULT_EXCLUDED_SMILES),
        }
        payloads = (
            (self._unimol_mapping_path(), smiles_to_path, False),
            (self._unimol_metadata_path(), conformer_metadata, True),
            (self._unimol_report_path(), report, True),
        )
        for path, payload, pretty in payloads:
            tmp_path = path.with_suffix(path.suffix + '.tmp')
            with open(tmp_path, "w") as fp:
                json.dump(
                    payload,
                    fp,
                    ensure_ascii=False,
                    indent=2 if pretty else None,
                )
            tmp_path.replace(path)
        return report

    def preprocess_unimol(self):
        assert self.preprocessed_dir is not None
        unimol_dir = self.preprocessed_dir / "unimol" / self.unimol_cache_tag
        unimol_dir.mkdir(parents=True, exist_ok=True)
        global_json = self._unimol_mapping_path()
        metadata_json = self._unimol_metadata_path()
        report_json = self._unimol_report_path()

        if global_json.exists():
            with open(global_json, "r") as fp:
                smiles_to_path: Dict[str, str] = json.load(fp)
        else:
            smiles_to_path = {}
        if metadata_json.exists():
            with open(metadata_json, "r") as fp:
                conformer_metadata: Dict[str, Dict[str, Any]] = json.load(fp)
        else:
            conformer_metadata = {}

        todo = [
            smi
            for smi in dict.fromkeys(self.smiles_list)
            if smi not in smiles_to_path or smi not in conformer_metadata
        ]
        if not todo:
            self.unimol_smiles_to_path = smiles_to_path
            self.unimol_conformer_metadata = conformer_metadata
            if not report_json.exists():
                self._write_unimol_manifests(smiles_to_path, conformer_metadata)
            return

        if not hasattr(self, 'unimol_dict'):
            dictionary_path = Path(__file__).resolve().parents[1] / 'data_provider' / 'unimol_dict.txt'
            self.unimol_dict = UnicoreDictionary.load(str(dictionary_path))
            self.unimol_dict.add_symbol("[MASK]", is_special=True)

        settings = self.unimol_conformer_settings
        print(
            f"[Uni-Mol preprocess] protocol={self.unimol_cache_tag}, "
            f"new={len(todo)}, cached={len(smiles_to_path)}, "
            f"force_field={settings.mmff_variant}, max_iters={settings.mmff_max_iters}"
        )
        for processed_count, smi in enumerate(
            tqdm(todo, desc="Uni-Mol preprocess"),
            start=1,
        ):
            smi_key = self._smi_hash(smi)
            rel = f"unimol/{self.unimol_cache_tag}/{smi_key}.pt"
            tensor_path = self.preprocessed_dir / rel
            tmp_tensor_path = tensor_path.with_suffix(tensor_path.suffix + '.tmp')
            try:
                unimol_input = build_unimol_input_isolated(
                    smi,
                    self.unimol_dict,
                    settings=settings,
                    max_atoms=256,
                    normalize_coords=True,
                )
                record = unimol_input.conformer_metadata
                torch.save(
                    {
                        'atom_vec': unimol_input.atom_vec,
                        'dist': unimol_input.dist,
                        'edge_type': unimol_input.edge_type,
                        'conformer_metadata': record,
                    },
                    tmp_tensor_path,
                )
            except Exception as e:
                print(f"[WARN] Uni-Mol preprocess failed for '{smi}': {e}")
                pad = torch.tensor([self.unimol_dict.bos(), self.unimol_dict.eos()], dtype=torch.long)
                n = pad.numel()
                dist_fb = torch.zeros((n, n), dtype=torch.float32)
                vocab = len(self.unimol_dict)
                et_fb = pad.view(-1, 1) * vocab + pad.view(1, -1)
                record = {
                    'success': False,
                    'input_smiles': smi,
                    'pipeline_version': self.unimol_cache_tag,
                    'optimization_method': 'failed',
                    'optimization_status': 'failed',
                    'error': f"{type(e).__name__}: {e}",
                }
                torch.save(
                    {
                        'atom_vec': pad,
                        'dist': dist_fb,
                        'edge_type': et_fb,
                        'conformer_metadata': record,
                    },
                    tmp_tensor_path,
                )
                self._append_failed_smiles(smi)
            tmp_tensor_path.replace(tensor_path)
            smiles_to_path[smi] = rel
            conformer_metadata[smi] = record

            if processed_count % 100 == 0:
                self._write_unimol_manifests(smiles_to_path, conformer_metadata)

        report = self._write_unimol_manifests(smiles_to_path, conformer_metadata)

        print(
            "[Uni-Mol preprocess] "
            f"MMFF94={report['optimization_method_counts'].get('MMFF94', 0)}, "
            f"UFF fallback={report['optimization_method_counts'].get('UFF', 0)}, "
            f"ETKDG-only={report['optimization_method_counts'].get('ETKDG-only', 0)}, "
            f"failed={report['failed_molecules']}; report={report_json}"
        )
        self.unimol_smiles_to_path = smiles_to_path
        self.unimol_conformer_metadata = conformer_metadata

    def load_unimol(self, smiles: str):
        if smiles in self._unimol_cache:
            return self._unimol_cache[smiles]
        if self.source_conformer_root is not None:
            molecule_id = self.smiles_to_molecule_id.get(smiles)
            if molecule_id is None:
                raise KeyError(f"SMILES {smiles!r} has no release molecule_id")
            if not hasattr(self, 'unimol_dict'):
                self.unimol_dict = UnicoreDictionary.load(
                    './data_provider/unimol_dict.txt'
                )
                self.unimol_dict.add_symbol("[MASK]", is_special=True)
            result = load_aquatox_unimol_input(
                conformer_path(self.source_conformer_root, molecule_id),
                self.unimol_dict,
                expected_smiles=smiles,
                expected_mmff_energy=self.aquatox_mmff_energy_by_molecule[molecule_id],
                max_atoms=256,
            )
            self._unimol_cache[smiles] = result
            return result
        assert self.preprocessed_dir is not None and self.unimol_smiles_to_path is not None
        rel = self.unimol_smiles_to_path.get(smiles)
        if rel is None:
            raise KeyError(f"SMILES '{smiles}' not found in Uni-Mol cache")
        data = torch.load(self.preprocessed_dir / rel, map_location='cpu')
        result = (data['atom_vec'], data['dist'], data['edge_type'])
        self._unimol_cache[smiles] = result
        return result

    def _append_failed_smiles(self, smi: str):
        if self.failed_smiles_file is None:
            return
        try:
            current: List[str] = []
            if self.failed_smiles_file.exists():
                with open(self.failed_smiles_file, 'r') as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    current = data
                elif isinstance(data, dict) and 'failed' in data:
                    current = data['failed']
            if smi not in current:
                current.append(smi)
                with open(self.failed_smiles_file, 'w') as fp:
                    json.dump(current, fp, ensure_ascii=False, indent=2)
            self.dynamic_failed_smiles.add(smi)
        except Exception as e:
            print(f"[WARN] 写入失败SMILES过滤表出错: {e}")
