from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gin
import torch
from torch import nn
import torch.nn.functional as F

from .unimol import SimpleUniMolModel
from unicore.data import Dictionary as UnicoreDictionary
from data_provider.conformer import (
    ConformerGenerationError,
    ConformerSettings,
    build_unimol_input,
    rdkit_available,
)


def _smiles_to_unimol_inputs(
    smiles_list: List[str],
    dictionary: UnicoreDictionary,
    max_atoms: int = 256,
    remove_hydrogen: bool = True,
    normalize_coords: bool = True,
    conformer_settings: Optional[ConformerSettings] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device('cpu')
    pad_idx = dictionary.pad()

    batch_atom_vec: List[torch.Tensor] = []
    batch_dist: List[torch.Tensor] = []
    batch_edge_type: List[torch.Tensor] = []

    if not remove_hydrogen:
        raise ValueError("The active no-hydrogen Uni-Mol checkpoint requires remove_hydrogen=True")

    if not rdkit_available():
        raise ImportError("RDKit is required to construct Uni-Mol inputs")

    settings = conformer_settings or ConformerSettings()
    for smi in smiles_list:
        if not smi:
            raise ConformerGenerationError("empty SMILES in Uni-Mol batch")

        try:
            unimol_input = build_unimol_input(
                smi,
                dictionary,
                settings=settings,
                max_atoms=max_atoms,
                normalize_coords=normalize_coords,
            )
            batch_atom_vec.append(unimol_input.atom_vec)
            batch_dist.append(unimol_input.dist)
            batch_edge_type.append(unimol_input.edge_type)
        except Exception as exc:
            raise ConformerGenerationError(
                f"failed to construct Uni-Mol input for SMILES {smi!r}: {exc}"
            ) from exc

    max_len = max(t.size(0) for t in batch_atom_vec)
    bsz = len(batch_atom_vec)
    atom_vec_batch = torch.full((bsz, max_len), pad_idx, dtype=torch.long, device=device)
    dist_batch = torch.zeros((bsz, max_len, max_len), dtype=torch.float32, device=device)
    edge_type_batch = torch.zeros((bsz, max_len, max_len), dtype=torch.long, device=device)

    for i in range(bsz):
        n = batch_atom_vec[i].size(0)
        atom_vec_batch[i, :n] = batch_atom_vec[i]
        dist_batch[i, :n, :n] = batch_dist[i]
        edge_type_batch[i, :n, :n] = batch_edge_type[i]

    return atom_vec_batch, dist_batch, edge_type_batch


class FeatureFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_effects: int, dropout: float):
        super().__init__()
        in_dim = hidden_dim + num_effects + 1
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        mol_repr: torch.Tensor,
        duration_values: torch.Tensor,
        effect_onehots: torch.Tensor,
    ) -> torch.Tensor:
        feats = [mol_repr, duration_values.unsqueeze(1), effect_onehots]
        x = torch.cat(feats, dim=1)
        return self.fusion(x)


class MMoELayer(nn.Module):
    def __init__(self, input_dim: int, expert_dim: int, num_experts: int,
                 tasks: List[str], dropout: float):
        super().__init__()
        self.tasks = tasks
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        self.gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(input_dim, num_experts),
            )
            for task in tasks
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        task_reprs = {}
        for task in self.tasks:
            gate_w = torch.softmax(self.gates[task](x), dim=-1)
            task_reprs[task] = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)
        return task_reprs


@gin.configurable()
class ToxicGNN(nn.Module):

    def __init__(
        self,
        *,
        tasks: List[str] = gin.REQUIRED,
        num_effects: int = 7,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        num_experts: int = 6,
        expert_dim: int = 256,
        unimol_args: Dict = gin.REQUIRED,
        unimol_ckpt_path: str = 'weights/mol_pre_no_h_220816.pt',
        freeze_unimol: bool = False,
        unimol_random_seed: int = 42,
        unimol_mmff_variant: str = 'MMFF94',
        unimol_mmff_max_iters: int = 1000,
        unimol_use_uff_fallback: bool = True,
        unimol_uff_max_iters: int = 1000,
        unimol_molecule_timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__()
        self.tasks = tasks
        self.hidden_dim = hidden_dim

        package_root = Path(__file__).resolve().parents[1]
        dictionary_path = package_root / 'data_provider' / 'unimol_dict.txt'
        self.unicore_dict = UnicoreDictionary.load(str(dictionary_path))
        self.unicore_dict.add_symbol("[MASK]", is_special=True)

        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self

        self.unimol_args = AttrDict(unimol_args)
        self.graph_encoder = SimpleUniMolModel(self.unimol_args, self.unicore_dict)
        self.ln_graph = nn.LayerNorm(self.graph_encoder.num_features)

        ckpt_path = Path(unimol_ckpt_path).expanduser()
        if not ckpt_path.is_absolute():
            repo_root = package_root.parents[1]
            package_candidate = package_root / ckpt_path
            repo_candidate = repo_root / ckpt_path
            ckpt_path = next(
                (candidate for candidate in (package_candidate, repo_candidate, ckpt_path) if candidate.exists()),
                repo_candidate,
            )
        ckpt_path = ckpt_path.resolve()
        self.unimol_checkpoint_path = str(ckpt_path)
        if ckpt_path.exists():
            state = torch.load(str(ckpt_path), map_location='cpu')
            model_state = state.get('model', state)
            self.graph_encoder.load_state_dict(model_state, strict=False)
            print(f"Loaded Uni-Mol checkpoint from {ckpt_path}")
        else:
            print(f"Warning: Uni-Mol checkpoint not found at {ckpt_path}. Using random init.")

        for p in self.graph_encoder.parameters():
            p.requires_grad = (not freeze_unimol)

        gnn_out = self.graph_encoder.num_features
        self.proj = nn.Sequential(
            nn.Linear(gnn_out, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.fusion = FeatureFusionLayer(hidden_dim, num_effects, dropout)

        self.mmoe = MMoELayer(
            input_dim=hidden_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            tasks=tasks,
            dropout=dropout,
        )

        self.towers = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(expert_dim, expert_dim // 2),
                nn.LayerNorm(expert_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim // 2, 1),
            )
            for task in tasks
        })

        self.max_atoms = 256
        self.remove_hydrogen = True
        self.normalize_coords = True
        self.unimol_conformer_settings = ConformerSettings(
            random_seed=unimol_random_seed,
            mmff_variant=unimol_mmff_variant,
            mmff_max_iters=unimol_mmff_max_iters,
            use_uff_fallback=unimol_use_uff_fallback,
            uff_max_iters=unimol_uff_max_iters,
            molecule_timeout_seconds=unimol_molecule_timeout_seconds,
        )

    def forward(
        self,
        graph,
        duration_values: torch.Tensor,
        effect_onehots: torch.Tensor,
        smiles_list: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        device = duration_values.device

        if isinstance(graph, tuple) and len(graph) == 3 and all(torch.is_tensor(t) for t in graph):
            atom_vec, dist, edge_type = graph
        else:
            if smiles_list is None:
                raise ValueError("ToxicGNN requires smiles_list or precomputed Uni-Mol tensors")
            atom_vec, dist, edge_type = _smiles_to_unimol_inputs(
                smiles_list, self.unicore_dict,
                max_atoms=self.max_atoms,
                remove_hydrogen=self.remove_hydrogen,
                normalize_coords=self.normalize_coords,
                conformer_settings=self.unimol_conformer_settings,
            )

        atom_vec = atom_vec.to(device)
        dist = dist.to(device)
        edge_type = edge_type.to(device)

        encoder_rep, _ = self.graph_encoder(
            src_tokens=atom_vec,
            src_distance=dist,
            src_edge_type=edge_type,
        )
        cls_repr = self.ln_graph(encoder_rep[:, 0, :])
        mol_repr = self.proj(cls_repr)

        fused = self.fusion(mol_repr, duration_values, effect_onehots)

        task_reprs = self.mmoe(fused)

        return {task: self.towers[task](task_reprs[task]).squeeze(-1) for task in self.tasks}
