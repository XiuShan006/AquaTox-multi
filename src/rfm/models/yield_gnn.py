import math
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List, Literal, Mapping, Optional, Tuple

import dgl
import gin
import torch
from torch import nn
from .gnns import AttentionGNN


TASK_TO_GROUP = {
    "fish_EC50": "fish",
    "fish_EC10": "fish",
    "aquatic_invertebrates_EC50": "aquatic_invertebrates",
    "aquatic_invertebrates_EC10": "aquatic_invertebrates",
    "algae_EC50": "algae",
    "algae_EC10": "algae",
}

MOLECULAR_MODALITIES = ("graph", "ecfp", "maccs", "descriptors")
FINGERPRINT_MODALITY_SLICES = {
    "ecfp": (0, 1024),
    "maccs": (1024, 1191),
    "descriptors": (1191, 1214),
}


def _resolve_molecular_modalities(
    molecular_modalities: Optional[List[str]],
) -> tuple[str, ...]:
    if molecular_modalities is None:
        return MOLECULAR_MODALITIES
    if isinstance(molecular_modalities, (str, bytes)) or not isinstance(
        molecular_modalities, (list, tuple)
    ):
        raise ValueError("molecular_modalities must be a list or None")
    requested = list(molecular_modalities)
    if not requested:
        raise ValueError("molecular_modalities must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("molecular_modalities contains duplicates")
    unknown = sorted(set(requested) - set(MOLECULAR_MODALITIES))
    if unknown:
        raise ValueError(f"Unknown molecular modalities: {unknown}")
    selected = set(requested)
    return tuple(name for name in MOLECULAR_MODALITIES if name in selected)


def _resolve_routed_tasks(
    tasks: List[str],
    sharing: Literal["shared", "group", "task"],
    private_tasks: Optional[List[str]],
    *,
    label: str,
) -> tuple[Dict[str, str], List[str], Optional[List[str]]]:
    if private_tasks is None:
        selected = set(tasks) if sharing != "shared" else set()
        resolved_private_tasks = None
    else:
        if isinstance(private_tasks, (str, bytes)) or not isinstance(
            private_tasks, (list, tuple)
        ):
            raise ValueError(f"{label}_private_tasks must be a list or None")
        requested = list(private_tasks)
        if len(requested) != len(set(requested)):
            raise ValueError(f"{label}_private_tasks contains duplicates")
        unknown = sorted(set(requested) - set(tasks))
        if unknown:
            raise ValueError(
                f"{label}_private_tasks contains tasks not configured in the model: "
                f"{unknown}"
            )
        if sharing == "shared" and requested:
            raise ValueError(
                f"{label}_private_tasks requires non-shared {label}_sharing"
            )
        if sharing != "shared" and not requested:
            raise ValueError(
                f"{label}_private_tasks must be non-empty for routed {label}_sharing"
            )
        selected = set(requested)
        resolved_private_tasks = [task for task in tasks if task in selected]

    routes: Dict[str, str] = {}
    for task in tasks:
        if task not in selected:
            routes[task] = "shared"
        elif sharing == "group":
            if task not in TASK_TO_GROUP:
                raise ValueError(
                    f"Group {label} has no organism mapping for task: {task}"
                )
            routes[task] = TASK_TO_GROUP[task]
        else:
            routes[task] = task

    ordered_routes = list(dict.fromkeys(routes.values()))
    if "shared" in ordered_routes:
        ordered_routes = ["shared", *[r for r in ordered_routes if r != "shared"]]
    return routes, ordered_routes, resolved_private_tasks


class _GradientScale(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.scale, None


class FeatureFusionLayer(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_effects: int = 7,
                 num_media_types: int = 3,
                 dropout: float = 0.2):
        super().__init__()

        input_dim = hidden_dim + 1 + num_effects + num_media_types

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self,
                mol_representation: torch.Tensor,
                duration_values: torch.Tensor,
                effect_onehots: torch.Tensor,
                media_onehots: torch.Tensor) -> torch.Tensor:
        features = [mol_representation]
        features.extend([
            duration_values.unsqueeze(1),
            effect_onehots,
            media_onehots,
        ])

        combined = torch.cat(features, dim=1)
        return self.fusion(combined)


class ResidualBottleneck(nn.Module):

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("bottleneck_dim", bottleneck_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not math.isfinite(float(dropout))
            or not 0.0 <= float(dropout) < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")

        self.input_dim = int(input_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.dropout_probability = float(dropout)
        self.norm = nn.LayerNorm(self.input_dim)
        self.down_projection = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_probability)
        self.up_projection = nn.Linear(self.bottleneck_dim, self.input_dim)

    def reset_identity(self) -> None:
        nn.init.zeros_(self.up_projection.weight)
        if self.up_projection.bias is not None:
            nn.init.zeros_(self.up_projection.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.up_projection(
            self.dropout(self.activation(self.down_projection(self.norm(inputs))))
        )
        return inputs + residual


class GroupedSharingLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tasks: List[str],
        task_to_group: Mapping[str, str],
        dropout: float,
    ) -> None:
        super().__init__()
        resolved_tasks = list(tasks)
        if not resolved_tasks or len(resolved_tasks) != len(set(resolved_tasks)):
            raise ValueError("tasks must contain unique task names")
        missing = sorted(set(resolved_tasks) - set(task_to_group))
        if missing:
            raise ValueError(f"No organism group is configured for tasks: {missing}")
        resolved_mapping = {
            task: str(task_to_group[task]) for task in resolved_tasks
        }
        if any(not group for group in resolved_mapping.values()):
            raise ValueError("task_to_group values must be non-empty")

        self.tasks = resolved_tasks
        self.task_to_group = resolved_mapping
        ordered_groups = list(dict.fromkeys(resolved_mapping.values()))
        self.blocks = nn.ModuleDict(
            {
                group: ResidualBottleneck(input_dim, hidden_dim, dropout)
                for group in ordered_groups
            }
        )

    def reset_identity(self) -> None:
        for block in self.blocks.values():
            block.reset_identity()

    def forward(
        self,
        task_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not task_inputs:
            raise ValueError("GroupedSharingLayer requires at least one task input")
        unknown = sorted(set(task_inputs) - set(self.tasks))
        if unknown:
            raise ValueError(f"Unknown grouped-sharing tasks: {unknown}")

        outputs: Dict[str, torch.Tensor] = {}
        residual_ratios: Dict[str, torch.Tensor] = {}
        for group, block in self.blocks.items():
            group_tasks = [
                task
                for task in self.tasks
                if task in task_inputs and self.task_to_group[task] == group
            ]
            if not group_tasks:
                continue
            batch_sizes = [task_inputs[task].shape[0] for task in group_tasks]
            combined = torch.cat([task_inputs[task] for task in group_tasks], dim=0)
            transformed = block(combined)
            for task, task_output, task_input in zip(
                group_tasks,
                torch.split(transformed, batch_sizes, dim=0),
                [task_inputs[task] for task in group_tasks],
            ):
                outputs[task] = task_output
                residual_ratios[task] = torch.linalg.vector_norm(
                    task_output - task_input, dim=-1
                ) / torch.linalg.vector_norm(task_input, dim=-1).clamp_min(1e-12)
        return outputs, residual_ratios


class GroupMMoEBlock(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tasks: List[str],
        num_experts: int,
        dropout: float,
        gate_temperature: float,
    ) -> None:
        super().__init__()
        if isinstance(num_experts, bool) or not isinstance(num_experts, Integral):
            raise ValueError("num_experts must be an integer")
        if num_experts < 2:
            raise ValueError("Grouped MMoE requires at least two experts")
        if (
            isinstance(gate_temperature, bool)
            or not isinstance(gate_temperature, Real)
            or not math.isfinite(float(gate_temperature))
            or float(gate_temperature) <= 0.0
        ):
            raise ValueError("gate_temperature must be finite and greater than zero")
        resolved_tasks = list(tasks)
        if not resolved_tasks or len(resolved_tasks) != len(set(resolved_tasks)):
            raise ValueError("tasks must contain unique task names")

        self.tasks = resolved_tasks
        self.num_experts = int(num_experts)
        self.gate_temperature = float(gate_temperature)
        self.experts = nn.ModuleList(
            [
                ResidualBottleneck(input_dim, hidden_dim, dropout)
                for _ in range(self.num_experts)
            ]
        )
        self.gates = nn.ModuleDict(
            {
                task: nn.Linear(input_dim, self.num_experts)
                for task in self.tasks
            }
        )

    def reset_identity(self) -> None:
        for expert in self.experts:
            expert.reset_identity()
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(
        self,
        task_inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if not task_inputs:
            raise ValueError("GroupMMoEBlock requires at least one task input")
        unknown = sorted(set(task_inputs) - set(self.tasks))
        if unknown:
            raise ValueError(f"Unknown group-MMoE tasks: {unknown}")

        ordered_tasks = [task for task in self.tasks if task in task_inputs]
        batch_sizes = [task_inputs[task].shape[0] for task in ordered_tasks]
        combined = torch.cat([task_inputs[task] for task in ordered_tasks], dim=0)
        expert_outputs = torch.stack(
            [expert(combined) for expert in self.experts], dim=1
        )
        split_outputs = torch.split(expert_outputs, batch_sizes, dim=0)

        outputs: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}
        for task, task_expert_outputs in zip(ordered_tasks, split_outputs):
            weights = torch.softmax(
                self.gates[task](task_inputs[task]) / self.gate_temperature,
                dim=-1,
            )
            gate_weights[task] = weights
            outputs[task] = (
                weights.unsqueeze(-1) * task_expert_outputs
            ).sum(dim=1)
        return outputs, gate_weights


class GroupedMMoESharingLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tasks: List[str],
        task_to_group: Mapping[str, str],
        dropout: float,
        num_experts: int,
        gate_temperature: float,
    ) -> None:
        super().__init__()
        resolved_tasks = list(tasks)
        if not resolved_tasks or len(resolved_tasks) != len(set(resolved_tasks)):
            raise ValueError("tasks must contain unique task names")
        missing = sorted(set(resolved_tasks) - set(task_to_group))
        if missing:
            raise ValueError(f"No organism group is configured for tasks: {missing}")
        self.tasks = resolved_tasks
        self.task_to_group = {
            task: str(task_to_group[task]) for task in resolved_tasks
        }
        ordered_groups = list(dict.fromkeys(self.task_to_group.values()))
        self.blocks = nn.ModuleDict(
            {
                group: GroupMMoEBlock(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    tasks=[
                        task
                        for task in self.tasks
                        if self.task_to_group[task] == group
                    ],
                    num_experts=num_experts,
                    dropout=dropout,
                    gate_temperature=gate_temperature,
                )
                for group in ordered_groups
            }
        )

    def reset_identity(self) -> None:
        for block in self.blocks.values():
            block.reset_identity()

    def forward(
        self,
        task_inputs: Dict[str, torch.Tensor],
        *,
        return_gate_weights: bool = False,
    ) -> (
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        | Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
        ]
    ):
        if not task_inputs:
            raise ValueError("GroupedMMoESharingLayer requires at least one task input")
        unknown = sorted(set(task_inputs) - set(self.tasks))
        if unknown:
            raise ValueError(f"Unknown grouped-MMoE tasks: {unknown}")

        outputs: Dict[str, torch.Tensor] = {}
        residual_ratios: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}
        for group, block in self.blocks.items():
            group_inputs = {
                task: task_inputs[task]
                for task in block.tasks
                if task in task_inputs
            }
            if not group_inputs:
                continue
            group_outputs, group_gate_weights = block(group_inputs)
            gate_weights.update(group_gate_weights)
            for task, task_output in group_outputs.items():
                task_input = task_inputs[task]
                outputs[task] = task_output
                residual_ratios[task] = torch.linalg.vector_norm(
                    task_output - task_input, dim=-1
                ) / torch.linalg.vector_norm(task_input, dim=-1).clamp_min(1e-12)
        if return_gate_weights:
            if set(gate_weights) != set(task_inputs):
                raise RuntimeError("Grouped MMoE gate export is incomplete")
            return outputs, residual_ratios, gate_weights
        return outputs, residual_ratios


class MMoELayer(nn.Module):

    def __init__(self, input_dim: int, expert_dim: int, num_experts: int,
                 tasks: List[str], dropout: float,
                 expert_hidden_dim: Optional[int] = None,
                 mmoe_dropout: Optional[float] = None,
                 gate_temperature: float = 1.0):
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("expert_dim", expert_dim),
            ("num_experts", num_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        resolved_expert_hidden_dim = (
            expert_dim if expert_hidden_dim is None else expert_hidden_dim
        )
        if (
            isinstance(resolved_expert_hidden_dim, bool)
            or not isinstance(resolved_expert_hidden_dim, Integral)
            or resolved_expert_hidden_dim <= 0
        ):
            raise ValueError("expert_hidden_dim must be a positive integer")

        resolved_mmoe_dropout = dropout if mmoe_dropout is None else mmoe_dropout
        if (
            isinstance(resolved_mmoe_dropout, bool)
            or not isinstance(resolved_mmoe_dropout, Real)
            or not math.isfinite(float(resolved_mmoe_dropout))
            or not 0.0 <= float(resolved_mmoe_dropout) < 1.0
        ):
            raise ValueError("mmoe_dropout must be finite and in [0, 1)")
        if (
            isinstance(gate_temperature, bool)
            or not isinstance(gate_temperature, Real)
            or not math.isfinite(float(gate_temperature))
            or float(gate_temperature) <= 0.0
        ):
            raise ValueError("gate_temperature must be finite and greater than zero")

        resolved_tasks = list(tasks)
        if not resolved_tasks or any(
            not isinstance(task, str) or not task for task in resolved_tasks
        ):
            raise ValueError("tasks must contain at least one non-empty task name")
        if len(resolved_tasks) != len(set(resolved_tasks)):
            raise ValueError("tasks must not contain duplicates")

        self.tasks = resolved_tasks
        self.num_experts = int(num_experts)
        self.expert_dim = int(expert_dim)
        self.expert_hidden_dim = int(resolved_expert_hidden_dim)
        self.mmoe_dropout = float(resolved_mmoe_dropout)
        self.gate_temperature = float(gate_temperature)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.expert_hidden_dim),
                nn.LayerNorm(self.expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.mmoe_dropout),
                nn.Linear(self.expert_hidden_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.ReLU(),
                nn.Dropout(self.mmoe_dropout),
            )
            for _ in range(num_experts)
        ])

        self.gates = nn.ModuleDict({
            task: nn.Linear(input_dim, num_experts)
            for task in self.tasks
        })

    def forward(
        self,
        task_inputs: Dict[str, torch.Tensor],
        *,
        return_diagnostics: bool = False,
    ) -> (
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        | Tuple[
            Dict[str, torch.Tensor],
            Dict[str, torch.Tensor],
            Dict[str, Dict[str, torch.Tensor]],
        ]
    ):
        if not task_inputs:
            raise ValueError("MMoELayer requires at least one task input")
        unknown = sorted(set(task_inputs) - set(self.tasks))
        if unknown:
            raise ValueError(f"Unknown MMoE tasks: {unknown}")

        ordered_tasks = [task for task in self.tasks if task in task_inputs]
        batch_sizes = [task_inputs[task].shape[0] for task in ordered_tasks]
        concatenated = torch.cat([task_inputs[task] for task in ordered_tasks], dim=0)
        all_expert_outputs = torch.stack(
            [expert(concatenated) for expert in self.experts], dim=1
        )
        split_expert_outputs = torch.split(all_expert_outputs, batch_sizes, dim=0)

        representations: Dict[str, torch.Tensor] = {}
        gate_weights: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, Dict[str, torch.Tensor]] = {}
        for task, expert_outputs in zip(ordered_tasks, split_expert_outputs):
            gate_logits = self.gates[task](task_inputs[task])
            weights = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
            gate_weights[task] = weights
            representations[task] = (
                weights.unsqueeze(-1) * expert_outputs
            ).sum(dim=1)
            if return_diagnostics:
                centered = expert_outputs.float() - expert_outputs.float().mean(
                    dim=0, keepdim=True
                )
                flattened = centered.permute(1, 0, 2).reshape(
                    self.num_experts, -1
                )
                normalized = flattened / torch.linalg.vector_norm(
                    flattened, dim=1, keepdim=True
                ).clamp_min(torch.finfo(flattened.dtype).eps)
                pair_indices = torch.triu_indices(
                    self.num_experts,
                    self.num_experts,
                    offset=1,
                    device=flattened.device,
                )
                pair_cosines = (normalized @ normalized.transpose(0, 1))[
                    pair_indices[0], pair_indices[1]
                ]
                gate_probabilities = weights.float().clamp_min(
                    torch.finfo(torch.float32).eps
                )
                diagnostics[task] = {
                    "expert_diversity_loss": pair_cosines.square().mean(),
                    "expert_pair_cosines": pair_cosines,
                    "gate_entropy": -(
                        gate_probabilities * gate_probabilities.log()
                    ).sum(dim=-1).mean(),
                }
        if return_diagnostics:
            return representations, gate_weights, diagnostics
        return representations, gate_weights


class MolecularEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_attention_heads: int,
        edge_in_dim: int,
        node_in_dim: int,
        fp_dim: int,
        use_attention: bool,
        attention_dropout: float,
        gnn_type: str,
        dropout: float,
        fusion_mode: Literal["direct", "projected"] = "direct",
        modality_dropout: float = 0.0,
        jk_mode: Literal["last", "weighted_2_4"] = "last",
        pooling_mode: Literal["gated", "mean_max_gated"] = "gated",
        message_dropout: float = 0.0,
        train_eps: bool = False,
    ):
        super().__init__()
        if fusion_mode not in {"direct", "projected"}:
            raise ValueError("fusion_mode must be 'direct' or 'projected'")
        if not 0.0 <= modality_dropout < 1.0:
            raise ValueError("modality_dropout must be in [0, 1)")
        if fusion_mode == "projected" and fp_dim != 1214:
            raise ValueError("projected fusion requires fp_dim=1214")
        self.fusion_mode = fusion_mode
        self.modality_dropout = float(modality_dropout)
        self.hidden_dim = int(hidden_dim)
        self.fp_dim = int(fp_dim)
        self.molecular_modalities = MOLECULAR_MODALITIES
        self.graph_encoder = AttentionGNN(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            edge_in_dim=edge_in_dim,
            node_in_dim=node_in_dim,
            use_attention=use_attention,
            attention_dropout=attention_dropout,
            gnn_type=gnn_type,
            jk_mode=jk_mode,
            pooling_mode=pooling_mode,
            message_dropout=message_dropout,
            train_eps=train_eps,
        )
        if self.fusion_mode == "direct":
            self.fp_proj = nn.Sequential(
                nn.Linear(hidden_dim + fp_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.graph_proj = None
            self.morgan_proj = None
            self.maccs_proj = None
            self.descriptor_proj = None
            self.projected_fusion = None
        else:
            self.fp_proj = None
            self.graph_proj = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.ReLU()
            )
            self.morgan_proj = nn.Sequential(
                nn.Linear(1024, 128), nn.LayerNorm(128), nn.ReLU()
            )
            self.maccs_proj = nn.Sequential(
                nn.Linear(167, 64), nn.LayerNorm(64), nn.ReLU()
            )
            self.descriptor_proj = nn.Sequential(
                nn.Linear(23, 64), nn.LayerNorm(64), nn.ReLU()
            )
            self.projected_fusion = nn.Sequential(
                nn.Linear(512, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.register_buffer("descriptor_mean", torch.zeros(23))
            self.register_buffer("descriptor_scale", torch.ones(23))
            self.register_buffer(
                "descriptor_scaler_fitted", torch.tensor(False, dtype=torch.bool)
            )

    def configure_modalities(self, molecular_modalities: tuple[str, ...]) -> None:
        resolved = _resolve_molecular_modalities(list(molecular_modalities))
        if self.molecular_modalities != MOLECULAR_MODALITIES:
            raise RuntimeError("Molecular modalities have already been configured")
        if resolved == MOLECULAR_MODALITIES:
            return
        if self.fusion_mode != "direct":
            raise ValueError("Molecular ablations require fusion_mode='direct'")
        if self.fp_dim != 1214:
            raise ValueError("Molecular ablations require fp_dim=1214")
        if self.fp_proj is None or not isinstance(self.fp_proj[0], nn.Linear):
            raise RuntimeError("Direct molecular fusion projection is unavailable")

        selected_columns: List[int] = []
        if "graph" in resolved:
            selected_columns.extend(range(self.hidden_dim))
        for modality in MOLECULAR_MODALITIES[1:]:
            if modality in resolved:
                start, stop = FINGERPRINT_MODALITY_SLICES[modality]
                selected_columns.extend(
                    range(self.hidden_dim + start, self.hidden_dim + stop)
                )

        old_projection = self.fp_proj[0]
        new_projection = nn.Linear(
            len(selected_columns),
            old_projection.out_features,
            bias=old_projection.bias is not None,
            device=old_projection.weight.device,
            dtype=old_projection.weight.dtype,
        )
        column_index = torch.tensor(
            selected_columns,
            dtype=torch.long,
            device=old_projection.weight.device,
        )
        with torch.no_grad():
            new_projection.weight.copy_(
                old_projection.weight.index_select(1, column_index)
            )
            if old_projection.bias is not None and new_projection.bias is not None:
                new_projection.bias.copy_(old_projection.bias)
        self.fp_proj[0] = new_projection
        if "graph" not in resolved:
            self.graph_encoder = None
        self.molecular_modalities = resolved

    @property
    def requires_descriptor_scaler(self) -> bool:
        return self.fusion_mode == "projected"

    def set_descriptor_scaler(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        if not self.requires_descriptor_scaler:
            raise RuntimeError("Descriptor scaling is only used by projected fusion")
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if mean.numel() != 23 or scale.numel() != 23:
            raise ValueError("Descriptor scaler must contain 23 means and scales")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("Descriptor scaler contains non-finite values")
        if (scale <= 0).any():
            raise ValueError("Descriptor scaler scales must be positive")
        self.descriptor_mean.copy_(mean.to(self.descriptor_mean.device))
        self.descriptor_scale.copy_(scale.to(self.descriptor_scale.device))
        self.descriptor_scaler_fitted.fill_(True)

    def _apply_modality_dropout(
        self,
        branches: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if not self.training or self.modality_dropout <= 0:
            return branches
        batch_size = branches[0].shape[0]
        suppress = torch.rand(batch_size, device=branches[0].device) < self.modality_dropout
        choices = torch.randint(0, len(branches), (batch_size,), device=branches[0].device)
        outputs = []
        for branch_index, branch in enumerate(branches):
            keep = ~(suppress & choices.eq(branch_index))
            outputs.append(branch * keep.unsqueeze(-1).to(branch.dtype))
        return outputs

    def encode_modalities(
        self, graph: dgl.DGLGraph, *, graph_route: str | None = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        graph_repr = None
        if "graph" in self.molecular_modalities:
            if self.graph_encoder is None:
                raise RuntimeError("Graph modality is active but graph encoder is unavailable")
            graph_repr = self.graph_encoder(graph, route=graph_route)
        elif graph_route is not None:
            raise ValueError("Graph routing cannot be used when graph is disabled")

        fingerprint_modalities = [
            modality
            for modality in MOLECULAR_MODALITIES[1:]
            if modality in self.molecular_modalities
        ]
        fp = None
        if fingerprint_modalities:
            full_fp = dgl.mean_nodes(graph, "fp")
            if self.molecular_modalities == MOLECULAR_MODALITIES:
                fp = full_fp
            else:
                fp = torch.cat(
                    [
                        full_fp[:, slice(*FINGERPRINT_MODALITY_SLICES[modality])]
                        for modality in fingerprint_modalities
                    ],
                    dim=-1,
                )
        return graph_repr, fp

    def fuse_modalities(
        self,
        graph_repr: Optional[torch.Tensor],
        fp: Optional[torch.Tensor],
        *,
        direct_projection: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if self.fusion_mode == "direct":
            projection = self.fp_proj if direct_projection is None else direct_projection
            if projection is None:
                raise RuntimeError("Direct molecular fusion projection is unavailable")
            branches = [branch for branch in (graph_repr, fp) if branch is not None]
            if not branches:
                raise RuntimeError("No molecular modalities are available")
            return projection(torch.cat(branches, dim=-1))

        if direct_projection is not None:
            raise ValueError("A routed direct projection cannot be used in projected mode")

        if not bool(self.descriptor_scaler_fitted.item()):
            raise RuntimeError("Projected fusion descriptor scaler has not been fitted")
        if graph_repr is None or fp is None:
            raise RuntimeError("Projected fusion requires every molecular modality")
        morgan = fp[:, :1024]
        maccs = fp[:, 1024:1191]
        descriptors = (fp[:, 1191:] - self.descriptor_mean) / self.descriptor_scale
        assert self.graph_proj is not None
        assert self.morgan_proj is not None
        assert self.maccs_proj is not None
        assert self.descriptor_proj is not None
        assert self.projected_fusion is not None
        branches = self._apply_modality_dropout(
            [
                self.graph_proj(graph_repr),
                self.morgan_proj(morgan),
                self.maccs_proj(maccs),
                self.descriptor_proj(descriptors),
            ]
        )
        return self.projected_fusion(torch.cat(branches, dim=-1))

    def forward(self, graph: dgl.DGLGraph) -> torch.Tensor:
        graph_repr, fp = self.encode_modalities(graph)
        return self.fuse_modalities(graph_repr, fp)


def _regression_output(
    mu: torch.Tensor,
    raw_log_sigma: Optional[torch.Tensor],
    regression_mode: str,
    sigma_parameterization: str,
    min_log_sigma: float,
    max_log_sigma: float,
) -> Dict[str, torch.Tensor]:
    output = {"mu": mu.reshape(-1)}
    if regression_mode == "heteroscedastic":
        if raw_log_sigma is None:
            raise ValueError("Heteroscedastic output requires a sigma head")
        raw_log_sigma = raw_log_sigma.reshape(-1)
        if sigma_parameterization == "clamp":
            output["log_sigma"] = raw_log_sigma.clamp(
                min=min_log_sigma, max=max_log_sigma
            )
        elif sigma_parameterization == "sigmoid":
            output["log_sigma"] = min_log_sigma + (
                max_log_sigma - min_log_sigma
            ) * torch.sigmoid(raw_log_sigma)
        else:
            raise ValueError(
                "sigma_parameterization must be 'clamp' or 'sigmoid'"
            )
    return output


def _validate_regression_mode(regression_mode: str) -> str:
    allowed = {"heteroscedastic", "deterministic"}
    if regression_mode not in allowed:
        raise ValueError(
            f"regression_mode must be one of {sorted(allowed)}, got {regression_mode!r}"
        )
    return regression_mode


@gin.configurable()
class YieldGNN(nn.Module):
    def __init__(
        self,
        num_effects: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        edge_in_dim: int = 13,
        node_in_dim: int = 81,
        fp_dim: int = 1214,
        use_attention: bool = True,
        num_experts: int = 6,
        tasks: List[str] = ['MOR', 'DVP', 'ITX', 'GRO', 'MPH', 'REP', 'POP'],
        expert_dim: int = 256,
        dropout: float = 0.2,
        checkpoint_path: Path | str | None = None,
        concat_type: Literal["simple", "fancy"] = "simple",
        mlp_dropout: float = 0.4,
        attention_dropout: float = 0.1,
        gnn_type: str = "our_gine",
        constant_effect_tasks: List[str] = None,
        regression_mode: Literal["heteroscedastic", "deterministic"] = "heteroscedastic",
        sigma_parameterization: Literal["clamp", "sigmoid"] = "clamp",
        min_log_sigma: float = -0.5,
        max_log_sigma: float = 2.0,
        fusion_mode: Literal["direct", "projected"] = "direct",
        modality_dropout: float = 0.0,
        jk_mode: Literal["last", "weighted_2_4"] = "last",
        pooling_mode: Literal["gated", "mean_max_gated"] = "gated",
        message_dropout: float = 0.0,
        train_eps: bool = False,
        initialization_seed: int = 20260825,
        expert_hidden_dim: Optional[int] = None,
        mmoe_dropout: Optional[float] = None,
        gate_temperature: float = 1.0,
        sharing_mode: Literal["global_mmoe", "grouped"] = "global_mmoe",
        use_task_adapters: bool = False,
        group_hidden_dim: int = 128,
        group_sharing_type: Literal["residual", "mmoe"] = "residual",
        group_num_experts: int = 2,
        group_gate_temperature: float = 1.0,
        adapter_dim: int = 32,
        group_dropout: float = 0.1,
        adapter_dropout: float = 0.1,
        condition_fusion_sharing: Literal["shared", "group", "task"] = "shared",
        condition_fusion_private_tasks: Optional[List[str]] = None,
        use_prefusion_task_adapters: bool = False,
        prefusion_adapter_tasks: Optional[List[str]] = None,
        prefusion_adapter_dim: int = 32,
        prefusion_adapter_dropout: float = 0.1,
        molecular_fusion_sharing: Literal["shared", "group", "task"] = "shared",
        molecular_fusion_private_tasks: Optional[List[str]] = None,
        molecular_gradient_scales: Optional[Mapping[str, float]] = None,
        graph_message_sharing: Literal["shared", "group", "task"] = "shared",
        graph_message_private_tasks: Optional[List[str]] = None,
        graph_private_start_layer: int = 2,
        molecular_modalities: Optional[List[str]] = None,
    ):
        super().__init__()

        resolved_molecular_modalities = _resolve_molecular_modalities(
            molecular_modalities
        )
        if resolved_molecular_modalities != MOLECULAR_MODALITIES:
            if fp_dim != 1214:
                raise ValueError("Molecular ablations require fp_dim=1214")
            if fusion_mode != "direct":
                raise ValueError("Molecular ablations require fusion_mode='direct'")
            if molecular_fusion_sharing != "shared":
                raise ValueError(
                    "Molecular ablations require molecular_fusion_sharing='shared'"
                )
            if graph_message_sharing != "shared":
                raise ValueError(
                    "Molecular ablations require graph_message_sharing='shared'"
                )

        self.constant_effect_tasks = frozenset(
            constant_effect_tasks if constant_effect_tasks is not None
            else ["fish_EC50", "algae_EC50", "algae_EC10"]
        )

        self.concat_type = concat_type
        self.mlp_dropout = mlp_dropout
        self.attention_dropout = attention_dropout
        self.gnn_type = gnn_type
        self.tasks = list(tasks)
        if sharing_mode not in {"global_mmoe", "grouped"}:
            raise ValueError("sharing_mode must be 'global_mmoe' or 'grouped'")
        if group_sharing_type not in {"residual", "mmoe"}:
            raise ValueError("group_sharing_type must be 'residual' or 'mmoe'")
        if not isinstance(use_task_adapters, bool):
            raise ValueError("use_task_adapters must be a bool")
        if condition_fusion_sharing not in {"shared", "group", "task"}:
            raise ValueError(
                "condition_fusion_sharing must be 'shared', 'group', or 'task'"
            )
        if molecular_fusion_sharing not in {"shared", "group", "task"}:
            raise ValueError(
                "molecular_fusion_sharing must be 'shared', 'group', or 'task'"
            )
        if graph_message_sharing not in {"shared", "group", "task"}:
            raise ValueError(
                "graph_message_sharing must be 'shared', 'group', or 'task'"
            )
        if molecular_fusion_sharing != "shared" and fusion_mode != "direct":
            raise ValueError(
                "Routed molecular fusion currently requires fusion_mode='direct'"
            )
        if not isinstance(use_prefusion_task_adapters, bool):
            raise ValueError("use_prefusion_task_adapters must be a bool")
        if molecular_gradient_scales is None:
            resolved_molecular_gradient_scales = {task: 1.0 for task in self.tasks}
            recorded_molecular_gradient_scales = None
        elif isinstance(molecular_gradient_scales, Mapping):
            provided_gradient_scales = dict(molecular_gradient_scales)
            unknown_gradient_tasks = sorted(
                set(provided_gradient_scales) - set(self.tasks)
            )
            if unknown_gradient_tasks:
                raise ValueError(
                    "molecular_gradient_scales contains tasks not configured in the model: "
                    f"{unknown_gradient_tasks}"
                )
            resolved_molecular_gradient_scales = {}
            for task in self.tasks:
                value = provided_gradient_scales.get(task, 1.0)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(
                        f"molecular_gradient_scales[{task!r}] must be finite and in [0, 1]"
                    )
                resolved_molecular_gradient_scales[task] = float(value)
            recorded_molecular_gradient_scales = {
                task: resolved_molecular_gradient_scales[task]
                for task in self.tasks
                if task in provided_gradient_scales
            }
        else:
            raise ValueError("molecular_gradient_scales must be a mapping or None")
        if prefusion_adapter_tasks is None:
            resolved_prefusion_adapter_tasks = (
                list(self.tasks) if use_prefusion_task_adapters else []
            )
        else:
            if isinstance(prefusion_adapter_tasks, (str, bytes)) or not isinstance(
                prefusion_adapter_tasks, (list, tuple)
            ):
                raise ValueError("prefusion_adapter_tasks must be a list or None")
            requested_prefusion_tasks = list(prefusion_adapter_tasks)
            if len(requested_prefusion_tasks) != len(set(requested_prefusion_tasks)):
                raise ValueError("prefusion_adapter_tasks contains duplicates")
            unknown_prefusion_tasks = sorted(
                set(requested_prefusion_tasks) - set(self.tasks)
            )
            if unknown_prefusion_tasks:
                raise ValueError(
                    "prefusion_adapter_tasks contains tasks not configured in the model: "
                    f"{unknown_prefusion_tasks}"
                )
            if requested_prefusion_tasks and not use_prefusion_task_adapters:
                raise ValueError(
                    "prefusion_adapter_tasks requires use_prefusion_task_adapters=True"
                )
            if use_prefusion_task_adapters and not requested_prefusion_tasks:
                raise ValueError(
                    "prefusion_adapter_tasks must be non-empty when adapters are enabled"
                )
            resolved_prefusion_adapter_tasks = [
                task for task in self.tasks if task in set(requested_prefusion_tasks)
            ]
        for name, value in (
            ("group_hidden_dim", group_hidden_dim),
            ("group_num_experts", group_num_experts),
            ("adapter_dim", adapter_dim),
            ("prefusion_adapter_dim", prefusion_adapter_dim),
            ("graph_private_start_layer", graph_private_start_layer),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if group_sharing_type == "mmoe" and group_num_experts < 2:
            raise ValueError("Grouped MMoE requires at least two experts")
        for name, value in (
            ("group_dropout", group_dropout),
            ("adapter_dropout", adapter_dropout),
            ("prefusion_adapter_dropout", prefusion_adapter_dropout),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if (
            isinstance(group_gate_temperature, bool)
            or not isinstance(group_gate_temperature, Real)
            or not math.isfinite(float(group_gate_temperature))
            or float(group_gate_temperature) <= 0.0
        ):
            raise ValueError(
                "group_gate_temperature must be finite and greater than zero"
            )
        if sharing_mode == "grouped" and hidden_dim != expert_dim:
            raise ValueError(
                "grouped sharing requires hidden_dim == expert_dim for task-trunk compatibility"
            )

        self.sharing_mode = sharing_mode
        self.use_task_adapters = use_task_adapters
        self.group_sharing_type = group_sharing_type
        self.condition_fusion_sharing = condition_fusion_sharing
        self.use_prefusion_task_adapters = use_prefusion_task_adapters
        self.prefusion_adapter_tasks = frozenset(resolved_prefusion_adapter_tasks)
        (
            self.condition_fusion_routes,
            ordered_fusion_routes,
            resolved_condition_private_tasks,
        ) = _resolve_routed_tasks(
            self.tasks,
            self.condition_fusion_sharing,
            condition_fusion_private_tasks,
            label="condition_fusion",
        )
        self.primary_condition_fusion_route = ordered_fusion_routes[0]
        self.molecular_fusion_sharing = molecular_fusion_sharing
        self.molecular_gradient_scales = resolved_molecular_gradient_scales
        (
            self.molecular_fusion_routes,
            ordered_molecular_fusion_routes,
            resolved_molecular_private_tasks,
        ) = _resolve_routed_tasks(
            self.tasks,
            self.molecular_fusion_sharing,
            molecular_fusion_private_tasks,
            label="molecular_fusion",
        )
        self.primary_molecular_fusion_route = ordered_molecular_fusion_routes[0]
        self.graph_message_sharing = graph_message_sharing
        (
            self.graph_message_routes,
            ordered_graph_message_routes,
            resolved_graph_private_tasks,
        ) = _resolve_routed_tasks(
            self.tasks,
            self.graph_message_sharing,
            graph_message_private_tasks,
            label="graph_message",
        )
        self.primary_graph_message_route = ordered_graph_message_routes[0]
        self.graph_private_start_layer = int(graph_private_start_layer)
        if self.graph_message_sharing != "shared" and not (
            0 < self.graph_private_start_layer < num_layers
        ):
            raise ValueError(
                "graph_private_start_layer must split the configured GNN layers"
            )
        self.supports_routing_diagnostics = True
        self.model_kind = "mmoe" if sharing_mode == "global_mmoe" else "grouped"
        self.regression_mode = _validate_regression_mode(regression_mode)
        if sigma_parameterization not in {"clamp", "sigmoid"}:
            raise ValueError("sigma_parameterization must be 'clamp' or 'sigmoid'")
        if not min_log_sigma < max_log_sigma:
            raise ValueError("min_log_sigma must be smaller than max_log_sigma")
        self.sigma_parameterization = sigma_parameterization
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.num_effects = num_effects
        self.hidden_dim = hidden_dim
        self.fp_dim = fp_dim
        self.molecular_modalities = resolved_molecular_modalities
        if (
            isinstance(expert_dim, bool)
            or not isinstance(expert_dim, Integral)
            or expert_dim < 2
        ):
            raise ValueError("expert_dim must be an integer greater than or equal to 2")
        resolved_expert_hidden_dim = (
            expert_dim if expert_hidden_dim is None else expert_hidden_dim
        )
        resolved_mmoe_dropout = dropout if mmoe_dropout is None else mmoe_dropout
        self.model_spec = {
            "num_effects": num_effects,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "edge_in_dim": edge_in_dim,
            "node_in_dim": node_in_dim,
            "fp_dim": fp_dim,
            "use_attention": use_attention,
            "num_experts": num_experts,
            "tasks": list(tasks),
            "expert_dim": expert_dim,
            "dropout": dropout,
            "expert_hidden_dim": resolved_expert_hidden_dim,
            "mmoe_dropout": resolved_mmoe_dropout,
            "gate_temperature": gate_temperature,
            "concat_type": concat_type,
            "mlp_dropout": mlp_dropout,
            "attention_dropout": attention_dropout,
            "gnn_type": gnn_type,
            "constant_effect_tasks": sorted(self.constant_effect_tasks),
            "regression_mode": self.regression_mode,
            "sigma_parameterization": self.sigma_parameterization,
            "min_log_sigma": self.min_log_sigma,
            "max_log_sigma": self.max_log_sigma,
            "fusion_mode": fusion_mode,
            "modality_dropout": modality_dropout,
            "jk_mode": jk_mode,
            "pooling_mode": pooling_mode,
            "message_dropout": message_dropout,
            "train_eps": bool(train_eps),
            "initialization_seed": int(initialization_seed),
            "sharing_mode": self.sharing_mode,
            "use_task_adapters": self.use_task_adapters,
            "group_hidden_dim": int(group_hidden_dim),
            "group_sharing_type": self.group_sharing_type,
            "group_num_experts": int(group_num_experts),
            "group_gate_temperature": float(group_gate_temperature),
            "adapter_dim": int(adapter_dim),
            "group_dropout": float(group_dropout),
            "adapter_dropout": float(adapter_dropout),
            "condition_fusion_sharing": self.condition_fusion_sharing,
            "condition_fusion_private_tasks": resolved_condition_private_tasks,
            "use_prefusion_task_adapters": self.use_prefusion_task_adapters,
            "prefusion_adapter_tasks": (
                None
                if prefusion_adapter_tasks is None
                else list(resolved_prefusion_adapter_tasks)
            ),
            "prefusion_adapter_dim": int(prefusion_adapter_dim),
            "prefusion_adapter_dropout": float(prefusion_adapter_dropout),
            "molecular_fusion_sharing": self.molecular_fusion_sharing,
            "molecular_fusion_private_tasks": resolved_molecular_private_tasks,
            "molecular_gradient_scales": recorded_molecular_gradient_scales,
            "graph_message_sharing": self.graph_message_sharing,
            "graph_message_private_tasks": resolved_graph_private_tasks,
            "graph_private_start_layer": self.graph_private_start_layer,
            "molecular_modalities": list(self.molecular_modalities),
        }

        self.feature_fusion = FeatureFusionLayer(
            hidden_dim=hidden_dim,
            num_effects=num_effects,
            num_media_types=3,
            dropout=dropout
        )

        self.molecular_encoder = MolecularEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            edge_in_dim=edge_in_dim,
            node_in_dim=node_in_dim,
            fp_dim=fp_dim,
            use_attention=use_attention,
            attention_dropout=attention_dropout,
            gnn_type=gnn_type,
            dropout=dropout,
            fusion_mode=fusion_mode,
            modality_dropout=modality_dropout,
            jk_mode=jk_mode,
            pooling_mode=pooling_mode,
            message_dropout=message_dropout,
            train_eps=train_eps,
        )

        if self.sharing_mode == "global_mmoe":
            self.mmoe: Optional[MMoELayer] = MMoELayer(
                input_dim=hidden_dim,
                expert_dim=expert_dim,
                num_experts=num_experts,
                tasks=self.tasks,
                dropout=dropout,
                expert_hidden_dim=expert_hidden_dim,
                mmoe_dropout=mmoe_dropout,
                gate_temperature=gate_temperature,
            )
            self.grouped_sharing: Optional[GroupedSharingLayer] = None
            self.model_spec["expert_hidden_dim"] = self.mmoe.expert_hidden_dim
            self.model_spec["mmoe_dropout"] = self.mmoe.mmoe_dropout
            self.model_spec["gate_temperature"] = self.mmoe.gate_temperature
        else:
            self.mmoe = None
            if self.group_sharing_type == "mmoe":
                self.grouped_sharing = GroupedMMoESharingLayer(
                    input_dim=hidden_dim,
                    hidden_dim=group_hidden_dim,
                    tasks=self.tasks,
                    task_to_group=TASK_TO_GROUP,
                    dropout=group_dropout,
                    num_experts=group_num_experts,
                    gate_temperature=group_gate_temperature,
                )
            else:
                self.grouped_sharing = GroupedSharingLayer(
                    input_dim=hidden_dim,
                    hidden_dim=group_hidden_dim,
                    tasks=self.tasks,
                    task_to_group=TASK_TO_GROUP,
                    dropout=group_dropout,
                )

        routed_dim = expert_dim if self.sharing_mode == "global_mmoe" else hidden_dim
        self.task_adapters = nn.ModuleDict(
            {
                task: ResidualBottleneck(routed_dim, adapter_dim, adapter_dropout)
                for task in self.tasks
            }
            if self.use_task_adapters
            else {}
        )

        self.task_trunks = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(routed_dim, routed_dim // 2),
                nn.LayerNorm(routed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for task in self.tasks
        })
        self.mu_heads = nn.ModuleDict({
            task: nn.Linear(routed_dim // 2, 1) for task in self.tasks
        })
        self.sigma_heads = nn.ModuleDict(
            {
                task: nn.Linear(routed_dim // 2, 1)
                for task in self.tasks
            }
            if self.regression_mode == "heteroscedastic"
            else {}
        )

        self.condition_fusions = nn.ModuleDict(
            {
                route: FeatureFusionLayer(
                    hidden_dim=hidden_dim,
                    num_effects=num_effects,
                    num_media_types=3,
                    dropout=dropout,
                )
                for route in ordered_fusion_routes[1:]
            }
        )
        self.prefusion_task_adapters = nn.ModuleDict(
            {
                task: ResidualBottleneck(
                    hidden_dim,
                    prefusion_adapter_dim,
                    prefusion_adapter_dropout,
                )
                for task in resolved_prefusion_adapter_tasks
            }
            if self.use_prefusion_task_adapters
            else {}
        )
        direct_molecular_input_dim = hidden_dim + fp_dim
        self.molecular_fusions = nn.ModuleDict(
            {
                route: nn.Sequential(
                    nn.Linear(direct_molecular_input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for route in ordered_molecular_fusion_routes[1:]
            }
        )


        self._init_weights(initialization_seed)
        for fusion in self.condition_fusions.values():
            fusion.load_state_dict(self.feature_fusion.state_dict(), strict=True)
        if self.molecular_fusions:
            if self.molecular_encoder.fp_proj is None:
                raise AssertionError("Routed molecular fusion requires a direct projection")
            for fusion in self.molecular_fusions.values():
                fusion.load_state_dict(
                    self.molecular_encoder.fp_proj.state_dict(), strict=True
                )
        if self.graph_message_sharing != "shared":
            self.molecular_encoder.graph_encoder.configure_routed_tails(
                ordered_graph_message_routes,
                start_layer=self.graph_private_start_layer,
            )
        if self.grouped_sharing is not None:
            self.grouped_sharing.reset_identity()
        for adapter in self.task_adapters.values():
            adapter.reset_identity()
        for adapter in self.prefusion_task_adapters.values():
            adapter.reset_identity()
        self.molecular_encoder.configure_modalities(self.molecular_modalities)

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.load_state_dict(state_dict)

    def _init_weights(self, initialization_seed: int):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Embedding):
                    nn.init.xavier_normal_(m.weight)

    def _condition_fusion_for_task(self, task: str) -> FeatureFusionLayer:
        route = self.condition_fusion_routes[task]
        if route == self.primary_condition_fusion_route:
            return self.feature_fusion
        return self.condition_fusions[route]

    def _molecular_fusion_for_task(self, task: str) -> nn.Module:
        route = self.molecular_fusion_routes[task]
        if route == self.primary_molecular_fusion_route:
            if self.molecular_encoder.fp_proj is None:
                raise RuntimeError("Direct molecular fusion projection is unavailable")
            return self.molecular_encoder.fp_proj
        return self.molecular_fusions[route]

    def _graph_message_route_for_task(self, task: str) -> str | None:
        if self.graph_message_sharing == "shared":
            return None
        return self.graph_message_routes[task]

    def forward(self,
                graph: dgl.DGLGraph,
                duration_values: torch.Tensor,
                effect_onehots: torch.Tensor,
                media_onehots: torch.Tensor,
                smiles_list: List[str] = None,
                requested_tasks: Optional[List[str]] = None,
                return_gate_weights: bool = False,
                return_routing_diagnostics: bool = False,
                return_mmoe_diagnostics: bool = False,
                ) -> Dict[str, Dict[str, torch.Tensor]]:
        active_tasks = list(self.tasks if requested_tasks is None else requested_tasks)
        unknown = sorted(set(active_tasks) - set(self.tasks))
        if unknown:
            raise ValueError(f"Requested tasks are not configured in the model: {unknown}")
        if len(active_tasks) != len(set(active_tasks)):
            raise ValueError("requested_tasks contains duplicates")
        if return_gate_weights and not (
            self.sharing_mode == "global_mmoe"
            or (
                self.sharing_mode == "grouped"
                and self.group_sharing_type == "mmoe"
            )
        ):
            raise ValueError(
                "Gate weights require global_mmoe or grouped MMoE sharing"
            )
        if return_mmoe_diagnostics and self.sharing_mode != "global_mmoe":
            raise ValueError("MMoE diagnostics are only available in global_mmoe mode")

        if (
            self.molecular_fusion_sharing == "shared"
            and self.graph_message_sharing == "shared"
        ):
            shared_molecular_representation = self.molecular_encoder(graph)
            task_molecular_representations = {
                task: shared_molecular_representation for task in active_tasks
            }
        else:
            graph_modalities: Dict[
                str | None, Tuple[torch.Tensor, torch.Tensor]
            ] = {}
            task_molecular_representations = {}
            for task in active_tasks:
                graph_route = self._graph_message_route_for_task(task)
                if graph_route not in graph_modalities:
                    graph_modalities[graph_route] = (
                        self.molecular_encoder.encode_modalities(
                            graph, graph_route=graph_route
                        )
                    )
                graph_representation, fingerprints = graph_modalities[graph_route]
                task_molecular_representations[task] = (
                    self.molecular_encoder.fuse_modalities(
                        graph_representation,
                        fingerprints,
                        direct_projection=self._molecular_fusion_for_task(task),
                    )
                )
        task_inputs = {}
        for task in active_tasks:
            task_effect = (
                torch.zeros_like(effect_onehots)
                if task in self.constant_effect_tasks
                else effect_onehots
            )
            task_molecular_representation = task_molecular_representations[task]
            molecular_gradient_scale = self.molecular_gradient_scales[task]
            if molecular_gradient_scale != 1.0:
                task_molecular_representation = _GradientScale.apply(
                    task_molecular_representation, molecular_gradient_scale
                )
            if task in self.prefusion_adapter_tasks:
                task_molecular_representation = self.prefusion_task_adapters[task](
                    task_molecular_representation
                )
            task_inputs[task] = self._condition_fusion_for_task(task)(
                task_molecular_representation,
                duration_values,
                task_effect,
                media_onehots,
            )
        if self.sharing_mode == "global_mmoe":
            assert self.mmoe is not None
            if return_mmoe_diagnostics:
                representations, gate_weights, mmoe_diagnostics = self.mmoe(
                    task_inputs, return_diagnostics=True
                )
            else:
                representations, gate_weights = self.mmoe(task_inputs)
                mmoe_diagnostics = {}
            group_residual_ratios: Dict[str, torch.Tensor] = {}
        else:
            assert self.grouped_sharing is not None
            if return_gate_weights:
                grouped_outputs = self.grouped_sharing(
                    task_inputs, return_gate_weights=True
                )
                (
                    representations,
                    group_residual_ratios,
                    gate_weights,
                ) = grouped_outputs
            else:
                representations, group_residual_ratios = self.grouped_sharing(
                    task_inputs
                )
                gate_weights = {}
            mmoe_diagnostics = {}
        task_outputs = {}
        for task in active_tasks:
            routed = representations[task]
            if self.use_task_adapters:
                adapted = self.task_adapters[task](routed)
                adapter_residual_ratio = torch.linalg.vector_norm(
                    adapted - routed, dim=-1
                ) / torch.linalg.vector_norm(routed, dim=-1).clamp_min(1e-12)
            else:
                adapted = routed
                adapter_residual_ratio = None
            trunk = self.task_trunks[task](adapted)
            raw_log_sigma = (
                self.sigma_heads[task](trunk)
                if self.regression_mode == "heteroscedastic"
                else None
            )
            task_outputs[task] = _regression_output(
                self.mu_heads[task](trunk),
                raw_log_sigma,
                self.regression_mode,
                self.sigma_parameterization,
                self.min_log_sigma,
                self.max_log_sigma,
            )
            if return_gate_weights:
                task_outputs[task]["gate_weights"] = gate_weights[task]
            if return_mmoe_diagnostics:
                task_outputs[task].update(mmoe_diagnostics[task])
            if return_routing_diagnostics:
                if task in group_residual_ratios:
                    task_outputs[task]["group_residual_ratio"] = (
                        group_residual_ratios[task]
                    )
                if adapter_residual_ratio is not None:
                    task_outputs[task]["adapter_residual_ratio"] = (
                        adapter_residual_ratio
                    )
        return task_outputs


@gin.configurable()
class SingleTaskYieldGNN(nn.Module):

    def __init__(
        self,
        task: str,
        num_effects: int = 7,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_attention_heads: int = 4,
        edge_in_dim: int = 13,
        node_in_dim: int = 81,
        fp_dim: int = 1214,
        use_attention: bool = True,
        head_dim: int = 128,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        gnn_type: str = "our_gine",
        constant_effect_tasks: List[str] = None,
        regression_mode: Literal["heteroscedastic", "deterministic"] = "heteroscedastic",
        sigma_parameterization: Literal["clamp", "sigmoid"] = "clamp",
        min_log_sigma: float = -0.5,
        max_log_sigma: float = 2.0,
        fusion_mode: Literal["direct", "projected"] = "direct",
        modality_dropout: float = 0.0,
        jk_mode: Literal["last", "weighted_2_4"] = "last",
        pooling_mode: Literal["gated", "mean_max_gated"] = "gated",
        message_dropout: float = 0.0,
        train_eps: bool = False,
        checkpoint_path: Path | str | None = None,
        initialization_seed: int = 20260825,
    ):
        super().__init__()
        self.task = str(task)
        self.tasks = [self.task]
        self.model_kind = "single_task"
        self.regression_mode = _validate_regression_mode(regression_mode)
        if sigma_parameterization not in {"clamp", "sigmoid"}:
            raise ValueError("sigma_parameterization must be 'clamp' or 'sigmoid'")
        if not min_log_sigma < max_log_sigma:
            raise ValueError("min_log_sigma must be smaller than max_log_sigma")
        self.sigma_parameterization = sigma_parameterization
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)
        self.constant_effect_tasks = frozenset(
            constant_effect_tasks
            if constant_effect_tasks is not None
            else ["fish_EC50", "algae_EC50", "algae_EC10"]
        )

        self.model_spec = {
            "task": self.task,
            "num_effects": num_effects,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "edge_in_dim": edge_in_dim,
            "node_in_dim": node_in_dim,
            "fp_dim": fp_dim,
            "use_attention": use_attention,
            "head_dim": head_dim,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "gnn_type": gnn_type,
            "constant_effect_tasks": sorted(self.constant_effect_tasks),
            "regression_mode": self.regression_mode,
            "sigma_parameterization": self.sigma_parameterization,
            "min_log_sigma": self.min_log_sigma,
            "max_log_sigma": self.max_log_sigma,
            "fusion_mode": fusion_mode,
            "modality_dropout": modality_dropout,
            "jk_mode": jk_mode,
            "pooling_mode": pooling_mode,
            "message_dropout": message_dropout,
            "train_eps": bool(train_eps),
            "initialization_seed": int(initialization_seed),
        }

        self.feature_fusion = FeatureFusionLayer(
            hidden_dim=hidden_dim,
            num_effects=num_effects,
            num_media_types=3,
            dropout=dropout,
        )

        self.molecular_encoder = MolecularEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            edge_in_dim=edge_in_dim,
            node_in_dim=node_in_dim,
            fp_dim=fp_dim,
            use_attention=use_attention,
            attention_dropout=attention_dropout,
            gnn_type=gnn_type,
            dropout=dropout,
            fusion_mode=fusion_mode,
            modality_dropout=modality_dropout,
            jk_mode=jk_mode,
            pooling_mode=pooling_mode,
            message_dropout=message_dropout,
            train_eps=train_eps,
        )
        self.head_trunk = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Linear(head_dim, 1)
        self.sigma_head = (
            nn.Linear(head_dim, 1)
            if self.regression_mode == "heteroscedastic"
            else None
        )
        self._init_weights(initialization_seed)

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.load_state_dict(state_dict)

    def _init_weights(self, initialization_seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        graph: dgl.DGLGraph,
        duration_values: torch.Tensor,
        effect_onehots: torch.Tensor,
        media_onehots: torch.Tensor,
        smiles_list: List[str] = None,
        requested_tasks: Optional[List[str]] = None,
        return_gate_weights: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        active_tasks = [self.task] if requested_tasks is None else list(requested_tasks)
        if active_tasks != [self.task]:
            raise ValueError(
                f"Single-task model for {self.task!r} cannot evaluate {active_tasks!r}"
            )
        if return_gate_weights:
            raise ValueError("Single-task models do not have gate weights")

        molecule = self.molecular_encoder(graph)
        task_effect = (
            torch.zeros_like(effect_onehots)
            if self.task in self.constant_effect_tasks
            else effect_onehots
        )
        fused = self.feature_fusion(
            molecule,
            duration_values,
            task_effect,
            media_onehots,
        )
        trunk = self.head_trunk(fused)
        raw_log_sigma = self.sigma_head(trunk) if self.sigma_head is not None else None
        return {
            self.task: _regression_output(
                self.mu_head(trunk),
                raw_log_sigma,
                self.regression_mode,
                self.sigma_parameterization,
                self.min_log_sigma,
                self.max_log_sigma,
            )
        }
