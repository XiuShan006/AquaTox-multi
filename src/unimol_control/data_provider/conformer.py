
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import time
from typing import Any, Dict, Iterable, List, Optional

import torch

try:
    from rdkit import Chem
    from rdkit import rdBase
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except Exception:
    Chem = None
    rdBase = None
    AllChem = None
    _HAS_RDKIT = False


CONFORMER_PIPELINE_VERSION = "etkdgv3_mmff94_v3"

DEFAULT_EXCLUDED_SMILES = (
    "F[Si-2](F)(F)(F)(F)F.[Na+].[Na+]",
    "[NH4+].[NH4+].F[Si-2](F)(F)(F)(F)F",
    "[Zn+2].[F-][Si+4]([F-])([F-])([F-])([F-])[F-]",
    "[H+].[H+].F[Si-2](F)(F)(F)(F)F",
)

_METAL_ATOMIC_NUMBERS = frozenset(
    {
        3, 4, 11, 12, 13, 19, 20,
        *range(21, 32),
        37, 38,
        *range(39, 51),
        55, 56,
        *range(57, 85),
        87, 88,
        *range(89, 113),
    }
)


class ConformerGenerationError(RuntimeError):
    pass


class ConformerGenerationTimeout(ConformerGenerationError):
    pass


@dataclass(frozen=True)
class ConformerSettings:

    random_seed: int = 42
    mmff_variant: str = "MMFF94"
    mmff_max_iters: int = 1000
    use_uff_fallback: bool = True
    uff_max_iters: int = 1000
    molecule_timeout_seconds: float = 60.0
    fragment_policy: str = "keep_all"

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative for reproducible embedding")
        if self.mmff_variant != "MMFF94":
            raise ValueError("This release requires the MMFF94 force-field variant")
        if self.mmff_max_iters <= 0 or self.uff_max_iters <= 0:
            raise ValueError("force-field iteration limits must be positive")
        if self.molecule_timeout_seconds <= 0:
            raise ValueError("molecule_timeout_seconds must be positive")
        if self.fragment_policy != "keep_all":
            raise ValueError("Only fragment_policy='keep_all' is currently supported")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "pipeline_version": CONFORMER_PIPELINE_VERSION,
                "rdkit_version": rdBase.rdkitVersion if rdBase is not None else None,
                "embedding_method": (
                    "ETKDGv3 with staged random-coordinate, embedding-chirality, "
                    "and distance-geometry fallbacks (ETKDG for older RDKit)"
                ),
                "hydrogen_policy": "add for embedding/optimization; remove for Uni-Mol input",
            }
        )
        return payload

    @property
    def cache_tag(self) -> str:

        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
        return f"{CONFORMER_PIPELINE_VERSION}_{digest}"


@dataclass
class UniMolInput:
    molecule: Any
    atom_vec: torch.Tensor
    dist: torch.Tensor
    edge_type: torch.Tensor
    atom_indices: List[int]
    conformer_metadata: Dict[str, Any]


def _isolated_build_worker(
    connection,
    smiles: str,
    dictionary,
    settings: ConformerSettings,
    max_atoms: int,
    normalize_coords: bool,
) -> None:

    try:
        result = build_unimol_input(
            smiles,
            dictionary,
            settings=settings,
            max_atoms=max_atoms,
            normalize_coords=normalize_coords,
        )
        connection.send(
            (
                True,
                result.molecule.ToBinary(),
                result.atom_vec.numpy(),
                result.dist.numpy(),
                result.edge_type.numpy(),
                result.atom_indices,
                result.conformer_metadata,
            )
        )
    except Exception as exc:
        connection.send((False, type(exc).__name__, str(exc)))
    finally:
        connection.close()


def rdkit_available() -> bool:
    return _HAS_RDKIT


def _embedding_parameters(
    random_seed: int,
    use_random_coords: bool,
    *,
    enforce_chirality: bool = True,
    use_basic_knowledge: bool = True,
    ignore_smoothing_failures: bool = False,
):
    try:
        params = AllChem.ETKDGv3()
        method = "ETKDGv3"
    except AttributeError:
        params = AllChem.ETKDG()
        method = "ETKDG"
    params.randomSeed = int(random_seed)
    params.useRandomCoords = bool(use_random_coords)
    params.enforceChirality = bool(enforce_chirality)
    params.useBasicKnowledge = bool(use_basic_knowledge)
    params.ignoreSmoothingFailures = bool(ignore_smoothing_failures)
    return params, method


def _status_label(status: Optional[int]) -> str:
    if status is None:
        return "not_run"
    if status == 0:
        return "converged"
    if status == 1:
        return "max_iterations_reached"
    return f"failed_status_{status}"


def _force_field_energy(mol, method: str, mmff_variant: str) -> Optional[float]:
    try:
        if method == mmff_variant:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
            if props is None:
                return None
            force_field = AllChem.MMFFGetMoleculeForceField(mol, props)
        elif method == "UFF":
            force_field = AllChem.UFFGetMoleculeForceField(mol)
        else:
            return None
        return float(force_field.CalcEnergy()) if force_field is not None else None
    except Exception:
        return None


def generate_optimized_conformer(
    smiles: str,
    settings: Optional[ConformerSettings] = None,
):

    if not _HAS_RDKIT:
        raise ImportError("RDKit is required to generate Uni-Mol conformers")
    if not smiles:
        raise ConformerGenerationError("empty SMILES")

    settings = settings or ConformerSettings()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ConformerGenerationError("RDKit failed to parse SMILES")

    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    metadata: Dict[str, Any] = {
        "success": False,
        "pipeline_version": CONFORMER_PIPELINE_VERSION,
        "input_smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "fragment_policy": settings.fragment_policy,
        "num_fragments": len(Chem.GetMolFrags(mol)),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "contains_formal_charge": any(
            atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()
        ),
        "elements": elements,
        "contains_metal": any(
            atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS for atom in mol.GetAtoms()
        ),
        "num_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "random_seed": settings.random_seed,
        "mmff_variant": settings.mmff_variant,
        "mmff_max_iters": settings.mmff_max_iters,
        "uff_fallback_enabled": settings.use_uff_fallback,
        "uff_max_iters": settings.uff_max_iters,
        "molecule_timeout_seconds": settings.molecule_timeout_seconds,
    }

    try:
        mol_h = Chem.AddHs(mol)
        embedding_protocols = (
            {
                "fallback": "none",
                "use_random_coords": False,
                "enforce_chirality": True,
                "use_basic_knowledge": True,
                "ignore_smoothing_failures": False,
            },
            {
                "fallback": "random_coordinates",
                "use_random_coords": True,
                "enforce_chirality": True,
                "use_basic_knowledge": True,
                "ignore_smoothing_failures": False,
            },
            {
                "fallback": "relaxed_embedding_chirality",
                "use_random_coords": True,
                "enforce_chirality": False,
                "use_basic_knowledge": True,
                "ignore_smoothing_failures": False,
            },
            {
                "fallback": "relaxed_distance_geometry",
                "use_random_coords": True,
                "enforce_chirality": False,
                "use_basic_knowledge": False,
                "ignore_smoothing_failures": True,
            },
        )
        embed_status = -1
        embedding_method = "ETKDGv3"
        embedding_options = embedding_protocols[-1]
        embedding_attempts = 0
        for embedding_attempts, embedding_options in enumerate(
            embedding_protocols,
            start=1,
        ):
            mol_h.RemoveAllConformers()
            params, embedding_method = _embedding_parameters(
                settings.random_seed,
                embedding_options["use_random_coords"],
                enforce_chirality=embedding_options["enforce_chirality"],
                use_basic_knowledge=embedding_options["use_basic_knowledge"],
                ignore_smoothing_failures=embedding_options[
                    "ignore_smoothing_failures"
                ],
            )
            embed_status = int(AllChem.EmbedMolecule(mol_h, params))
            if embed_status == 0:
                break

        metadata.update(
            {
                "embedding_method": embedding_method,
                "embedding_status_code": embed_status,
                "embedding_attempts": embedding_attempts,
                "embedding_fallback": embedding_options["fallback"],
                "embedding_used_random_coords": embedding_options[
                    "use_random_coords"
                ],
                "embedding_enforce_chirality": embedding_options[
                    "enforce_chirality"
                ],
                "embedding_use_basic_knowledge": embedding_options[
                    "use_basic_knowledge"
                ],
                "embedding_ignore_smoothing_failures": embedding_options[
                    "ignore_smoothing_failures"
                ],
            }
        )
        if embed_status != 0 or mol_h.GetNumConformers() == 0:
            raise ConformerGenerationError(
                f"RDKit embedding failed with status {embed_status}"
            )

        mmff_has_params = bool(AllChem.MMFFHasAllMoleculeParams(mol_h))
        metadata["mmff_has_all_params"] = mmff_has_params
        optimization_method: Optional[str] = None
        optimization_status: Optional[int] = None
        fallback_reason: Optional[str] = None

        if mmff_has_params:
            try:
                optimization_status = int(
                    AllChem.MMFFOptimizeMolecule(
                        mol_h,
                        mmffVariant=settings.mmff_variant,
                        maxIters=settings.mmff_max_iters,
                    )
                )
                if optimization_status in (0, 1):
                    optimization_method = settings.mmff_variant
                else:
                    fallback_reason = f"mmff94_status_{optimization_status}"
            except Exception as exc:
                fallback_reason = f"mmff94_error:{type(exc).__name__}"
        else:
            fallback_reason = "missing_mmff94_parameters"

        if optimization_method is None and settings.use_uff_fallback:
            try:
                with rdBase.BlockLogs():
                    uff_has_params = bool(AllChem.UFFHasAllMoleculeParams(mol_h))
            except Exception:
                uff_has_params = False
            metadata["uff_has_all_params"] = uff_has_params
            if uff_has_params:
                try:
                    with rdBase.BlockLogs():
                        uff_status = int(
                            AllChem.UFFOptimizeMolecule(
                                mol_h,
                                maxIters=settings.uff_max_iters,
                            )
                        )
                    if uff_status in (0, 1):
                        optimization_method = "UFF"
                        optimization_status = uff_status
                    else:
                        suffix = f"uff_status_{uff_status}"
                        fallback_reason = f"{fallback_reason};{suffix}"
                except Exception as exc:
                    suffix = f"uff_error:{type(exc).__name__}"
                    fallback_reason = f"{fallback_reason};{suffix}"
        elif optimization_method is None:
            metadata["uff_has_all_params"] = None

        if optimization_method is None:
            optimization_method = "ETKDG-only"
            optimization_status = None
            if not settings.use_uff_fallback:
                fallback_reason = f"{fallback_reason};uff_fallback_disabled"
            elif metadata.get("uff_has_all_params") is False:
                fallback_reason = f"{fallback_reason};missing_uff_parameters"

        final_energy = _force_field_energy(
            mol_h, optimization_method, settings.mmff_variant
        )
        mol_3d = Chem.RemoveHs(mol_h)
        if mol_3d.GetNumAtoms() == 0 or mol_3d.GetNumConformers() == 0:
            raise ConformerGenerationError("empty molecule after hydrogen removal")

        conf = mol_3d.GetConformer()
        for atom_idx in range(mol_3d.GetNumAtoms()):
            pos = conf.GetAtomPosition(atom_idx)
            if not all(math.isfinite(value) for value in (pos.x, pos.y, pos.z)):
                raise ConformerGenerationError("non-finite coordinate generated")

        metadata.update(
            {
                "success": True,
                "optimization_method": optimization_method,
                "optimization_status_code": optimization_status,
                "optimization_status": _status_label(optimization_status),
                "optimization_converged": optimization_status == 0,
                "fallback_reason": fallback_reason,
                "final_force_field_energy": final_energy,
                "num_input_atoms": int(mol_3d.GetNumAtoms()),
            }
        )
        return mol_3d, metadata
    except ConformerGenerationError:
        raise
    except Exception as exc:
        raise ConformerGenerationError(
            f"conformer generation failed: {type(exc).__name__}: {exc}"
        ) from exc


def build_unimol_input(
    smiles: str,
    dictionary,
    *,
    settings: Optional[ConformerSettings] = None,
    max_atoms: int = 256,
    normalize_coords: bool = True,
) -> UniMolInput:

    if not _HAS_RDKIT:
        raise ImportError("RDKit is required to generate Uni-Mol conformers")
    precheck_mol = Chem.MolFromSmiles(smiles)
    if precheck_mol is None:
        raise ConformerGenerationError("RDKit failed to parse SMILES")
    input_atom_count = precheck_mol.GetNumAtoms()
    if max_atoms > 0 and input_atom_count > max_atoms:
        raise ConformerGenerationError(
            f"molecule has {input_atom_count} atoms, above max_atoms={max_atoms}"
        )

    mol, metadata = generate_optimized_conformer(smiles, settings=settings)
    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 0:
        raise ConformerGenerationError("molecule has no atoms for Uni-Mol input")
    if max_atoms > 0 and num_atoms > max_atoms:
        raise ConformerGenerationError(
            f"molecule has {num_atoms} atoms, above max_atoms={max_atoms}"
        )

    conf = mol.GetConformer()
    atom_symbols: List[str] = []
    coordinates: List[List[float]] = []
    atom_indices = list(range(num_atoms))
    for atom_idx in atom_indices:
        atom_symbols.append(mol.GetAtomWithIdx(atom_idx).GetSymbol())
        pos = conf.GetAtomPosition(atom_idx)
        coordinates.append([pos.x, pos.y, pos.z])

    coords = torch.tensor(coordinates, dtype=torch.float32)
    if normalize_coords:
        coords = coords - coords.mean(dim=0, keepdim=True)

    atom_vec = torch.from_numpy(dictionary.vec_index(atom_symbols)).long()
    atom_vec = torch.cat(
        [
            torch.tensor([dictionary.bos()], dtype=torch.long),
            atom_vec,
            torch.tensor([dictionary.eos()], dtype=torch.long),
        ],
        dim=0,
    )
    coords = torch.cat(
        [torch.zeros((1, 3)), coords, torch.zeros((1, 3))],
        dim=0,
    )
    diff = coords[:, None, :] - coords[None, :, :]
    dist = torch.sqrt(torch.clamp((diff ** 2).sum(dim=-1), min=0.0))
    vocab_size = len(dictionary)
    edge_type = atom_vec.view(-1, 1) * vocab_size + atom_vec.view(1, -1)

    return UniMolInput(
        molecule=mol,
        atom_vec=atom_vec,
        dist=dist,
        edge_type=edge_type,
        atom_indices=atom_indices,
        conformer_metadata=metadata,
    )


def build_unimol_input_isolated(
    smiles: str,
    dictionary,
    *,
    settings: Optional[ConformerSettings] = None,
    max_atoms: int = 256,
    normalize_coords: bool = True,
) -> UniMolInput:

    settings = settings or ConformerSettings()
    if "fork" not in multiprocessing.get_all_start_methods():
        raise RuntimeError(
            "isolated conformer timeouts require the multiprocessing 'fork' method"
        )

    context = multiprocessing.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_build_worker,
        args=(
            send,
            smiles,
            dictionary,
            settings,
            max_atoms,
            normalize_coords,
        ),
        daemon=True,
    )
    started = time.monotonic()
    process.start()
    send.close()
    payload = None
    timed_out = False
    deadline = started + settings.molecule_timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if receive.poll(min(0.1, remaining)):
                try:
                    payload = receive.recv()
                except EOFError:
                    payload = None
                break
            if not process.is_alive():
                if receive.poll():
                    try:
                        payload = receive.recv()
                    except EOFError:
                        payload = None
                break
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        receive.close()

    elapsed = time.monotonic() - started
    if timed_out:
        raise ConformerGenerationTimeout(
            f"conformer generation exceeded {settings.molecule_timeout_seconds:g} s "
            f"for SMILES {smiles!r}"
        )
    if payload is None:
        raise ConformerGenerationError(
            f"isolated conformer worker exited with code {process.exitcode} "
            f"without a result for SMILES {smiles!r}"
        )
    if not payload[0]:
        _, error_type, error_message = payload
        raise ConformerGenerationError(f"{error_type}: {error_message}")

    (
        _,
        molecule_binary,
        atom_vec,
        dist,
        edge_type,
        atom_indices,
        metadata,
    ) = payload
    metadata["conformer_wall_time_seconds"] = elapsed
    return UniMolInput(
        molecule=Chem.Mol(molecule_binary),
        atom_vec=torch.from_numpy(atom_vec),
        dist=torch.from_numpy(dist),
        edge_type=torch.from_numpy(edge_type),
        atom_indices=atom_indices,
        conformer_metadata=metadata,
    )


def summarize_conformer_metadata(
    records: Iterable[Dict[str, Any]],
    settings: ConformerSettings,
) -> Dict[str, Any]:

    rows = list(records)
    successful = [row for row in rows if row.get("success")]
    failed = [row for row in rows if not row.get("success")]
    method_counts = Counter(
        str(row.get("optimization_method", "failed")) for row in rows
    )
    status_counts = Counter(
        str(row.get("optimization_status", "failed")) for row in rows
    )
    fallback_counts = Counter(
        str(row["fallback_reason"])
        for row in successful
        if row.get("fallback_reason")
    )
    embedding_fallback_counts = Counter(
        str(row.get("embedding_fallback", "failed")) for row in rows
    )

    return {
        "settings": settings.to_dict(),
        "cache_tag": settings.cache_tag,
        "total_molecules": len(rows),
        "successful_molecules": len(successful),
        "failed_molecules": len(failed),
        "failure_rate": (len(failed) / len(rows)) if rows else 0.0,
        "mmff94_parameterized_molecules": sum(
            bool(row.get("mmff_has_all_params")) for row in successful
        ),
        "mmff94_parameter_coverage": (
            sum(bool(row.get("mmff_has_all_params")) for row in successful)
            / len(successful)
            if successful
            else 0.0
        ),
        "optimization_method_counts": dict(sorted(method_counts.items())),
        "optimization_status_counts": dict(sorted(status_counts.items())),
        "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        "embedding_fallback_counts": dict(sorted(embedding_fallback_counts.items())),
        "multi_fragment_molecules": sum(
            int(row.get("num_fragments", 0)) > 1 for row in successful
        ),
        "charged_molecules": sum(
            int(row.get("formal_charge", 0)) != 0 for row in successful
        ),
        "ion_containing_molecules": sum(
            bool(row.get("contains_formal_charge")) for row in successful
        ),
        "metal_containing_molecules": sum(
            bool(row.get("contains_metal")) for row in successful
        ),
    }
