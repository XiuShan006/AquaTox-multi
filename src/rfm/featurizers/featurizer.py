from typing import Optional
import dgl
from dgl import DGLGraph
from dgllife.utils import CanonicalBondFeaturizer, WeaveAtomFeaturizer, mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit import DataStructs
import torch
import numpy as np

from .utils import ATOM_TYPES

_RDKIT_DESCRIPTORS = [
    ("MolLogP",           5.0),
    ("MolWt",           500.0),
    ("HeavyAtomCount",   30.0),
    ("TPSA",            100.0),
    ("NumHDonors",        5.0),
    ("NumHAcceptors",    10.0),
    ("NumRotatableBonds", 10.0),
    ("NumAromaticRings",  4.0),
    ("NumSaturatedRings", 4.0),
    ("RingCount",         5.0),
    ("FractionCSP3",      1.0),
    ("BertzCT",        1000.0),
    ("HallKierAlpha",     5.0),
    ("Kappa1",           20.0),
    ("Kappa2",           10.0),
    ("Kappa3",            5.0),
    ("Chi0v",            10.0),
    ("Chi1v",            10.0),
    ("MaxPartialCharge",  1.0),
    ("MinPartialCharge",  1.0),
    ("NumAmideBonds",     3.0),
    ("NumBridgeheadAtoms",3.0),
    ("NumSpiroAtoms",     2.0),
]

_DESC_FUNCS = {
    "MolLogP":            lambda m: Descriptors.MolLogP(m),
    "MolWt":              lambda m: Descriptors.MolWt(m),
    "HeavyAtomCount":     lambda m: Descriptors.HeavyAtomCount(m),
    "TPSA":               lambda m: Descriptors.TPSA(m),
    "NumHDonors":         lambda m: Descriptors.NumHDonors(m),
    "NumHAcceptors":      lambda m: Descriptors.NumHAcceptors(m),
    "NumRotatableBonds":  lambda m: Descriptors.NumRotatableBonds(m),
    "NumAromaticRings":   lambda m: Descriptors.NumAromaticRings(m),
    "NumSaturatedRings":  lambda m: Descriptors.NumSaturatedRings(m),
    "RingCount":          lambda m: Descriptors.RingCount(m),
    "FractionCSP3":       lambda m: Descriptors.FractionCSP3(m),
    "BertzCT":            lambda m: Descriptors.BertzCT(m),
    "HallKierAlpha":      lambda m: Descriptors.HallKierAlpha(m),
    "Kappa1":             lambda m: Descriptors.Kappa1(m),
    "Kappa2":             lambda m: Descriptors.Kappa2(m),
    "Kappa3":             lambda m: Descriptors.Kappa3(m),
    "Chi0v":              lambda m: Descriptors.Chi0v(m),
    "Chi1v":              lambda m: Descriptors.Chi1v(m),
    "MaxPartialCharge":   lambda m: Descriptors.MaxPartialCharge(m),
    "MinPartialCharge":   lambda m: Descriptors.MinPartialCharge(m),
    "NumAmideBonds":      lambda m: rdMolDescriptors.CalcNumAmideBonds(m),
    "NumBridgeheadAtoms": lambda m: rdMolDescriptors.CalcNumBridgeheadAtoms(m),
    "NumSpiroAtoms":      lambda m: rdMolDescriptors.CalcNumSpiroAtoms(m),
}

RDKIT_DIM = len(_RDKIT_DESCRIPTORS)
FP_TOTAL_DIM = 1024 + 167 + RDKIT_DIM


def compute_rdkit_descriptors(mol) -> np.ndarray:
    vals = []
    for name, scale in _RDKIT_DESCRIPTORS:
        try:
            v = float(_DESC_FUNCS[name](mol))
            v = v / scale
            v = float(np.clip(v, -5.0, 5.0))
        except Exception:
            v = 0.0
        if not np.isfinite(v):
            v = 0.0
        vals.append(v)
    return np.array(vals, dtype=np.float32)

class ReactionFeaturizer:
    def __init__(self):
        self.edge_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.node_featurizer = WeaveAtomFeaturizer(atom_types=ATOM_TYPES)

    def featurize_smiles_single(self, smiles: str) -> DGLGraph:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            Chem.RemoveHs(mol)

            graph = mol_to_bigraph(
                mol=mol,
                add_self_loop=True,
                canonical_atom_order=False,
                node_featurizer=self.node_featurizer,
                edge_featurizer=self.edge_featurizer,
            )

            AllChem.ComputeGasteigerCharges(mol)
            gasteiger_charges = []
            for atom in mol.GetAtoms():
                c = atom.GetPropsAsDict().get('_GasteigerCharge', 0.0)
                if not np.isfinite(c):
                    c = 0.0
                gasteiger_charges.append(float(np.clip(c / 0.5, -1.0, 1.0)))
            charges_tensor = torch.tensor(gasteiger_charges, dtype=torch.float).unsqueeze(1)
            graph.ndata['h'] = torch.cat([graph.ndata['h'], charges_tensor], dim=1)

            morgan_gen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_morgan = np.zeros((1024,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(morgan_gen, fp_morgan)

            fp_MACCS = MACCSkeys.GenMACCSKeys(mol)
            fp_MACCS_array = np.zeros((167,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp_MACCS, fp_MACCS_array)

            rdkit_desc = compute_rdkit_descriptors(mol)

            fps = np.concatenate([fp_morgan, fp_MACCS_array, rdkit_desc], axis=0)
            fps_features = torch.tensor(fps, dtype=torch.float)

            num_nodes = graph.ndata['h'].shape[0]
            fps_broadcast = fps_features.unsqueeze(0).repeat(num_nodes, 1)
            graph.ndata['fp'] = fps_broadcast

            return graph
        except Exception as exc:
            raise RuntimeError(
                f"Failed to featurize frozen structure {smiles!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
