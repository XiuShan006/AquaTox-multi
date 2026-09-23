from typing import Optional
import dgl
from dgl import DGLGraph
from dgllife.utils import CanonicalBondFeaturizer, WeaveAtomFeaturizer, mol_to_bigraph
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
import torch
import numpy as np

from .utils import ATOM_TYPES

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

            morgan_gen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_morgan = np.zeros((1,), dtype=int)
            DataStructs.ConvertToNumpyArray(morgan_gen, fp_morgan)

            fp_MACCS = MACCSkeys.GenMACCSKeys(mol)
            fp_MACCS_array = np.zeros((1,), dtype=int)
            DataStructs.ConvertToNumpyArray(fp_MACCS, fp_MACCS_array)

            fps = np.concatenate((fp_morgan, fp_MACCS_array), axis=0)
            fps_features = torch.tensor(fps, dtype=torch.float)

            node_feat = graph.ndata['h']
            num_nodes = node_feat.shape[0]

            fps_broadcast = fps_features.unsqueeze(0).repeat(num_nodes, 1)

            node_feat = torch.cat([node_feat, fps_broadcast], dim=1)
            graph.ndata['h'] = node_feat

            return graph
        except Exception as e:
            print(f"Error featurizing SMILES '{smiles}': {e}")
            g = dgl.graph(([0], [0]), idtype=torch.int32)
            g.ndata['h'] = torch.zeros(1, 1271)
            g.edata['e'] = torch.zeros(1, 13)
            return g
