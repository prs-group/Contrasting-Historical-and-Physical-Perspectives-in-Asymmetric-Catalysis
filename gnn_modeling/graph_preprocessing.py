import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from rdkit import Chem
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from rdkit import Chem
import numpy as np

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                  xyz = None,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=True,
                  use_partial_charge=True,
                  ):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    from rdkit import Chem
    results = one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
    if use_partial_charge:
      try:
        #print(atom.GetProp('_GasteigerCharge'))
        #print(type(atom.GetProp('_GasteigerCharge')))
        results = results + [float(atom.GetProp('_GasteigerCharge'))]
      except:
        print('Failed to compute GasteigerCharge')

    return np.array(results)

def bond_features(bond, use_chirality=True):
  from rdkit import Chem
  bt = bond.GetBondType()
  bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing(),
  ]
  if use_chirality:
    bond_feats = bond_feats + one_of_k_encoding_unk(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
  bond_feats = bond_feats + [0]
  return np.array(bond_feats)

def get_bond_pairs(molecules):
    res = [[], []]
    highest_index_0 = 0
    highest_index_1 = 0
    for mol in molecules:
        bonds = mol.GetBonds()
        for bond in bonds:
          res[0] += [highest_index_0  + bond.GetBeginAtomIdx(),  highest_index_0  + bond.GetEndAtomIdx()]
          res[1] += [highest_index_1 + bond.GetEndAtomIdx(),  highest_index_1 + bond.GetBeginAtomIdx()]

        highest_index_0 = max(res[0]) + 1
        highest_index_1 = max(res[1]) + 1

    return res

def multi_graph(smiles, ee, temp):
    
    try:
        from rdkit.Chem import AllChem
    except ModuleNotFoundError:
        raise ImportError("This class requires RDKit to be installed.")
    
    all_nodes = []
    all_edge_attr = []
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    for mol in molecules:
        AllChem.ComputeGasteigerCharges(mol)
        atoms = mol.GetAtoms()

        # Compute partial charges
        node_f = [atom_features(atom) for atom in atoms]
        all_nodes.extend(node_f)

        bonds = mol.GetBonds()

        edge_attr = [bond_features(bond, use_chirality=True) for bond in bonds]
        for bond in bonds:
            edge_attr.append(bond_features(bond))

        all_edge_attr.extend(edge_attr)

    all_edge_index = get_bond_pairs(molecules)
    neg_edge_index = negative_sampling(torch.tensor(all_edge_index, dtype=torch.long)).t()
    all_nodes = np.array(all_nodes)
    all_edge_attr = np.array(all_edge_attr)
    
    data = Data(x=torch.tensor(all_nodes, dtype=torch.float),
                edge_index=torch.tensor(all_edge_index, dtype=torch.long),
                neg_edges=neg_edge_index,
                edge_attr=torch.tensor(all_edge_attr, dtype=torch.float),
                y=torch.FloatTensor([ee]).view(1,1),
                temp=torch.FloatTensor([temp]).view(1,1)
                )
    return data

def my_disc_graph_only_dataloader(
                              dataframe,
                              smiles_list, 
                              bs = 9,
                              target_column = 'ee'
                              ):
    graph_list = [multi_graph(list(row[smiles_list]), row[target_column], row['T']) for _, row in dataframe.iterrows()]
    return DataLoader(graph_list, batch_size=bs, shuffle=False), graph_list

