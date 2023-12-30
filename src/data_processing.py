from rdkit import Chem
from torch_geometric.data import Data
import torch
from src.features import AtomFeatures, BondFeatures
import numpy as np
import random
import pandas as pd
import json

def read_data(csv_file, smiles_col='smiles', target_col='solubility'):
    """
    Reads data from a CSV file and extracts the specified columns.

    Parameters:
    csv_file (str): Path to the CSV file.
    smiles_col (str): The name of the column containing SMILES strings. Default is 'smiles'.
    target_col (str): The name of the column containing target properties. Default is 'solubility'.

    Returns:
    tuple: A tuple containing two lists - one for SMILES strings and one for target properties.
    """

    try:
        df = pd.read_csv(csv_file)

        # Check if specified columns exist in the dataframe
        if smiles_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Columns '{smiles_col}' and/or '{target_col}' not found in the CSV file.")

        smiles_list = df[smiles_col].tolist()
        target_properties = df[target_col].values

        return smiles_list, target_properties

    except Exception as e:
        print(f"Error reading data: {e}")
        return None, None


def normalize_targets(targets, params_file='norm_params.json'):
    targets = np.array(targets, dtype=float)
    mean = np.mean(targets)
    std = np.std(targets)
    normalized_targets = (targets - mean) / std

    # Save mean and std in a JSON file
    with open(params_file, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)

    return normalized_targets.tolist(), mean, std

def denormalize_targets(norm_targets, mean, std):
    return [target * std + mean for target in norm_targets]


def smiles_to_graph(smiles, atom_features, bond_features):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f'Invalid molecule for SMILES: {smiles}')
        return None

    N = mol.GetNumAtoms()

    # Atom features
    x = [atom_features.calc_feature_one_hot_vec(atom) for atom in mol.GetAtoms()]
    x = torch.stack(x)

    # Bond features
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])
        bf = bond_features.calc_feature_one_hot_vec(bond)
        edge_attrs.extend([bf, bf])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_features.get_total_dim()))

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def prepare_data(smiles_list, target_properties):
    # Convert SMILES to graph data objects and add target properties
    graph_data_list = [smiles_to_graph(smiles, atom_features=AtomFeatures(), bond_features=BondFeatures()) for smiles in smiles_list]
    for data, target in zip(graph_data_list, target_properties):
        data.y = torch.tensor([target], dtype=torch.float)
    return graph_data_list

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    random.shuffle(dataset)
    return dataset[:train_size], dataset[train_size:train_size+val_size], dataset[-test_size:]