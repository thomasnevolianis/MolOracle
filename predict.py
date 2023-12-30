import torch
from torch_geometric.loader import DataLoader
from src.model import GNNModel
from src.data_processing import smiles_to_graph, AtomFeatures, BondFeatures
import matplotlib.pyplot as plt
import pandas as pd
from src.data_processing import denormalize_targets
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def plot_predictions(actuals, predictions, save_path=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red')  # Diagonal line

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


def save_statistics(stats, file_path):
    with open(file_path, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

def load_model(model_path, node_feature_size, edge_feature_size, dropout_rate, device):
    model = GNNModel(node_feature_size, edge_feature_size, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_dataset(model, loader, device):
    model.eval()
    normalized_predictions = []
    actuals = []
    with torch.no_grad():
        for data in loader:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.y = data.y.to(device)
            if hasattr(data, 'batch'):
                data.batch = data.batch.to(device)

            normalized_output = model(data)
            normalized_predictions.extend(normalized_output.view(-1).cpu().numpy())
            actuals.extend(data.y.view(-1).cpu().numpy())

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, normalized_predictions))
    mae = mean_absolute_error(actuals, normalized_predictions)

    return actuals, normalized_predictions,  rmse, mae


def load_dataset(csv_file, smiles_col, target_col, batch_size):
    """
    Loads a dataset from a CSV file and converts it into a DataLoader.

    Parameters:
    csv_file (str): Path to the CSV file.
    smiles_col (str): Name of the column containing SMILES strings.
    target_col (str): Name of the column containing target properties.
    batch_size (int): Batch size for the DataLoader.

    Returns:
    DataLoader: A DataLoader containing the graph data.
    """
    df = pd.read_csv(csv_file)
    smiles_list = df[smiles_col].tolist()
    target_properties = df[target_col].values

    atom_features = AtomFeatures()
    bond_features = BondFeatures()

    graph_data_list = [smiles_to_graph(smiles, atom_features, bond_features) for smiles in smiles_list]
    for data, target in zip(graph_data_list, target_properties):
        data.y = torch.tensor([target], dtype=torch.float)

    return DataLoader(graph_data_list, batch_size=batch_size, shuffle=False)


def load_normalization_params(file_path='norm_params.json'):
    """
    Loads normalization parameters from a JSON file.

    Parameters:
    file_path (str): Path to the JSON file containing mean and standard deviation.

    Returns:
    tuple: A tuple containing mean and standard deviation.
    """
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        mean = params.get('mean', 0)
        std = params.get('std', 1)

        return mean, std

    except FileNotFoundError:
        print(f"Normalization parameters file not found: {file_path}")
        return 0, 1
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return 0, 1
    except Exception as e:
        print(f"An error occurred while loading normalization parameters: {e}")
        return 0, 1

def main(config):
    device = torch.device(config['device'])
    model = load_model(config['model_path'], config['node_feature_size'], config['edge_feature_size'], config['dropout_rate'], device)
    dataset_loader = load_dataset(config['csv_file_path'], config['smiles_col'], config['target_col'], config['batch_size'])
    actuals, normalized_predictions, rmse, mae = predict_dataset(model, dataset_loader, device)
    mean, std = load_normalization_params(config['norm_params_file'])
    predictions = denormalize_targets(normalized_predictions, mean, std)
    plot_predictions(actuals, predictions, config.get('plot_save_path'))
    # Save statistics if path is provided
    if config.get('stats_save_path'):
        stats = {'RMSE': rmse, 'MAE': mae}
        save_statistics(stats, config['stats_save_path'])

if __name__ == "__main__":
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_path': 'example/model.pt',
        'csv_file_path': 'example/esol.csv',
        'node_feature_size': 159,
        'edge_feature_size': 14,
        'dropout_rate': 0.5,
        'smiles_col': 'smiles',
        'target_col': 'solubility',
        'batch_size': 32,
        'norm_params_file': 'example/norm_params.json',
        'plot_save_path': 'example/predictions_plot.png',  # Optional: 'path/to/save/predictions_plot.png' or None
        'stats_save_path': 'example/statistics.txt'  # Optional: 'path/to/save/statistics.txt' or None
    }
    main(config)