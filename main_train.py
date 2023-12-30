import torch
from src.data_processing import prepare_data, split_dataset, read_data, normalize_targets
from torch_geometric.loader import DataLoader
from src.model import GNNModel
from src.train import train_model, evaluate_model

# Configuration Dictionary
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'csv_file_path': 'example/esol.csv',
    'smiles_col': 'smiles',
    'target_col': 'solubility',
    'batch_size': 32,
    'node_feature_size': 159,
    'edge_feature_size': 14,
    'dropout_rate': 0.3,
    'train_epochs': 100,
    'patience': 12,
    'learning_rate': 0.001,
    'model_save_path': 'example/model.pt',
    'params_file': 'example/norm_params.json'
}

# Print device information
print(f"Using {config['device'].upper()} for training.")

# Read SMILES strings and target properties from CSV file
smiles_list, target_properties = read_data(config['csv_file_path'], smiles_col=config['smiles_col'], target_col=config['target_col'])

# Normalize target properties
normalized_targets, mean, std = normalize_targets(target_properties, params_file=config['params_file'])

# Prepare data
graph_data_list = prepare_data(smiles_list, normalized_targets)

# Split dataset
train_dataset, val_dataset, test_dataset = split_dataset(graph_data_list, 0.8, 0.1, 0.1)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Initialize model
model = GNNModel(config['node_feature_size'], config['edge_feature_size'], dropout_rate=config['dropout_rate'])

# Train the model with early stopping and CUDA support
device = torch.device(config['device'])
train_model(model, train_loader, val_loader, device, epochs=config['train_epochs'], patience=config['patience'], lr=config['learning_rate'])

# Evaluate on the test dataset
evaluate_model(model, test_loader, device)

# Save the model
torch.save(model.state_dict(), config['model_save_path'])
print(f"Model saved as {config['model_save_path']}")
