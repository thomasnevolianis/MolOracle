import optuna
import torch
from torch_geometric.loader import DataLoader
from data_processing import prepare_data, split_dataset, read_data, normalize_targets
from model import GNNModel
from train import train_model, evaluate_model, EarlyStopping

def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    patience = trial.suggest_int("patience", 5, 20)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    # Read and prepare data
    csv_file_path = 'sol_dataset.csv'
    smiles_list, target_properties = read_data(csv_file_path)
    normalized_targets, mean, std = normalize_targets(target_properties)
    graph_data_list = prepare_data(smiles_list, normalized_targets)
    train_dataset, val_dataset, _ = split_dataset(graph_data_list, 0.8, 0.1, 0.1)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    node_feature_size = 159
    edge_feature_size = 14
    model = GNNModel(node_feature_size, edge_feature_size, dropout_rate=dropout_rate)
    model.to(device)

    # Training with early stopping
    train_model(model, train_loader, val_loader, device, epochs=300, patience=patience, lr=lr)

    # Evaluate the model
    val_loss = evaluate_model(model, val_loader, device)

    return val_loss

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    print("Best hyperparameters:", study.best_trial.params)

if __name__ == "__main__":
    main()
