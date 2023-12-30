import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def calculate_metrics(output, target):
    mae = torch.mean(torch.abs(output - target))
    rmse = torch.sqrt(torch.mean((output - target) ** 2))
    return mae.item(), rmse.item()

def train_model(model, train_loader, val_loader, device, epochs=100, patience=10, lr=0.01):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        model.train()
        train_mae, train_rmse, train_loss = 0, 0, 0

        for data in train_loader:
            # Move all components of data to the device
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.y = data.y.to(device)
            if hasattr(data, 'batch'):
                data.batch = data.batch.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            mae, rmse = calculate_metrics(output, data.y.view(-1, 1))
            train_mae += mae
            train_rmse += rmse

        train_mae /= len(train_loader)
        train_rmse /= len(train_loader)
        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_mae, val_rmse = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                # Move all components of data to the device
                data.x = data.x.to(device)
                data.edge_index = data.edge_index.to(device)
                data.edge_attr = data.edge_attr.to(device)
                data.y = data.y.to(device)
                if hasattr(data, 'batch'):
                    data.batch = data.batch.to(device)
                
                output = model(data)
                val_loss += criterion(output, data.y.view(-1, 1)).item()
                mae, rmse = calculate_metrics(output, data.y.view(-1, 1))
                val_mae += mae
                val_rmse += rmse

        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        val_rmse /= len(val_loader)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train RMSE: {train_rmse:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}')

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss, test_mae, test_rmse = 0, 0, 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for data in test_loader:
            data.x, data.edge_index, data.edge_attr, data.y = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.y.to(device)
            if hasattr(data, 'batch'):
                data.batch = data.batch.to(device)

            output = model(data)
            loss = criterion(output, data.y.view(-1, 1))
            test_loss += loss.item()
            mae, rmse = calculate_metrics(output, data.y.view(-1, 1))
            test_mae += mae
            test_rmse += rmse

    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    test_rmse /= len(test_loader)

    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}')
    return test_loss #, test_mae, test_rmse
