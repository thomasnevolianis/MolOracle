import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, MessagePassing


class GNNLayer(MessagePassing):
    def __init__(self, node_in_channels, node_out_channels, edge_in_channels, edge_out_channels):
        super(GNNLayer, self).__init__(aggr='add')  # 'add' for message aggregation
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        # Initial transformations
        self.node_transform = Linear(node_in_channels, node_out_channels)
        self.edge_transform = Linear(edge_in_channels, edge_out_channels)

        # Edge MLP
        self.edge_mlp = Sequential(
            Linear(edge_out_channels, node_out_channels),
            ReLU(),
            Linear(node_out_channels, node_out_channels)
        )

        # Update MLP
        self.update_mlp = Sequential(
            Linear(node_out_channels * 2, node_out_channels),
            ReLU(),
            Linear(node_out_channels, node_out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # Transform node and edge features
        x = self.node_transform(x)
        edge_attr = self.edge_transform(edge_attr)

        # Start message passing
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Message function: Combines node and edge features
        transformed_edge_attr = self.edge_mlp(edge_attr)
        return x_j + transformed_edge_attr

    def update(self, aggr_out, x):
        # Update function: Updates node features based on aggregated messages
        return self.update_mlp(torch.cat([x, aggr_out], dim=1))

    def __repr__(self):
        return (f'GNNLayer(node_in_channels={self.node_in_channels}, '
                f'node_out_channels={self.node_out_channels}, '
                f'edge_in_channels={self.edge_in_channels}, '
                f'edge_out_channels={self.edge_out_channels})')

class GNNModel(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, dropout_rate=0.5):
        super(GNNModel, self).__init__()
        self.conv1 = GNNLayer(node_feature_size, 16, edge_feature_size, 16)
        self.conv2 = GNNLayer(16, 16, edge_feature_size, 16)
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Apply the GNN layers with dropout
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.dropout(x)

        x = global_mean_pool(x, data.batch)  # Aggregate node features
        return self.fc(x)

    def __repr__(self):
        return (f'GNNModel(\n'
                f'  (conv1): {self.conv1}\n'
                f'  (conv2): {self.conv2}\n'
                f'  (dropout): {self.dropout}\n'
                f'  (fc): {self.fc}\n'
                f')')