import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class SensorSelectorGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super(SensorSelectorGNN, self).__init__()
        
        # 1. Marginal Encoder: Processes the (T+2) time distribution
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # 2. Graph Processing
        # self.conv1 = GCNConv(32, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.conv1 = GATConv(32, hidden_channels, heads=4, concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        
        # 3. Score Predictor
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        # x shape: [N, T+2]
        x = self.encoder(x) 
        
        # GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Final scalar score per node
        out = self.head(x) 
        return out.squeeze(-1) # Shape: [N]



class SensorSelectorMLP(nn.Module):
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x, edge_index=None):  # edge_index ignored
        return self.net(x).squeeze(-1)