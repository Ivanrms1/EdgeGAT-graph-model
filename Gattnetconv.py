import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATConv, knn_graph

"""
Originaly this file was created to do GAT + Tnet, then i used to perform other experiments here.
"""
def build_mlp(in_dim, hidden_dim, out_dim, dropout=0.2):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        
        nn.Linear(hidden_dim, out_dim)
    )

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(ResidualMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        return out + residual

    
class GATTNetConvHybrid(nn.Module):
    """
    Modelo que combina:
      - 2 capas de EdgeConv:
          
          
      - Concatenación de las salidas de ambas capas de EdgeConv.
      - 2 capas de GAT para refinar la información.
      - Clasificador final.
    
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, k=16, dynamic_graph=False, dropout=0.2):
        super(GATTNetConvHybrid, self).__init__()
        self.k = k
        self.dynamic_graph = dynamic_graph

        # Como EdgeConv trabaja con la concatenación (x_i, x_j), la dimensión de entrada es in_channels * 2.
        self.edgeconv1 = EdgeConv(
            nn=ResidualMLP(in_dim=in_channels * 2, hidden_dim=hidden_channels, out_dim=hidden_channels, dropout=dropout),
            aggr='sum'
        )
        # Segunda capa EdgeConv con residualmlp
    
        self.edgeconv2 = EdgeConv(
            nn=ResidualMLP(in_dim=hidden_channels * 2, hidden_dim=hidden_channels, out_dim=hidden_channels, dropout=dropout),
            aggr='mean'
        )

        self.gat1 = GATConv(2 * hidden_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        # Capa de clasificación
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        # Si se usa grafo dinámico, se recalcula el knn en cada forward.
        if self.dynamic_graph:
    
            x_tensor = x if isinstance(x, torch.Tensor) else x.x
            edge_index = knn_graph(x_tensor, k=self.k, batch=batch, loop=False)

        # Primera capa EdgeConv
        x1 = self.edgeconv1(x, edge_index)
        x1 = F.leaky_relu(x1)

        # Segunda capa EdgeConv 
        x2 = self.edgeconv2(x1, edge_index)
        x2 = F.leaky_relu(x2)

        # Concatenación de las salidas de ambas EdgeConv.
        x_concat = torch.cat([x1, x2], dim=1)  # [num_nodes, 2 * hidden_channels]

        # Capas GAT para refinar la representación.
        x_gat = self.gat1(x_concat, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = self.gat2(x_gat, edge_index)
        x_gat = F.elu(x_gat)

        # Clasificación final.
        out = self.classifier(x_gat)
        return out

