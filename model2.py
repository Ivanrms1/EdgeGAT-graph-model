import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool

class GCN_SAGPool(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, ratio=0.5):
        """
        Args:
            in_channels  Número de features de entrada por nodo
            hidden_channels : Dimensión de las capas intermedias del GCN.
            out_channels : Número de clases a predecir en la clasificación global del grafo.
            ratio (float): Proporción de nodos que conservamos tras cada pooling
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, ratio=ratio)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, ratio=ratio)
        
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Primera capa GCN
        x = F.relu(self.conv1(x, edge_index))
        # Primer pooling SAG
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, batch=batch)

        # Segunda capa GCN
        x = F.relu(self.conv2(x, edge_index))
        # Segundo pooling SAG
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, batch=batch)

        # Pooling global para agrupar todo el grafo 
        x = global_mean_pool(x, batch)

        # Clasificación 
        x = self.lin(x)
        return x
