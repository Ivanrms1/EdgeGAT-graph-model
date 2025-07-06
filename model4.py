# model4.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SimpleGCN(nn.Module):
    """
    GCN simple con n capas, orientado a clasificación de nodos.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        """
        Args:
            in_channels: Número de features de entrada.
            hidden_channels: Número de canales ocultos en cada capa.
            out_channels: Número de clases (node-level).
            num_layers: Cuántas capas GCNConv+ReLU aplicar.
            dropout: tasa de dropout.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout

        # Primera capa
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Capas intermedias
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Última capa
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            # Si num_layers=1, la primera capa va directo a out_channels
            self.convs[0] = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        batch: [num_nodes] 
        Retorna [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                # Activación y dropout en capas intermedias
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # La última capa no forzosamente lleva ReLU 
        return x
