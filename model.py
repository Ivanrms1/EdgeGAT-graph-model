# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    """
    Modelo de Graph Attention Network (GAT) para segmentación/clasificación de nubes de puntos,
    asumiendo que en data.x vienen las características geométricas (x, y, z, nx, ny, nz, curvatura, etc.)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        """
        Args:
            in_channels (int): Dimensión de entrada (ej. 6 o 7 si incluyes normales, curvatura, etc.).
            hidden_channels (int): Número de canales/ neuronas en capas ocultas.
            out_channels (int): Número de clases a predecir.
            heads (int): Número de cabezas de atención (multi-head).
        """
        super(GATModel, self).__init__()
        
        # Capa 1 GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # OJO: si concat=True, la salida será hidden_channels * heads en la primera capa
        
        # Capa 2 GAT (reducimos de hidden_channels * heads a hidden_channels)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)
        
        # Capa final para clasificación (segmentación de puntos).
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): [num_nodes, in_channels] 
                              (features de cada punto)
            edge_index (torch.LongTensor): [2, num_edges] 
                              (grafo que define las relaciones entre nodos)
        
        Returns:
            torch.Tensor: logits de forma [num_nodes, out_channels]
        """
        # Paso por la primera GAT
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # activación
        
        # Paso por la segunda GAT
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        
        # Capa fully-connected para obtener logits
        out = self.classifier(x)
        return out

def build_model(in_channels, hidden_channels, out_channels, heads=4):
    """
    Construye y retorna una instancia de GATModel
    """
    return GATModel(in_channels, hidden_channels, out_channels, heads=heads)
