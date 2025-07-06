# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    """
    Graph Attention Network model for segmentation/clasification tasks
    We also used xyz Point cloud features + other PCA extracted features
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        """
        Args:
            in_channels (int): Input dimention
            hidden_channels (int): hidden channels 
            out_channels (int): Output classes
            heads (int): head attention
        """
        super(GATModel, self).__init__()
        
        # Capa 1 GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # si concat=True, la salida será hidden_channels * heads en la primera capa
        
        # Capa 2 GAT (reducimos de hidden_channels * heads a hidden_channels)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)
        
        # Capa final para clasificación.
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): [num_nodes, in_channels] 
            edge_index (torch.LongTensor): [2, num_edges] 
        Returns:
            torch.Tensor: logits [num_nodes, out_channels]
        """
        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # activación elu
        
        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        
        # Capa fully-connected for logits
        out = self.classifier(x)
        return out

def build_model(in_channels, hidden_channels, out_channels, heads=4):

    return GATModel(in_channels, hidden_channels, out_channels, heads=heads)
