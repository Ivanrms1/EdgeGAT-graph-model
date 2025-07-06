import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class SimpleGCN(nn.Module):
    """
    GCN con 3 capas (32->64->128), BatchNorm y FC final que produce la salida.
    """
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        # Capas GCN
        self.conv1 = GCNConv(in_channels, 32)
        self.bn1   = BatchNorm(32)
        
        self.conv2 = GCNConv(32, 64)
        self.bn2   = BatchNorm(64)
        
        self.conv3 = GCNConv(64, 128)
        self.bn3   = BatchNorm(128)
        
        # Capa final FC (128 -> out_channels)
        self.fc = nn.Linear(128, out_channels)

    def forward(self, data_or_x, edge_index=None, batch=None):
        """
        Soporta dos patrones:
          1) forward(data), donde data.x y data.edge_index vienen en data.
          2) forward(x, edge_index) directamente.
        """
        if edge_index is None:
            # Caso Data de PyG
            x, edge_index = data_or_x.x, data_or_x.edge_index
        else:
            # Caso x y edge_index por separado
            x = data_or_x
        
        # Capa 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Capa 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Capa 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # FC final
        x = self.fc(x)
        return F.log_softmax(x, dim=1)