import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATConv
from torch_geometric.nn import knn_graph

#---------------------------
# ResiudalMLP para EdgeConv
#---------------------------

class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(ResidualMLP, self).__init__()
        #Fully conected + batchnorm 1
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        #Fully conected + batchnorm 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        #Fully conected + batchnorm 3
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Si la entrada no coincide con la de salida se adapta la shortcut
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)   #Skip MLP
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        return out + residual     
    
class GATEdgeConvHybrid(nn.Module):
    """
    Arquitectura:
      1. dos capas EdgeConv para capturar rasgos locales DGCNN style.
      2. dos capas GATConv para refinar con atención.
      3. Capa de clasificación.
    
    Params:
      in_channels    : dimensión de entrada (xyz, normales o curvatura).
      hidden_channels: canales en capas ocultas.
      out_channels   : número de clases de salida.
      heads          : número de cabezas de atención.
      k              : número de vecinos para knn_graph (si dinámico).
      dynamic_graph  : recalcular la gráfica en cada forward o no.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, k=16, dynamic_graph=False):
        super(GATEdgeConvHybrid, self).__init__()
        self.k = k
        self.dynamic_graph = dynamic_graph

        #  Primera capa 
        #   - El MLP recibe concat(x_i, x_j) o (x_i, x_j - x_i).
        self.edgeconv1 = EdgeConv(
            nn=ResidualMLP(in_dim=in_channels*2, hidden_dim=hidden_channels, out_dim=hidden_channels),
            aggr='sum'
        )

        #  Segunda capa EdgeConv 
        self.edgeconv2 = EdgeConv(
            nn=ResidualMLP(in_dim=hidden_channels*2, hidden_dim=hidden_channels, out_dim=hidden_channels),
            aggr='mean'
        )

        # Primera capa GAT
        #   si concat=True la salida tiene hidden_channels*heads
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)

        # Segunda capa GAT
        #    Reducimos de hidden_channels*heads a hidden_channels
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        # Clasificador final
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges] 
         None si dynamic_graph=True si recalculamos en cada forward.
        batch: distintos tamaños en un batch.
        """

        if self.dynamic_graph:
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        #  EdgeConv capa 1
        x = self.edgeconv1(x, edge_index)
        x = F.relu(x)

        #  EdgeConv capa 2
        x = self.edgeconv2(x, edge_index)
        x = F.relu(x)

        # GAT 1
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        #  GAT 2
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Clasificador
        out = self.classifier(x)
        return out

