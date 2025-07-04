# model_edgegat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, GATConv
from torch_geometric.nn import knn_graph

###################
# MLP para EdgeConv
###################

def build_mlp(in_dim, hidden_dim, out_dim, dropout=0.2):
    """
    MLP mejorado:
      - 3 capas lineales con activaciones ReLU
      - BatchNorm después de cada capa lineal (menos en la salida)
      - Dropout para regularización
    """
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
        
        # Si la dimensión de entrada no coincide con la de salida se adapta la shortcut
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.fc3(out)

        return out + residual
    
class GATEdgeConvHybrid(nn.Module):
    """
    Arquitectura híbrida:
      1.  dos capas EdgeConv para capturar rasgos locales DGCNN style.
      2. dos capas GATConv para refinar con atención.
      3. Capa de clasificación.
    
    Params:
      in_channels    (int): dimensión de entrada (xyz, normales o curvatura).
      hidden_channels(int): canales en capas ocultas.
      out_channels   (int): número de clases de salida.
      heads          (int): número de cabezas de atención.
      k              (int): número de vecinos para knn_graph (si dinámico).
      dynamic_graph  (bool): recalcular la gráfica en cada forward o no.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, k=16, dynamic_graph=False):
        super(GATEdgeConvHybrid, self).__init__()
        self.k = k
        self.dynamic_graph = dynamic_graph

        #  Primera capa 
        #   - El MLP recibe concat(x_i, x_j) o (x_i, x_j - x_i). Aquí usamos concat.
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
        #   concat=True => la salida tiene hidden_channels*heads
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)

        # Segunda capa GAT
        #    Reducimos de hidden_channels*heads a hidden_channels
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        # 5. Clasificador final
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges] (grafo knn pre-calculado) 
                    o None si dynamic_graph=True y queremos recalcular en cada forward.
        batch: (opcional) para manejar nubes de distintos tamaños en un batch.
        """

        if self.dynamic_graph:
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        #  EdgeConv 1
        x = self.edgeconv1(x, edge_index)
        x = F.relu(x)

        #  EdgeConv 2
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

class GATEdgeConvHybrid(nn.Module):
    """
    Arquitectura híbrida con tres capas 
    y skip connections, seguida de dos capas GAT y una capa final de clasificación.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, k=16, dynamic_graph=False):
        super(GATEdgeConvHybrid, self).__init__()
        self.k = k
        self.dynamic_graph = dynamic_graph

        # primera capa 
        self.edgeconv1 = EdgeConv(
            nn=build_mlp(in_dim=in_channels * 2, hidden_dim=hidden_channels, out_dim=hidden_channels),
            aggr='mean'
        )

        # segunda capa 
        self.edgeconv2 = EdgeConv(
            nn=build_mlp(in_dim=hidden_channels * 2, hidden_dim=hidden_channels, out_dim=hidden_channels),
            aggr='sum'
        )

        # Tercera 
        self.edgeconv3 = EdgeConv(
            nn=build_mlp(in_dim=hidden_channels * 2, hidden_dim=hidden_channels, out_dim=hidden_channels),
            aggr='mean'
        )

        if in_channels != hidden_channels:
            self.shortcut = nn.Linear(in_channels, hidden_channels)
        else:
            self.shortcut = None

        #  GAT
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=False)

        # Clasificador lineal
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        if self.dynamic_graph:
            edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        #
        x_input = x
        if self.shortcut is not None:
            x_input = self.shortcut(x_input)

        # Primera capa 
        x1 = self.edgeconv1(x, edge_index)
        x1 = F.relu(x1)

        # Segunda capa 
        x2 = self.edgeconv2(x1, edge_index)
        x2 = F.relu(x2)

        # Primera skip connection
        out_edge = x2 + x_input

        # Tercera capa 
        x3 = self.edgeconv3(out_edge, edge_index)
        x3 = F.relu(x3)

        # Segunda skip connection
        out_edge = x3 + out_edge

        # Capas GAT
        x_gat = self.gat1(out_edge, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = self.gat2(x_gat, edge_index)
        x_gat = F.elu(x_gat)

        # Capa final de clasificación
        out = self.classifier(x_gat)
        return out




def build_edgegat_model(in_channels, hidden_channels, out_channels, heads=4, k=16, dynamic_graph=False):
    return GATEdgeConvHybrid(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        k=k,
        dynamic_graph=dynamic_graph
    )

