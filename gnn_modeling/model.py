import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_add_pool as gap
from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn import global_max_pool as gmaxp 
from torch_geometric.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features,
                 num_layers, 
                 hidden_channels,
                 hl_size, 
                 with_T):
        super(GCN, self).__init__()

        self.with_T = with_T

        self.conv1 = GraphConv(num_features, hidden_channels)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        self.max_pool = gmaxp
        self.mean_pool = gmp
        self.add_pool = gap
        if with_T:
            self.lin1 = Linear(3 * hidden_channels + 1, hl_size)
        else:
            self.lin1 = Linear(3 * hidden_channels, hl_size)
        self.lin2 = Linear(hl_size, hl_size)
        self.lin3 = Linear(hl_size, 1)

    def forward(self, x, edge_index, batch, T):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.conv2(x, edge_index)
        if self.with_T:
            out = torch.cat([self.max_pool(x, batch),
                         self.mean_pool(x, batch),
                         self.add_pool(x, batch),
                         T], dim=1)
        else:
            out = torch.cat([self.max_pool(x, batch),
                        self.mean_pool(x, batch),
                        self.add_pool(x, batch)], dim=1)           
        
        x = self.lin1(out)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin3(x)

        return x