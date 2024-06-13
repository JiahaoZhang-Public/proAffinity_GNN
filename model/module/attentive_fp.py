"""
    @Date       :2024/6/13 10:31
    @Author     :Jiahao Zhang
    @Structure  :
                AttentiveFP Module
                ======================
                Input: Node Features, Edge Index, Batch
                ----------------------
                          |
                          v
                +--------------------+
                |  GATConv Layer 1   |
                | (in_channels ->    |
                |  hidden_channels)  |
                +--------------------+
                          |
                          v
                +--------------------+
                |      ReLU          |
                +--------------------+
                          |
                          v
                +--------------------+
                |     Dropout        |
                +--------------------+
                          |
                          v
                +--------------------+
                |  GATConv Layer 2   |
                | (hidden_channels   |
                |  -> hidden_channels)|
                +--------------------+
                          |
                          v
                +--------------------+
                |      ReLU          |
                +--------------------+
                          |
                          v
                +--------------------+
                |     Dropout        |
                +--------------------+
                          |
                          v
                +--------------------+
                |       GRU          |
                |  (hidden_channels  |
                |  -> hidden_channels)|
                +--------------------+
                          |
                          v
                +--------------------+
                |   Global Add Pool  |
                |   (over Batch)     |
                +--------------------+
                          |
                          v
                +--------------------+
                |  Super Node FC     |
                | (hidden_channels   |
                |  -> hidden_channels)|
                +--------------------+
                          |
                          v
                |       ReLU         |
                +--------------------+
                          |
                          v
                +--------------------+
                | Concatenate Super  |
                | Node with Nodes    |
                +--------------------+
                          |
                          v
                |     Output:        |
                |  Aggregated Features|
                +--------------------+
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_add_pool


class AttentiveFP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout_rate=0.5):
        super(AttentiveFP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_channels, hidden_channels)
        self.super_node_fc = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = torch.relu(h)
            h = self.dropout(h)

        h, _ = self.gru(h.unsqueeze(0))
        h = h.squeeze(0)
        h = global_add_pool(h, batch)

        # 超节点处理
        super_node = torch.mean(h, dim=0, keepdim=True)
        super_node = self.super_node_fc(super_node)
        super_node = torch.relu(super_node)

        # 聚合超节点信息
        h = torch.cat([h, super_node.repeat(h.size(0), 1)], dim=1)
        return h
