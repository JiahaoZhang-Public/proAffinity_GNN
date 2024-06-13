"""
    @Date       :2024/6/13 10:31
    @Author     :Jiahao Zhang
    @Structure  :
                    ProAffinityGNN
                ======================
                Input: Graph Data
                ----------------------
                          |
                          v
                +--------------------+
                |    AttentiveFP     |
                +--------------------+
                          |
                          v
                +--------------------+
                |   Concatenate      |
                | (with Super Node)  |
                +--------------------+
                          |
                          v
                +--------------------+
                |     Dropout        |
                | (rate=Config.      |
                |   DROPOUT_RATE)    |
                +--------------------+
                          |
                          v
                +--------------------+
                |    Linear (FC1)    |
                | (hidden_channels*2 |
                |  -> hidden_channels) |
                +--------------------+
                          |
                          v
                +--------------------+
                |       ReLU         |
                +--------------------+
                          |
                          v
                +--------------------+
                |    Linear (FC2)    |
                | (hidden_channels   |
                |  -> out_channels)  |
                +--------------------+
                          |
                          v
                +--------------------+
                |     Output:        |
                |   Predicted Value  |
                +--------------------+

"""
import torch
import torch.nn as nn
from config.config import Config
from model.module.attentive_fp import AttentiveFP


class ProAffinityGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ProAffinityGNN, self).__init__()
        self.attentive_fp = AttentiveFP(in_channels, hidden_channels)
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)  # 超节点的特征加上原节点特征
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(Config.DROPOUT_RATE)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.attentive_fp(x, edge_index, data.batch)

        h = self.dropout(h)
        h = self.fc1(h)
        h = torch.relu(h)
        h = self.fc2(h)
        return h