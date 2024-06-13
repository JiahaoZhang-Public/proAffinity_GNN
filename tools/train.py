"""
    @Date       :2024/6/13 10:31
    @Author     :Jiahao Zhang
"""


import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import DataLoader
from config.config import Config
from datasets.data_loader import load_data
from model.networks.proAffinity_GNN import ProAffinityGNN


def train_model():
    data_list = load_data()
    loader = DataLoader(data_list, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = ProAffinityGNN(Config.IN_CHANNELS, Config.HIDDEN_CHANNELS, Config.OUT_CHANNELS).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(Config.EPOCHS):
        total_loss = 0
        for data in loader:
            data = data.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{Config.EPOCHS}, Loss: {total_loss / len(loader)}')


if __name__ == "__main__":
    train_model()