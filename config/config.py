"""
    @Date       :2024/6/13 10:31
    @Author     :Jiahao Zhang
"""
import torch


class Config:
    # 数据路径
    PDB_FILES_PATH = 'datasets/pdb_files/'
    FASTA_FILES_PATH = 'datasets/fasta_files/'

    # 模型参数
    IN_CHANNELS = 1280  # 节点特征维度
    HIDDEN_CHANNELS = 256
    OUT_CHANNELS = 1

    # 训练参数
    EPOCHS = 25
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 0.001
    DROPOUT_RATE = 0.5

    # 其他参数
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'