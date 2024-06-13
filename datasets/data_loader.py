"""
    @Date       :2024/6/13 10:31
    @Author     :Jiahao Zhang
"""

import os
import torch
from torch_geometric.data import Data, DataLoader
from config.config import Config
from biopandas.pdb import PandasPdb
from transformers import EsmModel, EsmTokenizer


def load_data():
    pdb_files = [os.path.join(Config.PDB_FILES_PATH, f) for f in os.listdir(Config.PDB_FILES_PATH)]
    fasta_files = [os.path.join(Config.FASTA_FILES_PATH, f) for f in os.listdir(Config.FASTA_FILES_PATH)]

    data_list = preprocess_data(pdb_files, fasta_files)
    return data_list


def preprocess_data(pdb_files, fasta_files):
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

    data_list = []
    for pdb_file, fasta_file in zip(pdb_files, fasta_files):
        nodes, edges, labels = process_pdb_and_fasta(pdb_file, fasta_file, tokenizer, model)
        data = Data(x=nodes, edge_index=edges, y=labels)
        data_list.append(data)
    return data_list


def process_pdb_and_fasta(pdb_file, fasta_file, tokenizer, model):
    # 处理PDB文件
    ppdb = PandasPdb().read_pdb(pdb_file)
    atoms = ppdb.df['ATOM']
    coordinates = atoms[['x_coord', 'y_coord', 'z_coord']].values
    edges = compute_edges(coordinates)

    # 处理FASTA文件
    with open(fasta_file, 'r') as f:
        sequence = f.read().splitlines()[1]
    inputs = tokenizer(sequence, return_tensors='pt', max_length=1024, truncation=True)
    outputs = model(**inputs)
    nodes = outputs.last_hidden_state.squeeze(0)

    # 示例标签，实际应用中应从数据集中提取真实标签
    labels = torch.tensor([0.5])

    return nodes, edges, labels


def compute_edges(coordinates, intra_cutoff=3.5, inter_cutoff=15.0):
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(coordinates))
    edges_intra = (distances < intra_cutoff).astype(int)
    edges_inter = (distances < inter_cutoff).astype(int)
    edges = edges_intra + edges_inter
    edge_index = torch.nonzero(torch.tensor(edges)).t().contiguous()
    return edge_index