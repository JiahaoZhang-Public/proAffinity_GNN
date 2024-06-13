# ProAffinity-GNN

## 项目描述
本项目实现了论文["ProAffinity-GNN: A Novel Approach to Structure-based Protein-Protein Binding Affinity Prediction via a Curated Dataset and Graph Neural Networks"](https://www.biorxiv.org/content/10.1101/2024.03.14.584935v1)
中的模型。

## 文件结构
- config/: 配置文件
- datasets/: 数据加载和处理脚本
- model/: 模型定义
- results/: 训练结果保存
- tools/: 训练和评估脚本
- utils/: 辅助函数
- main.py: 主程序入口

## 使用方法
1. 配置环境和依赖库
2. 将PDB文件和FASTA文件放在datasets/pdb_files/和datasets/fasta_files/目录下
3. 运行`main.py`开始训练模型