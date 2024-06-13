# ProAffinity-GNN

## 项目描述
本项目用于复现论文["ProAffinity-GNN: A Novel Approach to Structure-based Protein-Protein Binding Affinity Prediction via a Curated Dataset and Graph Neural Networks"](https://www.biorxiv.org/content/10.1101/2024.03.14.584935v1)
中的模型。
由于缺乏相关生物背景，且缺少人力，难以获取论文中的数据集。
## 文件结构
- config/: 配置文件
- datasets/: 数据加载和处理脚本
- model/: 模型定义
- results/: 训练结果保存
- tools/: 训练和评估脚本
- utils/: 辅助函数
- main.py: 主程序入口

## 数据集收集及处理

### 数据收集
1. 从蛋白质数据银行（PDB）下载包含蛋白质-蛋白质复合物的PDB文件。
2. 提取蛋白质序列信息并保存为FASTA格式文件。

### 数据处理
1. 使用AutoDockFR将PDB文件转换为PDBQT格式，并添加极性氢以提高分子表示的准确性。
2. 使用ESM-2模型（进化尺度模型）生成蛋白质序列的嵌入，每个残基生成一个1280维的嵌入向量。
3. 构建蛋白质内和蛋白质间的图结构：
    - 蛋白质内图：如果两个节点的距离在3.5Å以内，则在它们之间添加一条边。
    - 蛋白质间图：使用15Å的距离阈值，连接两个蛋白质之间的节点。
4. 生成图的节点特征和边信息。
5. 从数据库记录或实验结果中获取相应的标签，用于训练和测试。

## 快速开始
1. 配置环境和依赖库：
    ```sh
    pip install -r requirements.txt
    ```

2. 下载PDB文件和FASTA文件，并放置在`datasets/pdb_files/`和`datasets/fasta_files/`目录下。

3. 运行`main.py`开始训练模型：
    ```sh
    python main.py
    ```

## 参考文献
ProAffinity-GNN: A Novel Approach to Structure-based Protein-Protein Binding Affinity Prediction via a Curated Dataset and Graph Neural Networks
Zhiyuan Zhou, Yueming Yin, Hao Han, Yiping Jia, Jun Hong Koh, Adams Wai-Kin Kong, Yuguang Mu
bioRxiv 2024.03.14.584935; doi: https://doi.org/10.1101/2024.03.14.58493
