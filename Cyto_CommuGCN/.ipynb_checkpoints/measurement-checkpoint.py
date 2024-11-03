import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . Cyto_CommuGCN import Cyto_CommuGCN


def calculation_communication_score(adata_LR, adata):
    # 交流强度矩阵的形状是 (3798, 1516)
    ligand_receptor_matrix = adata_LR.X
    ligand_receptor_df = pd.DataFrame(ligand_receptor_matrix, columns=[f'ligand_receptor_{i}' for i in range(adata_LR.X.shape[1])])

    # 计算每行的和
    communication_sum = ligand_receptor_df.sum(axis=1)
    communication_sum_df = pd.DataFrame(communication_sum, columns=['communication_sum'])

    # 将计算出的交流评分添加到 adata_LR.obs 中
    adata_LR.obs['communication_score'] = communication_sum_df['communication_sum'].values
    communication_score = 'communication_score'

    # 保持空间信息一致
    adata_LR.uns['spatial'] = adata.uns['spatial']
    adata_LR.obsm['spatial'] = adata.obsm['spatial']

    return adata_LR

