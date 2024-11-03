import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import numba


def pairwise_distance(X):
    n=X.shape[0]
    adj=np.empty((n, n), dtype=np.float32)
    for i in numba.prange(n):
        for j in numba.prange(n):
            adj[i][j]=euclid_dist(X[i], X[j])
    return adj


def euclid_dist(t1,t2):
    sum=0
    for i in range(t1.shape[0]):
        sum+=(t1[i]-t2[i])**2
    return np.sqrt(sum)


def calculate_adj_threshold(adj):
    sum_spot = adj.shape[0]
    kth_value = sum_spot // 2
    
    # Find the central values of the top 5 spots in each row
    low_line=sorted(adj[kth_value])
    adj_threshold = np.floor((low_line[6] + low_line[7]) / 2)
    
    return adj_threshold


def calculate_euclid_matrix(x, y):
        X=np.array([x, y]).T.astype(np.float32)
        return pairwise_distance(X)


def calculate_allcommu_matrix(adata, lr_file_path):
    # 将 adata.X 转化为稠密矩阵
    fx = adata.X.todense()
    fx = fx.A
    fx_nor = fx * 100000 / fx.sum(axis=1, keepdims=True)  # 数据格式为每列和为10^5
    
    # 构建成 DataFrame 格式
    Gene_Celldata = pd.DataFrame(fx_nor.T, columns=adata.obs.index, index=adata.var.index)
    
    # 读取 L-R 数据
    L_Rdata = pd.read_excel(lr_file_path)
    L_Rdata = pd.DataFrame(L_Rdata)
    
    # 筛选出在 Gene_Celldata 中存在的 Ligand 和 Receptor 对
    index = Gene_Celldata.reset_index()['index']
    L_R = L_Rdata[(L_Rdata.Ligand.isin(index)) & (L_Rdata.Receptor.isin(index))]
    
    # 准备计算 adj3 矩阵
    G_Cdata = pd.DataFrame(fx_nor, index=adata.obs.index, columns=adata.var.index)
    L = L_R['Ligand']
    R = L_R['Receptor']
    L_L = G_Cdata[L]
    R_R = G_Cdata[R]
    
    LL = np.mat(L_L.values)
    RR = np.mat(R_R.values)
    
    # 创建 L_R_pair 列
    L_R['L_R_pair'] = L_R.apply(lambda row: f"{row['Ligand']}_{row['Receptor']}", axis=1)
    
    # 计算 adj3
    allcommu_matrix = np.dot(LL, RR.T) + np.dot(RR, LL.T)
    
    return allcommu_matrix, L_R

def adj_commu(adata, lr_file_path, adj, adj_r):
    """

    参数:
    adata : AnnData
    lr_file_path : str
    adj_Ed : numpy.ndarray
    adj_threshold : float
    commu_matrix : numpy.ndarray
    G_C : pandas.DataFrame
    L_R : pandas.DataFrame
    
    """
    
    # 计算通讯矩阵和配体-受体对数据
    allcommu_matrix, L_R = calculate_allcommu_matrix(adata, lr_file_path)
    
    # 初始化通讯矩阵
    num_cells = adata.X.shape[0]
    commu_matrix = np.zeros((num_cells, num_cells))
    
    # 更新通讯矩阵
    for i in range(num_cells):
        commu_matrix[i, i] = allcommu_matrix[i, i]
        for j in range(num_cells):
            if adj[i, j] < adj_r and adj[i, j] != 0:
                commu_matrix[i, j] = allcommu_matrix[i, j]
    
    # 归一化处理
    np.fill_diagonal(commu_matrix, 0)
    non_zero_sum = np.sum(commu_matrix[commu_matrix != 0])
    num_spot = commu_matrix.shape[1]
    commu_matrix_normalized = (commu_matrix / non_zero_sum) * num_spot
    np.fill_diagonal(commu_matrix_normalized, 1)
    
    # 将 adata.X 转化为稠密矩阵并构建 DataFrame
    fx = adata.X.todense()
    fx = fx.A
    G_C = pd.DataFrame(fx, columns=adata.var.index, index=adata.obs.index)
    
    return commu_matrix_normalized, G_C, L_R



def interaction_score_matrix(adj1, G_C, L_R,adj_r):
    num_spots = adj1.shape[0]

    # Convert DataFrame to numpy array for faster calculations
    G_C_np = G_C.values
    ligand_indices = [G_C.columns.get_loc(ligand) for ligand in L_R['Ligand']]
    receptor_indices = [G_C.columns.get_loc(receptor) for receptor in L_R['Receptor']]
    L_R_pairs = [f"{row['Ligand']}_{row['Receptor']}" for _, row in L_R.iterrows()]

    # Initialize the result array
    result_np = np.zeros((num_spots, len(L_R_pairs)))

    # For each spot
    for i in range(num_spots):
        neighbors_idx = np.where((adj1[i] < adj_r) & (adj1[i] > 0))[0]
        
        for j, (ligand_index, receptor_index) in enumerate(zip(ligand_indices, receptor_indices)):
            ligand_value = G_C_np[i, ligand_index]
            neighbor_receptor_values = G_C_np[neighbors_idx, receptor_index]
            spot_receptor_value = G_C_np[i, receptor_index]
            neighbor_ligand_values = G_C_np[neighbors_idx, ligand_index]
            
            interaction = ligand_value * neighbor_receptor_values + neighbor_ligand_values * spot_receptor_value
            result_np[i, j] = interaction.sum()

    result_df = pd.DataFrame(result_np, index=G_C.index, columns=L_R_pairs)
    
    return result_df