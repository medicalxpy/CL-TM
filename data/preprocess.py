import os
from typing import Union, Tuple, Optional, List
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import logging
from scipy.sparse import issparse

# 设置日志
_logger = logging.getLogger(__name__)

def read_data(file_path: str, 
              file_format: str = 'h5ad', 
              **kwargs) -> anndata.AnnData:
    """
    读取单细胞RNA-seq数据。
    
    参数:
        file_path: 数据文件路径
        file_format: 数据格式，支持'h5ad'(AnnData),'csv','10x'等
        **kwargs: 传递给读取函数的额外参数
        
    返回:
        AnnData对象
    """
    if file_format == 'h5ad':
        adata = sc.read_h5ad(file_path)
    elif file_format == 'csv':
        adata = sc.read_csv(file_path, **kwargs)
    elif file_format == '10x':
        adata = sc.read_10x_mtx(file_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {file_format}")
    
    _logger.info(f"读取了{adata.n_obs}个细胞，{adata.n_vars}个基因")
    return adata

def preprocess_data(adata: anndata.AnnData, 
                   normalize: bool = True,
                   log_transform: bool = True,
                   scale: bool = False,
                   min_cells: int = 3,
                   min_genes: int = 200,
                   max_genes: Optional[int] = None,
                   min_counts: Optional[int] = None,
                   target_sum: Optional[int] = None,
                   hvg_selection: bool = False,
                   n_top_genes: int = 2000,
                   batch_key: Optional[str] = None) -> anndata.AnnData:
    """
    预处理单细胞RNA-seq数据。
    
    参数:
        adata: AnnData对象
        normalize: 是否进行归一化
        log_transform: 是否进行对数变换
        scale: 是否进行缩放
        min_cells: 过滤掉在少于min_cells个细胞中表达的基因
        min_genes: 过滤掉表达少于min_genes个基因的细胞
        max_genes: 过滤掉表达超过max_genes个基因的细胞
        min_counts: 最小表达计数
        target_sum: 归一化目标总和
        hvg_selection: 是否选择高变异基因
        n_top_genes: 保留多少个高变异基因
        batch_key: 批次键名，用于批次效应处理
        
    返回:
        预处理后的AnnData对象
    """
    # 复制数据，避免修改原始数据
    adata = adata.copy()
    
    # 基本过滤
    sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    
    # 归一化和对数变换
    if normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum)
    
    if log_transform:
        sc.pp.log1p(adata)
    
    # 选择高变异基因
    if hvg_selection:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, batch_key=batch_key)
        adata = adata[:, adata.var.highly_variable]
    
    # 缩放数据
    if scale:
        sc.pp.scale(adata)
    
    _logger.info(f"预处理后数据：{adata.n_obs}个细胞，{adata.n_vars}个基因")
    return adata

def train_test_split(adata: anndata.AnnData, 
                    test_ratio: float = 0.1, 
                    seed: int = 1) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    将数据分割为训练集和测试集。
    参考自scETM.trainers.trainer_utils.train_test_split
    
    参数:
        adata: AnnData对象
        test_ratio: 测试集比例
        seed: 随机种子
        
    返回:
        训练集和测试集的AnnData对象
    """
    rng = np.random.default_rng(seed=seed)
    test_indices = rng.choice(adata.n_obs, size=int(test_ratio * adata.n_obs), replace=False)
    train_indices = list(set(range(adata.n_obs)).difference(test_indices))
    train_adata = adata[adata.obs_names[train_indices], :]
    test_adata = adata[adata.obs_names[test_indices], :]
    _logger.info(f'保留{test_adata.n_obs}个细胞（{test_ratio:g}）作为测试数据。')
    return train_adata, test_adata

def setup_anndata(adata: anndata.AnnData, 
                 batch_col: str = 'batch_indices', 
                 cell_type_col: str = 'cell_types') -> anndata.AnnData:
    """
    设置AnnData对象以便用于模型。
    
    参数:
        adata: AnnData对象
        batch_col: 批次列名
        cell_type_col: 细胞类型列名
        
    返回:
        设置好的AnnData对象
    """
    adata = adata.copy()
    
    # 确保批次是分类数据类型
    if batch_col in adata.obs and not pd.api.types.is_categorical_dtype(adata.obs[batch_col]):
        adata.obs[batch_col] = adata.obs[batch_col].astype('category')
    
    # 确保细胞类型是分类数据类型
    if cell_type_col in adata.obs and not pd.api.types.is_categorical_dtype(adata.obs[cell_type_col]):
        adata.obs[cell_type_col] = adata.obs[cell_type_col].astype('category')
    
    return adata

def prepare_for_continual_learning(adata_list: List[anndata.AnnData], 
                                 batch_key: str = 'batch_indices', 
                                 merge: bool = False) -> Union[List[anndata.AnnData], anndata.AnnData]:
    """
    准备用于持续学习的数据集。
    
    参数:
        adata_list: AnnData对象列表
        batch_key: 批次键名，用于标记不同数据集
        merge: 是否合并数据集
        
    返回:
        处理后的AnnData对象或对象列表
    """
    # 拷贝数据，避免修改原始数据
    adata_list = [adata.copy() for adata in adata_list]
    
    if merge:
        # 在合并前添加批次信息
        for i, adata in enumerate(adata_list):
            adata.obs[batch_key] = i
        
        # 合并数据集
        merged_adata = adata_list[0].concatenate(adata_list[1:], join='outer', batch_key=batch_key)
        
        # 确保批次信息是分类类型
        merged_adata.obs[batch_key] = merged_adata.obs[batch_key].astype('category')
        
        return merged_adata
    else:
        # 如果不合并，为每个数据集添加批次信息
        for i, adata in enumerate(adata_list):
            adata.obs[batch_key] = i
            adata.obs[batch_key] = adata.obs[batch_key].astype('category')
        
        return adata_list

def prepare_for_transfer(source_adata: anndata.AnnData, 
                        target_adata: anndata.AnnData,
                        gene_list: Optional[List[str]] = None) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """
    准备用于迁移学习的数据。
    参考自scETM.trainers.trainer_utils.prepare_for_transfer
    
    参数:
        source_adata: 源数据集
        target_adata: 目标数据集
        gene_list: 用于对齐的基因列表，如果为None，则使用两个数据集的交集
        
    返回:
        对齐后的源数据集和目标数据集
    """
    source_adata = source_adata.copy()
    target_adata = target_adata.copy()
    
    if gene_list is None:
        # 获取两个数据集的基因交集
        source_genes = source_adata.var_names
        target_genes = target_adata.var_names
        common_genes = list(set(source_genes).intersection(set(target_genes)))
        
        if len(common_genes) == 0:
            raise ValueError("源数据集和目标数据集没有共同的基因！")
        
        _logger.info(f"源数据集和目标数据集有{len(common_genes)}个共同基因。")
        gene_list = common_genes
    
    # 确保基因列表中的所有基因都在两个数据集中
    source_genes = set(source_adata.var_names)
    target_genes = set(target_adata.var_names)
    
    valid_genes = [gene for gene in gene_list if gene in source_genes and gene in target_genes]
    
    if len(valid_genes) == 0:
        raise ValueError("没有找到两个数据集共有的有效基因！")
    
    # 筛选共同基因
    source_adata_aligned = source_adata[:, valid_genes].copy()
    target_adata_aligned = target_adata[:, valid_genes].copy()
    
    return source_adata_aligned, target_adata_aligned