from typing import Dict, Optional, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import anndata
import numpy as np
from scipy.sparse import issparse
import logging

# 设置日志
_logger = logging.getLogger(__name__)

import scipy.sparse
from torch.utils.data import Dataset, DataLoader
import torch
import anndata
from typing import Dict, Optional, Union, Callable

class SingleCellDataset(Dataset):
    def __init__(self, 
                adata: anndata.AnnData, 
                batch_col: str = 'batch_indices',
                transform: Optional[Callable] = None):
        """
        初始化单细胞数据集。
        
        参数:
            adata: AnnData对象
            batch_col: 批次列名
            transform: 数据转换函数
        """
        self.X = adata.X
        self.transform = transform
            
        # 处理批次信息
        self.has_batch = batch_col in adata.obs
        if self.has_batch:
            self.batch_indices = adata.obs[batch_col].astype('category').cat.codes.values
    
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本。
        
        参数:
            i: 样本索引
            
        返回:
            包含样本数据的字典
        """
        # 获取表达数据，处理稀疏矩阵
        if isinstance(self.X, scipy.sparse.csr_matrix):
            x = torch.FloatTensor(self.X[i].todense())
        else:
            x = torch.FloatTensor(self.X[i])
        x = x.squeeze()
        
        if self.transform:
            x = self.transform(x)
        
        result = {
            'cells': x
        }
        
        # 添加批次信息（如果有）
        if self.has_batch:
            result['batch_indices'] = torch.LongTensor([self.batch_indices[i]])
        
        return result

def create_data_loader(adata: anndata.AnnData, 
                      batch_size: int = 128, 
                      shuffle: bool = True, 
                      batch_col: str = 'batch_indices',
                      num_workers: int = 0,
                      transform: Optional[Callable] = None) -> DataLoader:
    """
    创建PyTorch DataLoader。
    
    参数:
        adata: AnnData对象
        batch_size: 批量大小
        shuffle: 是否打乱数据
        batch_col: 批次列名
        num_workers: 数据加载的工作线程数
        transform: 数据转换函数
        
    返回:
        PyTorch DataLoader
    """
    dataset = SingleCellDataset(adata, batch_col=batch_col, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=SingleCellDataset.collate_fn
    )
    
    return loader