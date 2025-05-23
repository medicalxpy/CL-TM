import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional

class DecoderETM(nn.Module):
    """
    scETM的解码器，将主题分布(theta)解码为基因表达谱。
    
    参数:
        n_topics (int): 主题数量
        n_genes (int): 基因数量
        gene_emb_dim (int): 基因嵌入维度
        normalize_beta (bool): 是否在使用beta矩阵前对其进行标准化，如原始ETM中所做
        enable_batch_bias (bool): 是否添加批次特定的偏置，用于批次效应校正
        enable_global_bias (bool): 是否添加全局偏置
        n_batches (int): 批次数量
    """
    def __init__(self, 
                 n_topics: int, 
                 n_genes: int, 
                 gene_emb_dim: int = 400,
                 normalize_beta: bool = False,
                 enable_batch_bias: bool = True,
                 enable_global_bias: bool = False,
                 n_batches: int = 1):
        super(DecoderETM, self).__init__()
        
        self.n_topics = n_topics
        self.n_genes = n_genes
        self.gene_emb_dim = gene_emb_dim
        self.normalize_beta = normalize_beta
        self.enable_batch_bias = enable_batch_bias
        self.enable_global_bias = enable_global_bias
        self.n_batches = n_batches
        
        # 主题嵌入矩阵 (参考自scETM.py中的alpha参数)
        self.alpha = nn.Parameter(torch.randn(self.n_topics, self.gene_emb_dim))
        
        # 基因嵌入矩阵 (参考自scETM.py中的rho相关参数)
        self.rho = nn.Parameter(torch.randn(self.gene_emb_dim, self.n_genes))
        
        # 批次特定偏置 (参考自scETM.py中的_init_batch_and_global_biases方法)
        if self.enable_batch_bias and self.n_batches > 1:
            self.batch_bias = nn.Parameter(torch.randn(self.n_batches, self.n_genes))
        else:
            self.register_parameter('batch_bias', None)
        
        # 全局偏置
        if self.enable_global_bias:
            self.global_bias = nn.Parameter(torch.randn(1, self.n_genes))
        else:
            self.register_parameter('global_bias', None)

    def get_beta(self):
        """
        获取主题-基因权重矩阵beta
        
        返回:
            beta: 主题-基因权重矩阵，形状为[n_topics, n_genes]
        """
        # 计算beta = alpha @ rho (参考自scETM.py中的decode方法)
        beta = torch.mm(self.alpha, self.rho)
        
        if self.normalize_beta:
            # 如果需要标准化beta，则应用softmax (参考自scETM.py中的decode方法)
            return F.softmax(beta, dim=-1)
        else:
            return beta

    def forward(self, theta: torch.Tensor, batch_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播函数，将主题分布解码为基因表达谱
        
        参数:
            theta: 主题分布，形状为[batch_size, n_topics]
            batch_indices: 批次索引，形状为[batch_size]，默认为None
            
        返回:
            recon_log: 重构的对数基因表达谱，形状为[batch_size, n_genes]
        """
        # 获取beta矩阵
        beta = self.get_beta()
        
        if self.normalize_beta:
            # 如果使用标准化的beta，直接计算重构 (参考自scETM.py中的decode方法)
            recon = torch.mm(theta, beta)
            recon_log = (recon + 1e-30).log()
        else:
            # 否则，计算重构并应用softmax (参考自scETM.py中的decode方法)
            recon_logit = torch.mm(theta, beta)
            
            # 添加全局偏置
            if self.enable_global_bias:
                recon_logit += self.global_bias
                
            # 添加批次特定偏置
            if self.enable_batch_bias and batch_indices is not None and self.batch_bias is not None:
                recon_logit += self.batch_bias[batch_indices]
                
            # 应用log_softmax生成对数概率
            recon_log = F.log_softmax(recon_logit, dim=-1)
            
        return recon_log