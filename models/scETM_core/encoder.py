import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from model_utils.nn import NonLinear
from typing import List, Union, Sequence
class EncoderETM(nn.Module):
    """
    scETM的编码器，将细胞基因表达数据编码为主题分布。
    
    参数:
        input_size (int): 输入特征的维度（通常是基因数量）
        output_size (int): 输出特征的维度（主题数量）
        hidden_sizes (List[int]): 隐藏层的大小列表
        bn (bool): 是否使用批标准化
        dropout_prob (float): Dropout概率
        n_batches (int): 批次数量
        input_batch_id (bool): 是否将批次ID作为输入
    """
    def __init__(self, 
                 input_size: int, 
                 output_size: int, 
                 hidden_sizes: List[int],
                 bn: bool = True,
                 dropout_prob: float = 0.1,
                 n_batches: int = 1,
                 input_batch_id: bool = False):
        super(EncoderETM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.bn = bn
        self.dropout_prob = dropout_prob
        self.n_batches = n_batches
        self.input_batch_id = input_batch_id
        
        # 计算实际输入大小，考虑批次信息
        actual_input_size = input_size + ((self.n_batches - 1) if self.input_batch_id else 0)
        
        # 构建编码器网络
        self.q_delta = self._get_fully_connected_layers(
            input_size=actual_input_size,
            hidden_sizes=hidden_sizes
        )
        
        # 均值和对数方差层
        hidden_dim = hidden_sizes[-1]
        self.mu_q_delta = nn.Linear(hidden_dim, output_size, bias=True)
        self.logsigma_q_delta = nn.Linear(hidden_dim, output_size, bias=True)
        
        # 参数范围限制
        self.min_logsigma = -10
        self.max_logsigma = 10

    def _get_fully_connected_layers(self, input_size, hidden_sizes):
        """
        构建全连接层网络
        参考自scETM的get_fully_connected_layers函数
        """
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        
        layers = []
        # 添加第一层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        if self.bn:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        if self.dropout_prob > 0:
            layers.append(nn.Dropout(self.dropout_prob))
        
        # 添加其他隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            if self.bn:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(self.dropout_prob))
        
        return nn.Sequential(*layers)

    def _get_batch_indices_oh(self, batch_indices):
        """
        获取批次索引的one-hot编码
        参考自scETM的_get_batch_indices_oh方法
        """
        batch_indices = batch_indices.unsqueeze(1)
        w_batch_id = torch.zeros((batch_indices.shape[0], self.n_batches), 
                                dtype=torch.float32, 
                                device=batch_indices.device)
        w_batch_id.scatter_(1, batch_indices, 1.)
        w_batch_id = w_batch_id[:, :self.n_batches - 1]
        return w_batch_id

    def forward(self, x, batch_indices=None):
        """
        前向传播函数
        
        参数:
            x: 输入特征张量，形状为[batch_size, input_size]
            batch_indices: 批次索引，形状为[batch_size]，默认为None
            
        返回:
            mu_q_delta: 隐变量分布的均值，形状为[batch_size, output_size]
            logsigma_q_delta: 隐变量分布的对数方差，形状为[batch_size, output_size]
        """
        # 如果需要批次信息，则添加批次索引的one-hot编码
        if self.input_batch_id and batch_indices is not None:
            batch_oh = self._get_batch_indices_oh(batch_indices)
            x = torch.cat((x, batch_oh), dim=1)
        
        # 通过编码器网络
        q_delta = self.q_delta(x)
        
        # 计算均值和对数方差
        mu_q_delta = self.mu_q_delta(q_delta)
        logsigma_q_delta = self.logsigma_q_delta(q_delta)
        logsigma_q_delta = logsigma_q_delta.clamp(self.min_logsigma, self.max_logsigma)
        
        return mu_q_delta, logsigma_q_delta