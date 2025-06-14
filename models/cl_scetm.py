import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Mapping, Any

# 导入核心组件
from models.scETM_core.encoder import EncoderETM
from models.scETM_core.decoder import DecoderETM

# 导入先验
from priors.mixture_prior import VampMixture
from priors.gaussian_prior import IncrementalGaussianPrior, StandardGaussianPrior

# 导入损失计算函数
from loss.RL import get_reconstruction_loss
from loss.KL import get_kl_divergence


class CL_scETM(nn.Module):
    """
    CL-scETM: 结合了scETM的单细胞分析能力和持续学习机制的模型
    
    这个模型继承了scETM的主题模型结构，并添加了持续学习能力。
    它可以使用标准先验或混合先验（用于持续学习）。
    
    参数:
        n_genes: 基因数量
        n_topics: 主题数量
        hidden_sizes: 编码器隐藏层大小
        gene_emb_dim: 基因嵌入维度
        bn: 是否使用批归一化
        dropout_prob: dropout概率
        n_batches: 批次数量
        normalize_beta: 是否标准化beta矩阵
        input_batch_id: 是否将批次ID作为输入
        enable_batch_bias: 是否添加批次特定的偏置
        enable_global_bias: 是否添加全局偏置
        prior_type: 先验类型，'standard'或'vamp'
        n_pseudoinputs: 为混合先验初始化的伪输入数量
        pseudoinputs_mean: 伪输入初始化均值
        pseudoinputs_std: 伪输入初始化标准差
        X_opt: 用于初始化伪输入的示例数据
        device: 模型使用的设备
    """
    
    # 用于聚类的输入变量名
    clustering_input = 'delta'
    
    # 可用作嵌入的变量
    emb_names = ['delta', 'theta']
    
    # logsigma的最大和最小值（防止数值问题）
    max_logsigma = 10
    min_logsigma = -10

    def __init__(self,
        n_genes: int,
        n_topics: int = 50,
        hidden_sizes: List[int] = [128],
        gene_emb_dim: int = 400,
        bn: bool = True,
        dropout_prob: float = 0.1,
        n_batches: int = 1,
        normalize_beta: bool = False,
        input_batch_id: bool = False,
        enable_batch_bias: bool = True,
        enable_global_bias: bool = False,
        prior_type: str = 'standard',
        n_pseudoinputs: int = 1,
        pseudoinputs_mean: float = 0.0,
        pseudoinputs_std: float = 0.1,
        X_opt: Optional[torch.Tensor] = None,
        prior_strength: float = 1.0,
        adaptive_strength: bool = True,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super(CL_scETM, self).__init__()
        
        # 保存参数
        self.n_genes = n_genes
        self.n_topics = n_topics
        self.gene_emb_dim = gene_emb_dim
        self.normalize_beta = normalize_beta
        self.input_batch_id = input_batch_id
        self.enable_batch_bias = enable_batch_bias
        self.enable_global_bias = enable_global_bias
        self.n_batches = n_batches
        self.prior_type = prior_type
        self.device = device
        
        # 定义编码器和解码器
        self.encoder = EncoderETM(
            input_size=n_genes,
            output_size=n_topics,
            hidden_sizes=hidden_sizes,
            bn=bn,
            dropout_prob=dropout_prob,
            n_batches=n_batches,
            input_batch_id=input_batch_id
        )
        
        self.decoder = DecoderETM(
            n_topics=n_topics,
            n_genes=n_genes,
            gene_emb_dim=gene_emb_dim,
            normalize_beta=normalize_beta,
            enable_batch_bias=enable_batch_bias,
            enable_global_bias=enable_global_bias,
            n_batches=n_batches
        )
        
        # 初始化先验
        if prior_type == 'vamp':
            # 初始化混合先验相关参数
            self.n_pseudoinputs = n_pseudoinputs
            self.pseudoinputs_mean = pseudoinputs_mean
            self.pseudoinputs_std = pseudoinputs_std
            
            # 获取示例数据，用于初始化伪输入
            self.X_opt = X_opt
            if X_opt is not None:
                mean_opt = X_opt.mean(0, keepdim=True)
                self.prior = VampMixture(pseudoinputs=[mean_opt], alpha=[1.0])
            else:
                # 否则随机初始化
                pseudoinput = torch.randn(1, n_genes) * pseudoinputs_std + pseudoinputs_mean
                self.prior = VampMixture(pseudoinputs=[pseudoinput], alpha=[1.0])
            
            # 存储编码后的伪输入分布
            self.pr_q_means = []
            self.pr_q_logvars = []
        elif prior_type == 'incremental':
            # 初始化增量高斯先验
            self.prior = IncrementalGaussianPrior(
                z_dim=n_topics,
                device=device,
                prior_strength=prior_strength,
                adaptive_strength=adaptive_strength
            )
        else:  # standard prior
            self.prior = StandardGaussianPrior(z_dim=n_topics)
        
        # 将模型移动到指定设备
        self.to(device)

    def forward(self, 
                x: torch.Tensor, 
                batch_indices: Optional[torch.Tensor] = None,
                hyper_param_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        模型的前向传播
        
        参数:
            x: 输入数据，形状为[batch_size, n_genes]
            batch_indices: 批次索引，形状为[batch_size]
            hyper_param_dict: 超参数字典，包含beta（KL权重）等
            
        返回:
            包含前向传播结果的字典
        """
        if hyper_param_dict is None:
            hyper_param_dict = {}
        
        # 使用编码器获取隐变量分布
        mu_q_delta, logsigma_q_delta = self.encoder(x, batch_indices)
        
        if self.training:
            # 训练模式下，使用重参数化采样
            delta = self.reparameterize(mu_q_delta, logsigma_q_delta)
        else:
            # 评估模式下，直接使用均值
            delta = mu_q_delta
        
        # 计算主题分布（正则化后的delta）
        theta = F.softmax(delta, dim=-1)
        
        # 使用解码器重构数据
        recon_log = self.decoder(theta, batch_indices)
        
        if not self.training:
            # 评估模式下的返回
            ret_dict = {
                'delta': mu_q_delta,  # 使用均值作为delta
                'theta': theta,  # 主题分布
                'recon_log': recon_log  # 重构的对数概率
            }
            
            # 如果需要，计算负对数似然
            if hyper_param_dict.get('decode', False):
                ret_dict['nll'] = -torch.sum(x * recon_log, dim=1).sum()
            
            return ret_dict
        
        # 训练模式下的返回
        return {
            'delta': delta,  # 采样的delta
            'theta': theta,  # 主题分布
            'recon_log': recon_log,  # 重构的对数概率
            'mu_q_delta': mu_q_delta,  # 均值
            'logsigma_q_delta': logsigma_q_delta  # 对数方差
        }

    def reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从正态分布中采样
        参考自vae/model/simple_vae.py中的reparameterize方法
        
        参数:
            mu: 均值
            logsigma: 对数标准差
            
        返回:
            采样结果
        """
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def calculate_loss(self, 
                      x: torch.Tensor, 
                      batch_indices: Optional[torch.Tensor] = None,
                      beta: float = 1.0,
                      pseudoinputs = None,
                      weights= None,
                      average: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算模型损失
        
        参数:
            x: 输入数据
            batch_indices: 批次索引
            beta: KL散度权重
            average: 是否对batch维度计算平均值
            
        返回:
            总损失和包含各组成部分的字典
        """
        # 前向传播
        fwd_dict = self.forward(x, batch_indices)
        
        # 重构损失（负对数似然）
        nll = -torch.sum(x * fwd_dict['recon_log'], dim=1)
        
        # KL散度计算
        if self.prior_type == 'vamp':
            prior_params = {
                'encoder': self.encoder,
                'pseudoinputs': pseudoinputs,
                'weights': weights
            }
            kl = get_kl_divergence(z=fwd_dict['delta'],
                                 q_mu=fwd_dict['mu_q_delta'],
                                 q_logvar=fwd_dict['logsigma_q_delta'],
                                 prior_type=self.prior_type,
                                 prior_params=prior_params)
        elif self.prior_type == 'incremental':
            # 使用增量高斯先验计算KL散度
            kl = self.prior.get_kl_divergence(
                z_mu=fwd_dict['mu_q_delta'],
                z_logvar=fwd_dict['logsigma_q_delta']
            )
        else:  # standard prior
            # 使用标准高斯先验计算KL散度
            kl = self.prior.get_kl_divergence(
                z_mu=fwd_dict['mu_q_delta'],
                z_logvar=fwd_dict['logsigma_q_delta']
            )
        
        # 计算总损失
        loss = nll + beta * kl
        
        # 如果需要求平均
        if average:
            nll = nll.mean()
            kl = kl.mean()
            loss = loss.mean()
        
        # 返回总损失和各部分
        return loss, {'nll': nll, 'kl': kl}

    def train_step(self, 
                   optimizer: torch.optim.Optimizer,
                   data_dict: Dict[str, torch.Tensor],
                   hyper_param_dict: Dict[str, Any] = None) -> Dict[str, float]:
        """
        执行一个训练步骤
        
        参数:
            optimizer: 优化器
            data_dict: 数据字典，包含'X'和'batch_indices'
            hyper_param_dict: 超参数字典，包含'kl_weight'等
            
        返回:
            训练记录字典
        """
        if hyper_param_dict is None:
            hyper_param_dict = {}
        
        # 获取数据
        x = data_dict['X']
        batch_indices = data_dict.get('batch_indices', None)
        kl_weight = hyper_param_dict.get('kl_weight', 1.0)
        
        # 前向传播和损失计算
        loss, loss_dict = self.calculate_loss(
            x=x,
            batch_indices=batch_indices,
            beta=kl_weight,
            pseudoinputs=hyper_param_dict.get('pseudoinputs', None),
            weights=hyper_param_dict.get('weights', None),
            average=True
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 返回训练记录
        return {
            'loss': float(loss.item()),
            'nll': float(loss_dict['nll'].item()),
            'kl': float(loss_dict['kl'].item()),
            'kl_weight': kl_weight
        }

    def finalize_dataset_training(self):
        """
        完成当前数据集的训练，更新增量先验
        """
        if self.prior_type == 'incremental' and isinstance(self.prior, IncrementalGaussianPrior):
            self.prior.finalize_current_dataset()
            self.prior.update_prior_from_current()
            print("增量先验已更新")

    def get_prior_info(self) -> Dict:
        """
        获取先验信息
        """
        if hasattr(self.prior, 'get_prior_info'):
            return self.prior.get_prior_info()
        else:
            return {'type': self.prior_type}

    def save_incremental_state(self) -> Dict:
        """
        保存增量训练状态
        """
        state = {
            'model_state_dict': self.state_dict(),
            'prior_type': self.prior_type,
            'model_config': {
                'n_genes': self.n_genes,
                'n_topics': self.n_topics,
                'gene_emb_dim': self.gene_emb_dim,
                'normalize_beta': self.normalize_beta,
                'n_batches': self.n_batches,
            }
        }
        
        # 保存先验状态
        if self.prior_type == 'incremental' and isinstance(self.prior, IncrementalGaussianPrior):
            state['prior_state'] = self.prior.save_state()
        
        return state

    def load_incremental_state(self, state_dict: Dict):
        """
        加载增量训练状态
        """
        # 加载模型参数
        self.load_state_dict(state_dict['model_state_dict'])
        
        # 加载先验状态
        if 'prior_state' in state_dict and self.prior_type == 'incremental':
            if isinstance(self.prior, IncrementalGaussianPrior):
                self.prior.load_state(state_dict['prior_state'])
                print("增量先验状态已加载")
