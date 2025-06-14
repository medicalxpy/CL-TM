import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple


class IncrementalGaussianPrior(nn.Module):
    """
    增量学习的高斯先验，基于上一次训练的编码器输出统计信息
    
    该先验保存了前一个数据集训练后编码器输出的均值和方差统计，
    用作下一个数据集训练时的先验分布。
    """
    
    def __init__(self, 
                 z_dim: int,
                 device: torch.device = None,
                 prior_strength: float = 1.0,
                 adaptive_strength: bool = True):
        """
        初始化增量高斯先验
        
        参数:
            z_dim: 隐变量维度
            device: 计算设备
            prior_strength: 先验强度，控制历史信息的影响力
            adaptive_strength: 是否根据数据集大小自适应调整先验强度
        """
        super().__init__()
        
        self.z_dim = z_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prior_strength = prior_strength
        self.adaptive_strength = adaptive_strength
        
        # 历史统计信息
        self.has_prior = False
        self.register_buffer('prior_mu', torch.zeros(z_dim))
        self.register_buffer('prior_logvar', torch.zeros(z_dim))
        self.register_buffer('prior_count', torch.tensor(0.0))
        
        # 当前数据集的统计信息
        self.register_buffer('current_mu', torch.zeros(z_dim))
        self.register_buffer('current_logvar', torch.zeros(z_dim))
        self.register_buffer('current_count', torch.tensor(0.0))
        
        # 运行时统计累积器
        self.running_sum = torch.zeros(z_dim, device=self.device)
        self.running_sum_sq = torch.zeros(z_dim, device=self.device)
        self.running_count = 0.0
        
        self.to(self.device)
    
    def update_statistics(self, z_mu: torch.Tensor, z_logvar: torch.Tensor):
        """
        更新当前数据集的统计信息（在训练过程中调用）
        
        参数:
            z_mu: 编码器输出的均值 [batch_size, z_dim]
            z_logvar: 编码器输出的对数方差 [batch_size, z_dim]
        """
        if not self.training:
            return
            
        batch_size = z_mu.size(0)
        
        # 累积统计量
        with torch.no_grad():
            self.running_sum += z_mu.sum(dim=0)
            self.running_sum_sq += (z_mu ** 2).sum(dim=0)
            self.running_count += batch_size
    
    def finalize_current_dataset(self):
        """
        完成当前数据集的训练，计算最终统计信息
        """
        if self.running_count > 1:
            # 计算当前数据集的均值和方差
            self.current_mu = self.running_sum / self.running_count
            current_var = (self.running_sum_sq / self.running_count) - (self.current_mu ** 2)
            current_var = torch.clamp(current_var, min=1e-8)  # 防止数值问题
            self.current_logvar = torch.log(current_var)
            self.current_count = torch.tensor(self.running_count)
            
            print(f"当前数据集统计信息已计算: 样本数={self.running_count:.0f}")
            print(f"均值范围: [{self.current_mu.min():.4f}, {self.current_mu.max():.4f}]")
            print(f"方差范围: [{current_var.min():.4f}, {current_var.max():.4f}]")
    
    def update_prior_from_current(self):
        """
        将当前数据集的统计信息更新为先验（在完成一个数据集训练后调用）
        """
        if self.current_count > 0:
            if not self.has_prior:
                # 第一次设置先验
                self.prior_mu = self.current_mu.clone()
                self.prior_logvar = self.current_logvar.clone()
                self.prior_count = self.current_count.clone()
                self.has_prior = True
                print("设置初始先验分布")
            else:
                # 融合历史先验和当前统计
                total_count = self.prior_count + self.current_count
                weight_prior = self.prior_count / total_count
                weight_current = self.current_count / total_count
                
                # 加权平均更新均值
                new_mu = weight_prior * self.prior_mu + weight_current * self.current_mu
                
                # 更新方差（考虑均值变化）
                prior_var = torch.exp(self.prior_logvar)
                current_var = torch.exp(self.current_logvar)
                
                # 使用加权方差公式
                new_var = (weight_prior * (prior_var + (self.prior_mu - new_mu) ** 2) + 
                          weight_current * (current_var + (self.current_mu - new_mu) ** 2))
                new_var = torch.clamp(new_var, min=1e-8)
                
                self.prior_mu = new_mu
                self.prior_logvar = torch.log(new_var)
                self.prior_count = total_count
                
                print(f"先验分布已更新: 总样本数={total_count:.0f}")
            
            # 重置当前统计
            self.reset_current_statistics()
    
    def reset_current_statistics(self):
        """
        重置当前数据集的统计信息
        """
        self.running_sum.zero_()
        self.running_sum_sq.zero_()
        self.running_count = 0.0
        self.current_mu.zero_()
        self.current_logvar.zero_()
        self.current_count.zero_()
    
    def get_kl_divergence(self, 
                         z_mu: torch.Tensor, 
                         z_logvar: torch.Tensor) -> torch.Tensor:
        """
        计算与先验分布的KL散度
        
        参数:
            z_mu: 编码器输出的均值 [batch_size, z_dim]
            z_logvar: 编码器输出的对数方差 [batch_size, z_dim]
            
        返回:
            KL散度 [batch_size]
        """
        # 更新统计信息
        self.update_statistics(z_mu, z_logvar)
        
        if not self.has_prior:
            # 没有先验时，使用标准正态分布
            kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
        else:
            # 使用历史先验
            effective_strength = self.prior_strength
            if self.adaptive_strength and self.current_count > 0:
                # 根据数据集大小调整先验强度
                effective_strength = self.prior_strength * torch.sqrt(self.prior_count / (self.prior_count + self.current_count))
            
            # KL(q(z|x) || p(z)) where p(z) ~ N(prior_mu, exp(prior_logvar))
            kl = -0.5 * torch.sum(
                1 + z_logvar - self.prior_logvar.unsqueeze(0) - 
                ((z_mu - self.prior_mu.unsqueeze(0)).pow(2) + z_logvar.exp()) / self.prior_logvar.exp().unsqueeze(0),
                dim=1
            )
            
            # 应用先验强度
            kl = effective_strength * kl
        
        return kl
    
    def sample_prior(self, batch_size: int) -> torch.Tensor:
        """
        从先验分布中采样
        
        参数:
            batch_size: 采样数量
            
        返回:
            采样结果 [batch_size, z_dim]
        """
        if not self.has_prior:
            # 标准正态分布采样
            return torch.randn(batch_size, self.z_dim, device=self.device)
        else:
            # 从历史先验分布采样
            std = torch.exp(0.5 * self.prior_logvar)
            eps = torch.randn(batch_size, self.z_dim, device=self.device)
            return self.prior_mu.unsqueeze(0) + eps * std.unsqueeze(0)
    
    def get_prior_info(self) -> Dict:
        """
        获取先验信息用于日志记录
        """
        info = {
            'has_prior': self.has_prior,
            'prior_strength': self.prior_strength,
            'prior_count': float(self.prior_count) if self.has_prior else 0,
            'current_count': float(self.current_count),
            'running_count': self.running_count
        }
        
        if self.has_prior:
            info.update({
                'prior_mu_stats': {
                    'mean': float(self.prior_mu.mean()),
                    'std': float(self.prior_mu.std()),
                    'min': float(self.prior_mu.min()),
                    'max': float(self.prior_mu.max())
                },
                'prior_var_stats': {
                    'mean': float(torch.exp(self.prior_logvar).mean()),
                    'std': float(torch.exp(self.prior_logvar).std()),
                    'min': float(torch.exp(self.prior_logvar).min()),
                    'max': float(torch.exp(self.prior_logvar).max())
                }
            })
        
        return info
    
    def save_state(self) -> Dict:
        """
        保存先验状态
        """
        return {
            'has_prior': self.has_prior,
            'prior_mu': self.prior_mu.cpu(),
            'prior_logvar': self.prior_logvar.cpu(),
            'prior_count': self.prior_count.cpu(),
            'prior_strength': self.prior_strength,
            'adaptive_strength': self.adaptive_strength,
            'z_dim': self.z_dim
        }
    
    def load_state(self, state_dict: Dict):
        """
        加载先验状态
        """
        self.has_prior = state_dict['has_prior']
        self.prior_mu = state_dict['prior_mu'].to(self.device)
        self.prior_logvar = state_dict['prior_logvar'].to(self.device)
        self.prior_count = state_dict['prior_count'].to(self.device)
        self.prior_strength = state_dict['prior_strength']
        self.adaptive_strength = state_dict['adaptive_strength']
        
        # 重置当前统计
        self.reset_current_statistics()
        
        print(f"先验状态已加载: has_prior={self.has_prior}, count={float(self.prior_count)}")


class StandardGaussianPrior(nn.Module):
    """
    标准高斯先验 N(0, I)，用于对比和第一次训练
    """
    
    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim
    
    def get_kl_divergence(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        计算标准KL散度 KL(q(z|x) || N(0,I))
        """
        return -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp(), dim=1)
    
    def sample_prior(self, batch_size: int) -> torch.Tensor:
        """
        从标准正态分布采样
        """
        return torch.randn(batch_size, self.z_dim)
    
    def get_prior_info(self) -> Dict:
        """
        获取先验信息
        """
        return {'type': 'standard', 'z_dim': self.z_dim}