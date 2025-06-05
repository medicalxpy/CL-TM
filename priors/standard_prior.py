import math
import torch
import torch.nn.functional as F
from typing import Optional

def log_Normal_standard(z: torch.Tensor, dim: Optional[int] = None):
    """计算标准正态分布N(0, I)的对数概率密度。
    
    直接参考自 vae/utils/distributions.py 中的 log_Normal_standard 函数。
    
    参数:
        z: 隐变量，形状为[batch_size, latent_dim]
        dim: 沿着哪个维度求和，如果为None，不求和
        
    返回:
        log_normal: 标准正态分布的对数概率密度
    """
    log_normal = -0.5 * (math.log(2.0 * math.pi) + torch.pow(z, 2))
    if dim is not None:
        return log_normal.sum(dim=dim)
    else:
        return log_normal

def KL_scETM(mu: torch.Tensor, logsigma: torch.Tensor):
    """计算KL散度：KL(q(z|x) || p(z))，其中p(z)是标准正态分布N(0, I)，q(z|x)是N(mu, sigma)。
    
    参考自 scETM/models/model_utils.py 中的 get_kl 函数。
    
    参数:
        mu: q分布的均值，形状为[batch_size, latent_dim]
        logsigma: q分布的对数标准差，形状为[batch_size, latent_dim]
        
    返回:
        kl: KL散度，形状为[batch_size]
    """
    return -0.5 * (1 + 2 * logsigma - mu.pow(2) - torch.exp(2 * logsigma)).sum(-1)