# model/loss/RL.py
import math
import torch
import torch.nn.functional as F
import torch.distributions as D
from typing import Optional, Union, Literal, Dict, Any

# 定义最小和最大误差值，用于裁剪概率值，防止计算对数时出现数值问题
MIN_EPSILON = 1e-5
MAX_EPSILON = 1. - 1e-5

def bernoulli_loss(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    dim: Optional[int] = 1, 
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算伯努利分布的负对数似然损失，适用于二元数据。
    参考自vae/utils/distributions.py中的log_Bernoulli函数。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        mean: 重构的均值，形状为[batch_size, feature_dim]
        dim: 沿哪个维度进行求和，默认为1（特征维度）
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        负对数似然损失
    """
    # 裁剪概率值，避免出现0和1，这会导致对数运算时出现问题
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    # 计算伯努利分布的负对数似然
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    
    # 沿特定维度求和
    if dim is not None:
        loss = -log_bernoulli.sum(dim=dim)
    else:
        loss = -log_bernoulli
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def gaussian_loss(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    logvar: torch.Tensor, 
    dim: Optional[int] = 1, 
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算高斯分布的负对数似然损失，适用于连续数据。
    参考自vae/utils/distributions.py中的log_Normal_diag函数。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        mean: 重构的均值，形状为[batch_size, feature_dim]
        logvar: 对数方差，形状为[batch_size, feature_dim]
        dim: 沿哪个维度进行求和，默认为1（特征维度）
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        负对数似然损失
    """
    # 计算高斯分布的负对数似然
    log_normal = -0.5 * (math.log(2.0 * math.pi) + logvar + 
                         torch.pow(x - mean, 2) / (torch.exp(logvar) + 1e-5))
    
    # 沿特定维度求和
    if dim is not None:
        loss = -log_normal.sum(dim=dim)
    else:
        loss = -log_normal
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def logistic_loss(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    logvar: torch.Tensor, 
    dim: Optional[int] = 1, 
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算Logistic分布的负对数似然损失，适用于256色图像等数据。
    参考自vae/utils/distributions.py中的log_Logistic_256函数。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        mean: 重构的均值，形状为[batch_size, feature_dim]
        logvar: 对数方差，形状为[batch_size, feature_dim]
        dim: 沿哪个维度进行求和，默认为1（特征维度）
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        负对数似然损失
    """
    bin_size = 1. / 256.

    # 计算Logistic CDF
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size / scale)
    cdf_minus = torch.sigmoid(x)

    # 计算log-likelihood
    log_logistic = torch.log(cdf_plus - cdf_minus + 1.e-7)
    
    # 沿特定维度求和
    if dim is not None:
        loss = -log_logistic.sum(dim=dim)
    else:
        loss = -log_logistic
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def negative_binomial_loss(
    x: torch.Tensor, 
    total_count: torch.Tensor, 
    logits: torch.Tensor, 
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    计算负二项分布的负对数似然损失，适用于计数数据。
    参考自scETM/models/scVI.py中的get_reconstruction_loss方法。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        total_count: 负二项分布的总计数参数，形状为[batch_size, feature_dim]
        logits: 负二项分布的logits参数，形状为[batch_size, feature_dim]
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        负对数似然损失
    """
    # 使用PyTorch的负二项分布计算负对数似然
    dist = D.NegativeBinomial(total_count=total_count, logits=logits)
    nb = D.Independent(dist, 1)
    loss = -nb.log_prob(x)
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def l1_loss(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算L1损失（绝对误差），适用于某些连续数据。
    参考自vae/model/simple_vae.py中的NLL方法。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        mean: 重构的均值，形状为[batch_size, feature_dim]
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        L1损失
    """
    loss = torch.abs(x - mean).sum(1)
    
    # 应用reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss

def mse_loss(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    reduction: str = 'none'
) -> torch.Tensor:
    """
    计算均方误差损失，适用于连续数据。
    参考自scETM/models/scVI.py中的get_reconstruction_loss方法。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        mean: 重构的均值，形状为[batch_size, feature_dim]
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        MSE损失
    """
    if reduction == 'none':
        return F.mse_loss(mean, x, reduction='none').sum(-1)
    return F.mse_loss(mean, x, reduction=reduction)

def get_reconstruction_loss(
    x: torch.Tensor,
    recon_x: torch.Tensor,
    recon_params: Dict[str, Any] = None,
    loss_type: Literal['bernoulli', 'gaussian', 'logistic', 'nb', 'l1', 'mse'] = 'bernoulli',
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    根据指定的损失类型计算重构损失。
    提供统一接口以支持不同类型的重构损失。
    
    参数:
        x: 输入数据，形状为[batch_size, feature_dim]
        recon_x: 重构的数据，通常是均值，形状为[batch_size, feature_dim]
        recon_params: 重构所需的额外参数，如logvar, total_count等
        loss_type: 损失类型
        reduction: 如何处理损失，可选'none', 'sum', 'mean'
        
    返回:
        重构损失
    """
    if recon_params is None:
        recon_params = {}
        
    if loss_type == 'bernoulli':
        return bernoulli_loss(x, recon_x, reduction=reduction)
    elif loss_type == 'gaussian':
        logvar = recon_params.get('logvar', torch.zeros_like(recon_x))
        return gaussian_loss(x, recon_x, logvar, reduction=reduction)
    elif loss_type == 'logistic':
        logvar = recon_params.get('logvar', torch.zeros_like(recon_x))
        return logistic_loss(x, recon_x, logvar, reduction=reduction)
    elif loss_type == 'nb':
        total_count = recon_params.get('total_count')
        logits = recon_params.get('logits')
        if total_count is None or logits is None:
            raise ValueError("总计数和logits是负二项分布损失所必需的")
        return negative_binomial_loss(x, total_count, logits, reduction=reduction)
    elif loss_type == 'l1':
        return l1_loss(x, recon_x, reduction=reduction)
    elif loss_type == 'mse':
        return mse_loss(x, recon_x, reduction=reduction)
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")