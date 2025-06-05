# model/loss/KL.py
import math
import torch
import torch.nn.functional as F
from typing import Optional, Callable, Union, List, Tuple, Dict, Any

def gaussian_log_density(
    z: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    dim: Optional[int] = 1
) -> torch.Tensor:
    """
    计算多元高斯分布的对数密度。
    参考自vae/utils/distributions.py中的log_Normal_diag函数。
    
    参数:
        z: 点，形状为[batch_size, latent_dim]
        mu: 均值，形状为[batch_size, latent_dim]或[1, latent_dim]
        logvar: 对数方差，形状为[batch_size, latent_dim]或[1, latent_dim]
        dim: 沿哪个维度进行求和，默认为1（特征维度）
        
    返回:
        对数概率密度，形状为[batch_size]
    """
    log_density = -0.5 * (math.log(2.0 * math.pi) + logvar + 
                         torch.pow(z - mu, 2) / (torch.exp(logvar) + 1e-5))
    
    if dim is not None:
        return log_density.sum(dim=dim)
    else:
        return log_density

def reparameterize(
    mu: torch.Tensor, 
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    使用重参数化技巧从正态分布中采样。
    参考自vae/model/simple_vae.py中的reparameterize方法。
    
    参数:
        mu: 均值，形状为[batch_size, latent_dim]
        logvar: 对数方差，形状为[batch_size, latent_dim]
        
    返回:
        采样结果，形状为[batch_size, latent_dim]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# ======================== 不同先验的对数概率密度计算 ========================

def standard_logpz(
    z: torch.Tensor,
    dim: int = 1,
    **kwargs
) -> torch.Tensor:
    """
    计算标准正态分布N(0, I)的对数概率密度。
    参考自vae/model/standard.py中的log_p_z函数。
    
    参数:
        z: 隐变量，形状为[batch_size, latent_dim]
        dim: 沿哪个维度求和，默认为1
        
    返回:
        对数概率密度，形状为[batch_size]
    """
    # 直接计算标准正态分布的对数概率
    log_normal = -0.5 * (math.log(2.0 * math.pi) + torch.pow(z, 2))
    if dim is not None:
        return log_normal.sum(dim=dim)
    else:
        return log_normal

def mog_logpz(
    z: torch.Tensor,
    mog_mu: torch.Tensor,
    mog_logvar: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    learned_mu: Optional[List[torch.Tensor]] = None,
    learned_logvar: Optional[List[torch.Tensor]] = None,
    incremental: bool = False,
    num_comp: int = 1,
    **kwargs
) -> torch.Tensor:
    """
    计算高斯混合模型(MoG)先验的对数概率密度。
    参考自vae/model/mog.py中的log_p_z函数。
    
    参数:
        z: 隐变量，形状为[batch_size, latent_dim]
        mog_mu: 混合模型的均值，形状为[n_components, latent_dim]
        mog_logvar: 混合模型的对数方差，形状为[n_components, latent_dim]
        weights: 各组件的权重，形状为[n_components]
        learned_mu: 已学习组件的均值列表（用于增量学习）
        learned_logvar: 已学习组件的对数方差列表（用于增量学习）
        incremental: 是否使用增量学习
        num_comp: 混合模型的组件数
        
    返回:
        对数概率密度，形状为[batch_size]
    """
    z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
    
    # 计算当前组件的对数概率
    log_comps = gaussian_log_density(z_expand, mog_mu, mog_logvar, dim=2)  # [batch_size, n_components]
    
    # 如果使用增量学习并且有已学习的组件
    num_tsk = 0
    if incremental and learned_mu is not None and len(learned_mu) > 0:
        num_tsk = len(learned_mu)
        learned_mu_concat = torch.cat([learned_mu[i] for i in range(num_tsk)], 1)
        learned_logvar_concat = torch.cat([learned_logvar[i] for i in range(num_tsk)], 1)
        log_learned_comps = gaussian_log_density(z_expand, learned_mu_concat, learned_logvar_concat, dim=2)
        log_comps = torch.cat((log_comps, log_learned_comps), 1)
    
    # 根据权重调整对数概率
    if weights is not None:
        log_w = torch.log(weights).unsqueeze(0).to(z.device)
        log_comps = log_comps + log_w
    else:
        # 如果没有提供权重，则假设均匀分布
        log_comps = log_comps - math.log(num_comp * (1 + num_tsk))
    
    # 使用logsumexp计算混合分布的对数概率
    log_prior = torch.logsumexp(log_comps, 1)  # [batch_size]
    return log_prior

def vamp_logpz(
    z: torch.Tensor,
    encoder: Callable,
    pseudoinputs: List[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """
    计算VAE with a Mixture Prior (VampPrior)的对数概率密度。
    参考自vae/model/boost.py和vae/utils/mixture.py中的VampMixture类的log_density方法。
    
    参数:
        z: 隐变量，形状为[batch_size, latent_dim]
        encoder: 将伪输入编码到潜在空间的编码器函数
        pseudoinputs: 伪输入列表，每个形状为[1, input_dim]
        weights: 各组件的权重，如未提供则假设均匀分布
        
    返回:
        对数概率密度，形状为[batch_size]
    """
    log_probs = []
    
    # 将z扩展为与伪输入数量相符的形状
    z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
    
    all_mus = []
    all_logvars = []
    
    # 对每个伪输入计算其潜在空间表示
    for pseudoinput in pseudoinputs:
        # 编码伪输入
        with torch.no_grad():
            pseudo_mu, pseudo_logvar = encoder(pseudoinput)
        all_mus.append(pseudo_mu)
        all_logvars.append(pseudo_logvar)
    
    # 将所有均值和方差拼接成批次形式
    mus = torch.cat(all_mus, dim=0)  # [n_components, latent_dim]
    logvars = torch.cat(all_logvars, dim=0)  # [n_components, latent_dim]
    
    # 计算z在每个组件下的对数概率
    log_probs = gaussian_log_density(z_expand, mus.unsqueeze(0), logvars.unsqueeze(0), dim=2)  # [batch_size, n_components]
    
    # 应用组件权重
    if weights is not None:
        log_w = torch.log(weights).unsqueeze(0).to(z.device)
        log_probs = log_probs + log_w
    else:
        # 如果未提供权重，使用均匀分布
        log_probs = log_probs - math.log(len(pseudoinputs))
    
    # 使用logsumexp计算混合分布的对数概率
    log_prior = torch.logsumexp(log_probs, 1)  # [batch_size]
    return log_prior

# ===================== KL散度计算 =====================

def calculate_kl(
    z: torch.Tensor,
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    logpz: torch.Tensor,
    importance_samples: int = 1,
    reduce: bool = True
) -> torch.Tensor:
    """
    计算KL散度：KL(q(z|x) || p(z)) = E_q[log q(z|x) - log p(z)]
    
    参数:
        z: 从q分布采样的隐变量，形状为[batch_size, latent_dim]
        q_mu: q分布的均值，形状为[batch_size, latent_dim]
        q_logvar: q分布的对数方差，形状为[batch_size, latent_dim]
        logpz: 先验概率log p(z)，形状为[batch_size]
        importance_samples: 重要性采样的样本数
        reduce: 是否对batch维度求平均
        
    返回:
        KL散度，如果reduce=True则为标量，否则形状为[batch_size]
    """
    # 计算log q(z|x)
    logqz = gaussian_log_density(z, q_mu, q_logvar, dim=1)
    
    # 计算KL散度：log q(z|x) - log p(z)
    kl_value = logqz - logpz
    
    
    # 如果需要对batch维度求平均
    if reduce:
        return kl_value.mean()
    else:
        return kl_value

def get_kl_divergence(
    z: torch.Tensor,
    q_mu: torch.Tensor, 
    q_logvar: torch.Tensor, 
    prior_type: str = 'standard',
    prior_params: Dict[str, Any] = None,
    reduce: bool = False
) -> torch.Tensor:
    """
    统一接口计算KL散度，根据先验类型选择适当的logpz计算函数。
    
    参数:
        z: 采样的隐变量，形状为[batch_size, latent_dim]
        q_mu: q分布的均值，形状为[batch_size, latent_dim]
        q_logvar: q分布的对数方差，形状为[batch_size, latent_dim]
        prior_type: 先验类型，可选'standard'、'mog'或'vamp'
        prior_params: 各类先验所需的参数字典
        reduce: 是否对batch维度求平均
        
    返回:
        KL散度
    """
    if prior_params is None:
        prior_params = {}
    
    # 根据先验类型选择合适的logpz计算函数
    if prior_type == 'standard':
        logpz = standard_logpz(z)
    
    elif prior_type == 'mog':
        if 'mog_mu' not in prior_params or 'mog_logvar' not in prior_params:
            raise ValueError("对于MoG先验，需要指定mog_mu和mog_logvar")
        logpz = mog_logpz(z, **prior_params)
    
    elif prior_type == 'vamp':
        if 'encoder' not in prior_params or 'pseudoinputs' not in prior_params:
            raise ValueError("对于VAMP先验，需要指定encoder和pseudoinputs")
        logpz = vamp_logpz(z, **prior_params)
    
    else:
        raise ValueError(f"不支持的先验类型: {prior_type}")
    
    logqz = gaussian_log_density(z, q_mu, q_logvar, dim=1)
    
    # 计算KL散度：log q(z|x) - log p(z)
    kl_value = logqz - logpz
    
    
    # 如果需要对batch维度求平均
    if reduce:
        return kl_value.mean()
    else:
        return kl_value