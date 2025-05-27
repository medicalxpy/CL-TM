# continual_learning/pruning.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, List, Optional, Tuple, Union
import logging

from loss.KL import gaussian_log_density

_logger = logging.getLogger(__name__)


class ComponentPruner:
    """
    组件修剪器
    
    实现BooVAE的组件修剪算法，通过优化组件权重并移除冗余组件来防止模型复杂度爆炸。
    参考自: BooVAE vae/model/boost.py 的 update_component_weigts 方法
    
    主要功能:
    - 优化混合先验中组件的权重
    - 移除权重过小的组件
    - 防止模型复杂度无限增长
    - 保持有效组件的表达能力
    
    Args:
        model: 主模型
        min_weight_threshold: 组件最小权重阈值，低于此值的组件将被移除
        learning_rate: 权重优化学习率
        max_iterations: 权重优化最大迭代次数
        patience: 早停耐心值
        device: 计算设备
    """
    
    def __init__(self,
        model: nn.Module,
        min_weight_threshold: float = 1e-3,
        learning_rate: float = 0.0005,
        max_iterations: int = 500,
        patience: int = 100,
        device: torch.device = None
    ):
        self.model = model
        self.min_weight_threshold = min_weight_threshold
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.patience = patience
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        _logger.info(f"初始化ComponentPruner: 最小权重阈值={min_weight_threshold}")

    def prune_components(self, n_samples: int = 1000) -> Dict[str, Union[int, float]]:
        """
        执行组件修剪
        
        直接参考自: BooVAE vae/model/boost.py 的 update_component_weigts 方法
        核心思想是通过最小化两个分布之间的KL散度来优化组件权重
        
        Args:
            n_samples: 用于优化的样本数量
            
        Returns:
            修剪统计信息字典
        """
        if not self._has_valid_prior():
            _logger.warning("模型没有有效的混合先验，跳过组件修剪")
            return {}
            
        _logger.info("开始组件修剪...")
        
        # 获取当前任务的组件 (来自boost.py)
        original_count, current_weights, component_indices = self._get_current_task_components()
        
        if len(current_weights) <= 1:
            _logger.info("当前任务只有一个或没有组件，跳过修剪")
            return {'pruned_count': 0, 'remaining_count': len(current_weights)}
            
        # 获取组件对应的先验分布参数 (来自boost.py)
        means, logvars = self._get_component_distributions(component_indices)
        
        # 优化权重 (来自boost.py的核心优化逻辑)
        optimized_weights = self._optimize_weights(current_weights, means, logvars, n_samples)
        
        # 执行修剪 (来自boost.py的prune逻辑)
        pruning_stats = self._apply_pruning(optimized_weights, component_indices)
        
        _logger.info(f"组件修剪完成: {pruning_stats}")
        return pruning_stats

    def _has_valid_prior(self) -> bool:
        """检查模型是否有有效的混合先验"""
        return (hasattr(self.model, 'prior') and 
                hasattr(self.model.prior, 'mu_list') and
                hasattr(self.model.prior, 'weights') and
                hasattr(self.model.prior, 'task_weight') and
                len(self.model.prior.mu_list) > 0)

    def _get_current_task_components(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        获取当前任务的组件信息
        参考自: BooVAE vae/model/boost.py 中获取当前任务组件的逻辑
        """
        # 获取当前任务的组件掩码 (来自boost.py)
        current_task_mask = self.model.prior.task_weight == self.model.prior.num_tasks
        current_weights = self.model.prior.weights[current_task_mask].clone()
        component_indices = torch.where(current_task_mask)[0]
        
        original_count = len(self.model.prior.mu_list)
        
        return original_count, current_weights, component_indices

    def _get_component_distributions(self, component_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取组件对应的分布参数
        参考自: BooVAE vae/model/boost.py 中编码伪输入的逻辑
        """
        means_list = []
        logvars_list = []
        
        # 对当前任务的每个组件进行编码 (来自boost.py)
        self.model.eval()
        with torch.no_grad():
            for idx in component_indices:
                pseudo_input = self.model.prior.mu_list[idx]
                
                # 编码伪输入到潜在空间
                if hasattr(self.model, 'encoder'):
                    mu, logvar = self.model.encoder(pseudo_input)
                else:
                    fwd_dict = self.model.forward(pseudo_input)
                    mu = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
                    logvar = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
                    
                means_list.append(mu)  
                logvars_list.append(logvar)
                
        means = torch.cat(means_list, dim=0)
        logvars = torch.cat(logvars_list, dim=0)
        
        return means, logvars

    def _optimize_weights(self, 
                         initial_weights: torch.Tensor,
                         means: torch.Tensor, 
                         logvars: torch.Tensor,
                         n_samples: int) -> torch.Tensor:
        """
        优化组件权重
        
        直接参考自: BooVAE vae/model/boost.py 的权重优化循环
        通过最小化KL散度来重新分配组件权重
        
        Args:
            initial_weights: 初始权重
            means: 组件均值
            logvars: 组件对数方差  
            n_samples: 优化样本数
            
        Returns:
            优化后的权重
        """
        # 创建可优化的权重参数 (来自boost.py)
        w_new = nn.Parameter(initial_weights.clone())
        optimizer = optim.Adam([w_new], lr=self.learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=0.1)
        
        n_components = len(initial_weights)
        
        # 获取最优分布参数 (来自boost.py)
        optimal_means, optimal_logvars = self._get_optimal_distribution()
        
        # 优化循环 (完全来自boost.py的优化逻辑)
        for iteration in range(self.max_iterations):
            optimizer.zero_grad()
            
            # 确保权重非负且和为1 (来自boost.py)
            w_normalized = torch.clamp(w_new, 0, 1)
            w_normalized = w_normalized / (w_normalized.sum() + 1e-10)
            
            # 第一个KL项: KL(当前先验 || 最优先验) (来自boost.py)
            kl1 = self._compute_kl_divergence(
                means[w_normalized > 0], logvars[w_normalized > 0], w_normalized[w_normalized > 0],
                optimal_means, optimal_logvars, n_samples, forward=True
            )
            
            # 第二个KL项: KL(最优先验 || 当前先验) (来自boost.py)  
            kl2 = self._compute_kl_divergence(
                optimal_means, optimal_logvars,
                means[w_normalized > 0], logvars[w_normalized > 0], w_normalized[w_normalized > 0],
                n_samples, forward=False
            )
            
            # 总损失: 对称KL散度 (来自boost.py)
            loss = 0.5 * kl1 + 0.5 * kl2
            
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            # 更新权重约束 (来自boost.py)
            with torch.no_grad():
                w_new.data = torch.clamp(w_new.data, 0, 1)
                w_new.data = w_new.data / (w_new.data.sum() + 1e-10)
                
        return w_new.data.detach()

    def _get_optimal_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取最优分布参数
        参考自: BooVAE boosting优化器中的optimal_means和optimal_logvars
        """
        # 如果BoostingOptimizer可用，使用其最优分布
        if (hasattr(self.model, '_boosting_optimizer') and 
            hasattr(self.model._boosting_optimizer, 'optimal_means') and
            self.model._boosting_optimizer.optimal_means is not None):
            return (self.model._boosting_optimizer.optimal_means, 
                   self.model._boosting_optimizer.optimal_logvars)
        
        # 否则，使用当前数据的经验分布作为近似
        _logger.warning("没有找到最优分布，使用标准正态分布作为近似")
        latent_dim = means.shape[-1] if len(means.shape) > 1 else 1
        return (torch.zeros(1, latent_dim, device=self.device),
                torch.zeros(1, latent_dim, device=self.device))

    def _compute_kl_divergence(self,
                              source_means: torch.Tensor,
                              source_logvars: torch.Tensor, 
                              source_weights: Optional[torch.Tensor],
                              target_means: torch.Tensor,
                              target_logvars: torch.Tensor,
                              n_samples: int,
                              forward: bool = True) -> torch.Tensor:
        """
        计算两个混合高斯分布之间的KL散度
        
        参考自: BooVAE vae/model/boost.py 中KL散度计算的核心逻辑
        使用蒙特卡洛采样估计KL散度
        
        Args:
            source_means: 源分布均值
            source_logvars: 源分布对数方差
            source_weights: 源分布权重
            target_means: 目标分布均值  
            target_logvars: 目标分布对数方差
            n_samples: 采样数量
            forward: 是否是前向KL
            
        Returns:
            KL散度值
        """
        if source_weights is None:
            source_weights = torch.ones(len(source_means), device=self.device) / len(source_means)
        target_weights = torch.ones(len(target_means), device=self.device) / len(target_means)
        
        # 从源分布采样 (来自boost.py)
        if forward:
            # 根据权重选择组件进行采样
            component_choices = torch.multinomial(source_weights, n_samples, replacement=True)
            z_samples = []
            
            for i in range(n_samples):
                comp_idx = component_choices[i]
                mu = source_means[comp_idx]
                logvar = source_logvars[comp_idx]
                
                # 重参数化采样
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_sample = mu + eps * std
                z_samples.append(z_sample)
                
            z_samples = torch.stack(z_samples)
        else:
            # 从目标分布采样
            component_choices = torch.randint(0, len(target_means), (n_samples,))
            z_samples = []
            
            for i in range(n_samples):
                comp_idx = component_choices[i]
                mu = target_means[comp_idx]
                logvar = target_logvars[comp_idx]
                
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z_sample = mu + eps * std
                z_samples.append(z_sample)
                
            z_samples = torch.stack(z_samples)
        
        # 计算对数概率 (来自boost.py)
        log_p_source = self._mixture_log_prob(z_samples, source_means, source_logvars, source_weights)
        log_p_target = self._mixture_log_prob(z_samples, target_means, target_logvars, target_weights)
        
        # KL散度估计 (来自boost.py)
        kl_estimate = (log_p_source - log_p_target).mean()
        
        return kl_estimate

    def _mixture_log_prob(self, z: torch.Tensor, means: torch.Tensor, 
                         logvars: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        计算混合高斯分布的对数概率
        参考自: BoostingOptimizer中的_mixture_log_prob方法
        """
        z_expand = z.unsqueeze(1)  # [n_samples, 1, latent_dim]
        means_expand = means.unsqueeze(0)  # [1, n_components, latent_dim]
        logvars_expand = logvars.unsqueeze(0)  # [1, n_components, latent_dim]
        
        # 计算各组件的对数概率
        log_comps = gaussian_log_density(z_expand, means_expand, logvars_expand, dim=2)
        
        # 应用权重
        log_w = torch.log(weights + 1e-10).unsqueeze(0).to(z.device)
        log_probs = log_comps + log_w
        
        # 混合分布的对数概率
        return torch.logsumexp(log_probs, dim=1)

    def _apply_pruning(self, optimized_weights: torch.Tensor, 
                      component_indices: torch.Tensor) -> Dict[str, Union[int, float]]:
        """
        应用修剪，移除权重过小的组件
        
        直接参考自: BooVAE vae/utils/mixture.py 的 prune 方法
        移除权重低于阈值的组件并重新组织先验结构
        
        Args:
            optimized_weights: 优化后的权重
            component_indices: 组件索引
            
        Returns:
            修剪统计信息
        """
        # 找出要保留的组件 (来自mixture.py的prune)
        keep_mask = optimized_weights > self.min_weight_threshold
        n_pruned = len(optimized_weights) - keep_mask.sum().item()
        
        if n_pruned == 0:
            return {
                'pruned_count': 0,
                'remaining_count': len(optimized_weights),
                'min_weight': optimized_weights.min().item(),
                'max_weight': optimized_weights.max().item()
            }
        
        # 更新权重 (来自mixture.py)
        kept_weights = optimized_weights[keep_mask]
        kept_indices = component_indices[keep_mask]
        
        # 重新归一化权重
        kept_weights = kept_weights / kept_weights.sum()
        
        # 更新先验结构 (来自mixture.py的prune逻辑)
        self._update_prior_structure(kept_weights, kept_indices, component_indices)
        
        return {
            'pruned_count': n_pruned,
            'remaining_count': len(kept_weights),
            'min_weight': kept_weights.min().item(),
            'max_weight': kept_weights.max().item(),
            'total_components': len(self.model.prior.mu_list)
        }

    def _update_prior_structure(self, kept_weights: torch.Tensor, 
                               kept_indices: torch.Tensor,
                               original_indices: torch.Tensor):
        """
        更新先验结构，移除被修剪的组件
        参考自: BooVAE vae/utils/mixture.py 的prune方法中的结构更新逻辑
        """
        # 获取非当前任务的组件
        non_current_mask = self.model.prior.task_weight != self.model.prior.num_tasks
        non_current_weights = self.model.prior.weights[non_current_mask]
        non_current_task_weights = self.model.prior.task_weight[non_current_mask]
        
        # 合并保留的当前任务组件和非当前任务组件
        new_weights = torch.cat([non_current_weights.cpu(), kept_weights.cpu()])
        new_task_weights = torch.cat([
            non_current_task_weights.cpu(), 
            self.model.prior.task_weight[original_indices[kept_weights > 0]].cpu()
        ])
        
        # 更新权重和任务权重
        self.model.prior.weights = new_weights.to(self.device)  
        self.model.prior.task_weight = new_task_weights.to(self.device)
        
        # 更新伪输入列表 (来自mixture.py)
        # 保留非当前任务的组件
        total_components = len(self.model.prior.mu_list)
        new_mu_list = []
        
        # 添加非当前任务的组件
        for i in range(total_components):
            if i < len(original_indices) and i not in original_indices:
                new_mu_list.append(self.model.prior.mu_list[i])
        
        # 添加保留的当前任务组件  
        for idx in kept_indices:
            new_mu_list.append(self.model.prior.mu_list[idx])
            
        self.model.prior.mu_list = new_mu_list
        self.model.prior.num_comp = len(new_mu_list)
        
        _logger.info(f"更新先验结构: 总组件数 {len(new_mu_list)}, 当前任务组件数 {len(kept_weights)}")

    def get_pruning_stats(self) -> Dict[str, Union[int, float]]:
        """获取当前的修剪统计信息"""
        if not self._has_valid_prior():
            return {}
            
        total_components = len(self.model.prior.mu_list)
        current_task_mask = self.model.prior.task_weight == self.model.prior.num_tasks
        current_task_components = current_task_mask.sum().item()
        
        weights = self.model.prior.weights
        min_weight = weights.min().item() if len(weights) > 0 else 0.0
        max_weight = weights.max().item() if len(weights) > 0 else 0.0
        
        return {
            'total_components': total_components,
            'current_task_components': current_task_components,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'weight_below_threshold': (weights < self.min_weight_threshold).sum().item()
        }

    def should_prune(self) -> bool:
        """
        判断是否需要进行组件修剪
        基于组件数量和权重分布进行判断
        """
        if not self._has_valid_prior():
            return False
            
        stats = self.get_pruning_stats()
        
        # 如果有权重过小的组件，建议修剪
        if stats.get('weight_below_threshold', 0) > 0:
            return True
            
        # 如果当前任务组件数过多，建议修剪  
        if stats.get('current_task_components', 0) > 10:
            return True
            
        return False