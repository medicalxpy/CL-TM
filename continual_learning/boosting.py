# continual_learning/boosting.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging

# 导入先验相关
from priors.mixture_prior import VampMixture
from priors.standard_prior import log_Normal_standard
from loss.KL import gaussian_log_density

_logger = logging.getLogger(__name__)


class BoostingOptimizer:
    """
    BooVAE核心Boosting优化器
    
    实现BooVAE的核心防遗忘机制，通过逐步添加组件到混合先验中实现持续学习。
    参考自: BooVAE vae/model/boost.py 的核心boosting算法
    
    主要功能:
    - 添加新组件到混合先验
    - 训练单个组件防止遗忘
    - 计算最优组件权重
    - 管理组件生命周期
    
    Args:
        model: 主模型(如CL_scETM)
        n_components: 最大组件数量
        pseudoinput_mean: 伪输入均值
        pseudoinput_std: 伪输入标准差
        component_weight_type: 组件权重类型 ('equal', 'fixed', 'grad')
        device: 计算设备
    """
    
    def __init__(self,
        model: nn.Module,
        n_components: int = 500,
        pseudoinput_mean: float = 0.0,
        pseudoinput_std: float = 0.1,
        component_weight_type: str = 'grad',
        device: torch.device = None
    ):
        self.model = model
        self.n_components = n_components
        self.pseudoinput_mean = pseudoinput_mean
        self.pseudoinput_std = pseudoinput_std
        self.component_weight_type = component_weight_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 当前组件计数
        self.current_components = 0
        
        # 存储最优分布参数 (来自boost.py的mean_opt, logvar_opt)
        self.optimal_means = None
        self.optimal_logvars = None
        
        _logger.info(f"初始化BoostingOptimizer: 最大组件数={n_components}, 权重类型={component_weight_type}")

    def add_component(self, 
                     X_opt: torch.Tensor,
                     max_steps: int = 30000,
                     lambda_coeff: float = 1.0,
                     learning_rate: float = 3e-3,
                     from_prev: bool = False,
                     from_input: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        添加新组件到混合先验中
        
        直接参考自: BooVAE vae/model/boost.py 的 add_component 方法
        
        Args:
            X_opt: 用于优化的数据样本
            max_steps: 最大训练步数
            lambda_coeff: lambda系数
            learning_rate: 学习率
            from_prev: 是否从之前的组件初始化
            from_input: 从指定输入初始化
            
        Returns:
            训练历史字典
        """
        if self.current_components >= self.n_components:
            _logger.warning(f"已达到最大组件数 {self.n_components}")
            return {}
            
        _logger.info(f"添加第 {self.current_components + 1} 个组件")
        
        # 如果是第一个组件，初始化先验 (来自boost.py的init_prior逻辑)
        if self.current_components == 0:
            self._init_prior(X_opt)
            
        # 计算最优分布 (来自boost.py)
        with torch.no_grad():
            self.optimal_means, self.optimal_logvars = self._compute_optimal_distribution(X_opt)
            
        # 尝试训练组件，直到获得有效权重
        weight, iteration = 0, 0
        while weight == 0 and iteration < 3:
            iteration += 1
            
            # 重置伪输入参数 (来自boost.py的reset_parameters)
            if from_prev:
                self._reset_from_previous()
            elif from_input is not None:
                self._reset_from_input(from_input)
            else:
                self._reset_parameters()
                
            # 训练组件 (来自boost.py的train_component)
            history = self._train_component(max_steps, lambda_coeff, learning_rate)
            
            # 计算组件权重 (来自boost.py)
            if self.component_weight_type == 'fixed':
                weight = 0.5
            elif self.component_weight_type == 'grad':
                weight = self._get_optimal_alpha()
            else:  # 'equal'
                weight = None
                
        _logger.info(f'训练完成，组件权重: {weight}')
        
        # 添加组件到先验 (来自boost.py)
        self._add_to_prior(weight)
        self.current_components += 1
        
        return history

    def _init_prior(self, X_opt: torch.Tensor):
        """
        初始化混合先验
        参考自: BooVAE vae/model/boost.py 的 init_prior 方法
        """
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'add_component'):
            # 使用数据的平均值作为第一个伪输入
            mean_opt = X_opt.mean(0, keepdim=True)
            self.model.prior.add_component(mean_opt, alpha=1.0)
        else:
            _logger.warning("模型没有合适的先验组件，无法初始化")

    def _compute_optimal_distribution(self, X_opt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算最优分布的均值和方差
        参考自: BooVAE vae/model/boost.py 中的 mean_opt, logvar_opt 计算
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'encoder'):
                mu, logvar = self.model.encoder(X_opt)
            else:
                # 如果模型结构不同，尝试其他方式
                fwd_dict = self.model.forward(X_opt)
                mu = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
                logvar = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
                
        return mu, logvar

    def _reset_parameters(self):
        """
        重置伪输入参数
        参考自: BooVAE vae/model/boost.py 的 reset_parameters 方法
        """
        if hasattr(self.model, 'h_mu'):
            self.model.h_mu.data.normal_(self.pseudoinput_mean, self.pseudoinput_std)

    def _reset_from_previous(self):
        """从之前的组件重置参数"""
        # 这里可以实现从之前组件采样的逻辑
        self._reset_parameters()

    def _reset_from_input(self, input_tensor: torch.Tensor):
        """从指定输入重置参数"""
        if hasattr(self.model, 'h_mu'):
            noise = torch.randn_like(input_tensor) * 0.05
            self.model.h_mu.data = input_tensor.reshape_as(self.model.h_mu) + noise

    def _train_component(self, max_steps: int, lambda_coeff: float, learning_rate: float) -> Dict[str, float]:
        """
        训练单个组件
        
        直接参考自: BooVAE vae/model/boost.py 的 train_component 方法
        保持完整的boosting优化逻辑
        
        Args:
            max_steps: 最大训练步数
            lambda_coeff: lambda系数
            learning_rate: 学习率
            
        Returns:
            训练历史
        """
        history = {
            'train_loss_boost': 0,
            'entropy': 0, 
            'log_mean_q': 0,
            'log_p_z': 0
        }
        
        # 创建优化器 (来自boost.py)
        if hasattr(self.model, 'h_mu'):
            optimizer = optim.Adam([self.model.h_mu], lr=learning_rate)
        else:
            _logger.warning("模型没有h_mu参数，无法进行boosting优化")
            return history
            
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.5)
        
        loss_history = [1e10]
        mu_prev = None
        
        # 获取当前先验参数 (来自boost.py)
        with torch.no_grad():
            means, logvars, weights = self._get_current_prior(current_task=True)
            
        # 训练循环 (完全来自boost.py的训练逻辑)
        for step in range(1, max_steps + 1):
            optimizer.zero_grad()
            
            # 计算boosting损失 (来自boost.py的calculate_boosting_loss)
            loss, entropy, log_mean_q, log_p_z = self._calculate_boosting_loss(
                means, logvars, weights, lambda_coeff
            )
            
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()
            
            # 累积统计
            history['train_loss_boost'] += loss.item()
            history['entropy'] += entropy.item()
            history['log_mean_q'] += log_mean_q.item()
            history['log_p_z'] += log_p_z.item()
            
            # 早停检查 (来自boost.py)
            if np.abs(loss_history[-1] - loss_history[-2]) < 1e-2 and step > 2000:
                _logger.info(f'组件在 {step} 步后收敛')
                break
                
            scheduler.step(loss)
            
        # 计算平均值
        for key in history:
            history[key] /= step
            
        return history

    def _calculate_boosting_loss(self, 
                                means: torch.Tensor, 
                                logvars: torch.Tensor, 
                                weights: torch.Tensor,
                                lambda_coeff: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算boosting损失
        
        直接参考自: BooVAE vae/model/boost.py 的 calculate_boosting_loss 方法
        这是BooVAE防遗忘的核心数学机制
        
        Args:
            means: 先验均值
            logvars: 先验对数方差
            weights: 先验权重
            lambda_coeff: lambda系数
            
        Returns:
            (总损失, 熵, log_mean_q, log_p_z)
        """
        self.model.eval()
        
        # 获取当前伪输入的编码 (来自boost.py)
        if hasattr(self.model, 'h_mu') and hasattr(self.model, 'pseudo_prep'):
            h_input = self.model.pseudo_prep(self.model.h_mu)
        elif hasattr(self.model, 'h_mu'):
            h_input = self.model.h_mu
        else:
            raise AttributeError("模型缺少必要的伪输入参数")
            
        # 编码伪输入
        if hasattr(self.model, 'encoder'):
            z_q_mean, z_q_logvar = self.model.encoder(h_input)
        else:
            fwd_dict = self.model.forward(h_input)
            z_q_mean = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
            z_q_logvar = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
            
        # 重参数化采样
        z_sample = self._reparameterize(z_q_mean, z_q_logvar)
        
        # 计算熵 (来自boost.py: -log h)
        entropy = 0.5 * (1 + math.log(2 * math.pi) + z_q_logvar).sum()
        
        # 计算log q (来自boost.py: log_mean_q)
        log_mean_q = self._optimal_prior_log_prob(z_sample)
        
        # 计算log p (来自boost.py: log_p_z)
        log_p_z = self._mixture_log_prob(z_sample, means, logvars, weights)
        
        # 总损失 (来自boost.py)
        loss = -entropy - lambda_coeff * log_mean_q + lambda_coeff * log_p_z
        
        return loss, entropy, log_mean_q, log_p_z

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _optimal_prior_log_prob(self, z_sample: torch.Tensor) -> torch.Tensor:
        """
        计算最优先验的对数概率
        参考自: BooVAE vae/model/boost.py 的 opt_prior 方法
        """
        if self.optimal_means is None or self.optimal_logvars is None:
            return torch.tensor(0.0, device=z_sample.device)
            
        # 计算混合高斯的对数概率
        c = self.optimal_means.shape[0]
        weights = torch.ones(c, device=z_sample.device) / c
        
        return self._mixture_log_prob(z_sample, self.optimal_means, self.optimal_logvars, weights)

    def _mixture_log_prob(self, z: torch.Tensor, means: torch.Tensor, 
                         logvars: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        计算混合高斯分布的对数概率
        参考自: BooVAE vae/model/boost.py 的 log_gaus_mixture 方法
        """
        z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expand = means.unsqueeze(0)  # [1, n_components, latent_dim]
        logvars_expand = logvars.unsqueeze(0)  # [1, n_components, latent_dim]
        
        # 计算各组件的对数概率 (来自boost.py)
        log_comps = gaussian_log_density(z_expand, means_expand, logvars_expand, dim=2)
        
        # 应用权重 (来自boost.py)
        log_w = torch.log(weights).unsqueeze(0).to(z.device)
        log_probs = log_comps + log_w
        
        # 混合分布的对数概率 (来自boost.py)
        return torch.logsumexp(log_probs, dim=1)

    def _get_current_prior(self, current_task: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取当前先验参数
        参考自: BooVAE vae/model/boost.py 的 get_current_prior 方法
        """
        if not hasattr(self.model, 'prior') or not hasattr(self.model.prior, 'mu_list'):
            # 如果没有混合先验，返回标准先验
            return torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1)
            
        # 获取伪输入列表
        pseudo_inputs = self.model.prior.mu_list
        if not pseudo_inputs:
            return torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1)
            
        # 编码所有伪输入
        means_list = []
        logvars_list = []
        
        for pseudo_input in pseudo_inputs:
            with torch.no_grad():
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
        
        # 获取权重
        if hasattr(self.model.prior, 'weights'):
            weights = self.model.prior.weights.to(self.device)
            if current_task and hasattr(self.model.prior, 'task_weight'):
                # 只返回当前任务的组件
                current_mask = self.model.prior.task_weight == self.model.prior.num_tasks
                weights = weights[current_mask]
                means = means[current_mask]
                logvars = logvars[current_mask]
        else:
            weights = torch.ones(len(means), device=self.device) / len(means)
            
        return means, logvars, weights

    def _get_optimal_alpha(self, max_iter: int = int(1e4), tol: float = 1e-4, lr: float = 5e-1) -> float:
        """
        获取最优alpha权重
        
        直接参考自: BooVAE vae/model/boost.py 的 get_opt_alpha 方法
        使用梯度上升法优化组件权重
        
        Args:
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            lr: 学习率
            
        Returns:
            最优权重值
        """
        alpha = torch.tensor(0.5, requires_grad=True)
        trace_alpha = []
        
        for i in range(max_iter):
            # 计算梯度 (来自boost.py的alpha_grad)
            grad = self._alpha_gradient(alpha)
            
            # 梯度上升步骤
            with torch.no_grad():
                alpha -= (lr / (i + 1.0)) * grad
                alpha.clamp_(1e-4, 1.0)
                
            trace_alpha.append(alpha.item())
            
            # 收敛检查
            if i > 20 and abs(trace_alpha[-1] - trace_alpha[-2]) <= tol:
                break
                
        return alpha.detach().item()

    def _alpha_gradient(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        计算alpha的梯度
        参考自: BooVAE vae/model/boost.py 的 alpha_grad 方法  
        """
        self.model.eval()
        
        with torch.no_grad():
            # 获取h的采样 (来自boost.py)
            if hasattr(self.model, 'h_mu') and hasattr(self.model, 'pseudo_prep'):
                h_input = self.model.pseudo_prep(self.model.h_mu)
            elif hasattr(self.model, 'h_mu'):
                h_input = self.model.h_mu
            else:
                return torch.tensor(0.0)
                
            if hasattr(self.model, 'encoder'):
                z_q_mean, z_q_logvar = self.model.encoder(h_input)
            else:
                fwd_dict = self.model.forward(h_input)
                z_q_mean = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
                z_q_logvar = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
                
            h_sample = self._reparameterize(z_q_mean, z_q_logvar)
            
            # 从先验采样 (来自boost.py)
            N = 10
            if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'mu_list'):
                c = len(self.model.prior.mu_list)
                if c > 0:
                    weights = self.model.prior.weights / self.model.prior.task_weight[-1]
                    ids = np.random.choice(c, size=N, replace=True, p=weights.cpu().numpy())
                    
                    x_samples = torch.cat([self.model.prior.mu_list[i] for i in ids])
                    if hasattr(self.model, 'encoder'):
                        p_z_mean, p_z_logvar = self.model.encoder(x_samples)
                    else:
                        p_fwd = self.model.forward(x_samples)
                        p_z_mean = p_fwd.get('mu_q_delta', p_fwd.get('mu'))
                        p_z_logvar = p_fwd.get('logsigma_q_delta', p_fwd.get('logvar'))
                        
                    p_sample = self._reparameterize(p_z_mean, p_z_logvar)
                else:
                    p_sample = torch.randn_like(h_sample)
            else:
                p_sample = torch.randn_like(h_sample)
                
            # 计算梯度 (来自boost.py的grad_weight)
            grad_h = self._gradient_weight(h_sample, alpha)
            grad_p = self._gradient_weight(p_sample, alpha)
            
        return grad_h - grad_p

    def _gradient_weight(self, z_sample: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        计算权重梯度
        参考自: BooVAE vae/model/boost.py 的 grad_weight 方法
        """
        with torch.no_grad():
            # 计算log_q_z (来自boost.py)
            log_q_z = self._optimal_prior_log_prob(z_sample).mean(0)
            
            # 计算log_p_z (来自boost.py) 
            means, logvars, weights = self._get_current_prior(current_task=True)
            log_p_z = self._mixture_log_prob(z_sample, means, logvars, weights).mean(0)
            
            # 计算log_h_z (来自boost.py)
            if hasattr(self.model, 'h_mu') and hasattr(self.model, 'pseudo_prep'):
                h_input = self.model.pseudo_prep(self.model.h_mu)
            elif hasattr(self.model, 'h_mu'):
                h_input = self.model.h_mu
            else:
                return torch.tensor(0.0)
                
            if hasattr(self.model, 'encoder'):
                h_z_mean, h_z_logvar = self.model.encoder(h_input)
            else:
                h_fwd = self.model.forward(h_input)
                h_z_mean = h_fwd.get('mu_q_delta', h_fwd.get('mu'))
                h_z_logvar = h_fwd.get('logsigma_q_delta', h_fwd.get('logvar'))
                
            log_h_z = gaussian_log_density(z_sample, h_z_mean, h_z_logvar, dim=1).mean(0)
            
        # 组合对数概率 (来自boost.py)
        log_h_z += torch.log(alpha)
        log_p_z += torch.log(1.0 - alpha)
        
        comb_log_p = torch.logsumexp(torch.stack([log_p_z, log_h_z]), 0)
        
        return comb_log_p - log_q_z

    def _add_to_prior(self, weight: Optional[float]):
        """
        将训练好的组件添加到先验中
        参考自: BooVAE vae/model/boost.py 的组件添加逻辑
        """
        if not hasattr(self.model, 'prior'):
            _logger.warning("模型没有混合先验，无法添加组件")
            return
            
        # 获取当前伪输入
        if hasattr(self.model, 'h_mu') and hasattr(self.model, 'pseudo_prep'):
            pseudo_input = self.model.pseudo_prep(self.model.h_mu).clone().detach()
        elif hasattr(self.model, 'h_mu'):
            pseudo_input = self.model.h_mu.clone().detach()
        else:
            _logger.warning("模型没有伪输入参数，无法添加组件")
            return
            
        # 添加到先验
        self.model.prior.add_component(pseudo_input, alpha=weight)
        _logger.info(f"成功添加组件到先验，当前组件数: {len(self.model.prior.mu_list)}")

    def get_component_count(self) -> int:
        """获取当前组件数量"""
        return self.current_components
        
    def is_full(self) -> bool:
        """检查是否已达到最大组件数"""
        return self.current_components >= self.n_components