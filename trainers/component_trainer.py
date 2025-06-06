# component_trainer.py
"""
BooVAE风格的组件训练器
从BooVAE代码库中提取的组件训练逻辑，用于实现incremental topic model
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any

_logger = logging.getLogger(__name__)


class ComponentTrainer:
    """
    BooVAE风格的组件训练器
    
    提取自vae/model/boost.py中的组件训练逻辑，
    用于在topic model中实现incremental learning
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 pseudoinputs_mean: float = 0.0,
                 pseudoinputs_std: float = 0.1):
        """
        初始化组件训练器
        
        参数:
            model: 要训练的模型（需要有encoder方法）
            device: 训练设备
            pseudoinputs_mean: 伪输入初始化均值
            pseudoinputs_std: 伪输入初始化标准差
        """
        self.model = model
        self.device = device
        self.pseudoinputs_mean = pseudoinputs_mean
        self.pseudoinputs_std = pseudoinputs_std
        
        # 用于训练的临时伪输入参数
        self.h_mu = None
        
        # 缓存的最优先验参数
        self._cached_optimal_means = None
        self._cached_optimal_logvars = None
        self._cached_optimal_weights = None

    def train_new_component(self, 
                           X_opt: torch.Tensor,
                           max_steps: int = 30000,
                           lbd: float = 1.0,
                           lr: float = 0.003,
                           patience: int = 100,
                           tolerance: float = 1e-2,
                           min_steps: int = 2000) -> float:
        """
        训练新的伪输入组件
        参考自vae/model/boost.py中的train_component和add_component方法
        
        参数:
            X_opt: 当前task的优化数据
            max_steps: 最大训练步数
            lbd: BooVAE的lambda参数
            lr: 学习率
            patience: 学习率调度的patience
            tolerance: 收敛容忍度
            min_steps: 最小训练步数
            
        返回:
            训练完成后的组件权重
        """
        print("开始BooVAE风格的组件训练...")
        
        # 缓存最优先验（基于当前数据）
        self._cache_optimal_prior(X_opt)
        
        # 初始化新的伪输入参数
        self._reset_component_parameters()
        
        # 设置优化器和调度器
        h_optimizer = optim.Adam([self.h_mu], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            h_optimizer, patience=patience, factor=0.5
        )
        
        # 训练参数
        loss_hist = [1e10]
        
        # 获取当前先验参数
        current_means, current_logvars, current_weights = self._get_current_prior_params()
        
        print(f"开始训练，最大步数: {max_steps}")
        
        # BooVAE训练循环（参考自vae/model/boost.py中的train_component方法）
        for boost_ep in range(1, max_steps + 1):
            h_optimizer.zero_grad()
            
            # 计算BooVAE的boosting损失
            loss, entropy, log_mean_q, log_p_z = self._calculate_boosting_loss(
                current_means, current_logvars, current_weights, lbd
            )
            
            loss.backward()
            loss_hist.append(loss.item())
            h_optimizer.step()
            scheduler.step(loss)
            
            # 检查收敛条件（参考自vae/model/boost.py）
            if (abs(loss_hist[-1] - loss_hist[-2]) < tolerance and 
                boost_ep > min_steps):
                print(f'组件训练完成，共 {boost_ep} 步')
                break
            
            if boost_ep % 1000 == 0:
                print(f"Step {boost_ep}, Loss: {loss.item():.6f}, "
                      f"Entropy: {entropy.item():.4f}, "
                      f"LogMeanQ: {log_mean_q.item():.4f}, "
                      f"LogPZ: {log_p_z.item():.4f}")
        
        # 计算最优权重（参考自vae/model/boost.py中的get_opt_alpha方法）
        optimal_weight = self._compute_optimal_weight()
        
        print(f"组件训练完成，最优权重: {optimal_weight:.6f}")
        
        return optimal_weight

    def update_existing_component_weights(self, 
                                        X_opt: torch.Tensor,
                                        n_steps: int = 500,
                                        lr: float = 0.0005) -> None:
        """
        更新现有组件权重
        参考自vae/model/boost.py中的update_component_weigts方法
        
        参数:
            X_opt: 优化数据
            n_steps: 优化步数
            lr: 学习率
        """
        if not hasattr(self.model.prior, 'weights') or len(self.model.prior.weights) <= 1:
            print("没有足够的组件需要更新权重")
            return
            
        print('开始更新现有组件权重...')
        
        # 获取当前任务的组件
        curr_task = self.model.prior.task_weight == self.model.prior.num_tasks
        w = self.model.prior.weights[curr_task].clone().requires_grad_(True)
        
        # 获取对应的伪输入
        ps_indices = torch.where(curr_task)[0]
        ps = torch.cat([self.model.prior.mu_list[i] for i in ps_indices])
        
        # 编码伪输入
        with torch.no_grad():
            mean_pr, logvar_pr = self.model.encoder(ps)
        
        # 设置优化器
        w_optimizer = optim.Adam([w], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(w_optimizer, patience=100, factor=0.1)
        
        # 缓存最优先验
        self._cache_optimal_prior(X_opt)
        
        # 权重优化循环
        for it in range(n_steps):
            w_optimizer.zero_grad()
            
            # 计算KL散度损失（参考自vae/model/boost.py）
            kl_loss = self._compute_weight_optimization_loss(
                mean_pr, logvar_pr, w, n_samples=1000
            )
            
            kl_loss.backward()
            w_optimizer.step()
            scheduler.step(kl_loss)
            
            # 确保权重有效
            with torch.no_grad():
                w.data = torch.clamp(w.data, 0, 1)
                w.data = w.data / (w.data.sum() + 1e-10)
            
            if it % 100 == 0:
                print(f"权重优化步骤 {it}, 损失: {kl_loss.item():.6f}")
        
        # 更新模型中的权重
        self.model.prior.weights[curr_task] = w.data
        print(f"权重更新完成: {w.data}")

    def accept_new_component_with_weight(self, weight: float) -> None:
        """
        接受新组件并设置权重
        参考自vae/model/boost.py中的add_component方法
        
        参数:
            weight: 新组件的权重
        """
        if self.h_mu is None:
            raise ValueError("没有待接受的组件，请先调用train_new_component")
        
        print(f"接受新组件，权重: {weight:.6f}")
        
        # 编码新组件
        with torch.no_grad():
            mu, logvar = self.model.encoder(self.h_mu)
        
        # 添加到先验
        self.model.prior.add_component(self.h_mu.data.clone(), alpha=weight)
        
        # 如果模型有存储编码结果的属性，也要更新
        if hasattr(self.model, 'pr_q_means'):
            self.model.pr_q_means.append(nn.Parameter(mu.data.clone(), requires_grad=False))
        if hasattr(self.model, 'pr_q_logvars'):
            self.model.pr_q_logvars.append(nn.Parameter(logvar.data.clone(), requires_grad=False))
        
        print(f"✓ 新组件已接受，当前总组件数: {len(self.model.prior.mu_list)}")

    def prune_components(self, threshold: float = 0.01) -> int:
        """
        修剪权重过小的组件
        参考自vae/model/boost.py中的update_component_weigts方法
        
        参数:
            threshold: 权重阈值，低于此值的组件将被删除
            
        返回:
            删除的组件数量
        """
        if not hasattr(self.model.prior, 'weights'):
            return 0
            
        initial_count = len(self.model.prior.weights)
        
        # 找出权重大于阈值的组件
        valid_mask = self.model.prior.weights > threshold
        
        if valid_mask.sum() == len(self.model.prior.weights):
            print("没有需要修剪的组件")
            return 0
        
        print(f"修剪 {(~valid_mask).sum()} 个权重过小的组件")
        
        # 更新权重
        self.model.prior.weights = self.model.prior.weights[valid_mask]
        self.model.prior.task_weight = self.model.prior.task_weight[valid_mask]
        
        # 更新伪输入列表
        self.model.prior.mu_list = [
            self.model.prior.mu_list[i] for i in range(len(valid_mask)) if valid_mask[i]
        ]
        
        # 更新编码结果
        if hasattr(self.model, 'pr_q_means'):
            self.model.pr_q_means = [
                self.model.pr_q_means[i] for i in range(len(valid_mask)) if valid_mask[i]
            ]
        if hasattr(self.model, 'pr_q_logvars'):
            self.model.pr_q_logvars = [
                self.model.pr_q_logvars[i] for i in range(len(valid_mask)) if valid_mask[i]
            ]
        
        # 重新归一化权重
        self.model.prior.weights = self.model.prior.weights / self.model.prior.weights.sum()
        
        removed_count = initial_count - len(self.model.prior.weights)
        print(f"修剪完成，删除了 {removed_count} 个组件")
        
        return removed_count

    def _cache_optimal_prior(self, X_opt: torch.Tensor) -> None:
        """
        缓存最优先验参数
        参考自models/cl_scetm.py中的_cache_optimal_prior方法
        """
        print("缓存最优先验参数...")
        
        with torch.no_grad():
            means, logvars = self.model.encoder(X_opt)
        
        self._cached_optimal_means = means
        self._cached_optimal_logvars = logvars
        self._cached_optimal_weights = torch.ones(means.shape[0], device=self.device) / means.shape[0]

    def _reset_component_parameters(self) -> None:
        """
        重置组件参数
        参考自vae/model/boost.py中的reset_parameters方法
        """
        input_size = getattr(self.model, 'n_genes', getattr(self.model, 'input_size', 784))
        
        self.h_mu = nn.Parameter(
            torch.randn(1, input_size, device=self.device) * self.pseudoinputs_std + self.pseudoinputs_mean
        )

    def _get_current_prior_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取当前先验参数
        参考自models/cl_scetm.py中的_get_current_prior_params方法
        """
        if (hasattr(self.model, 'pr_q_means') and 
            hasattr(self.model, 'pr_q_logvars') and 
            len(self.model.pr_q_means) > 0):
            
            current_means = torch.cat([mu for mu in self.model.pr_q_means], dim=0)
            current_logvars = torch.cat([logvar for logvar in self.model.pr_q_logvars], dim=0)
            current_weights = self.model.prior.weights[:len(self.model.pr_q_means)].clone()
        else:
            # 如果还没有组件，使用标准正态分布
            latent_dim = getattr(self.model, 'n_topics', getattr(self.model, 'latent_dim', 50))
            current_means = torch.zeros(1, latent_dim, device=self.device)
            current_logvars = torch.zeros(1, latent_dim, device=self.device)
            current_weights = torch.ones(1, device=self.device)
        
        return current_means, current_logvars, current_weights

    def _calculate_boosting_loss(self, 
                               pr_means: torch.Tensor,
                               pr_logvars: torch.Tensor, 
                               pr_w: torch.Tensor,
                               lbd: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算BooVAE风格的boosting损失
        参考自vae/model/boost.py中的calculate_boosting_loss方法
        """
        initial_training = self.model.training
        if initial_training:
            self.model.eval()
        
        # 编码新的伪输入
        z_q_mean, z_q_logvar = self.model.encoder(self.h_mu)
        z_sample = self.model.reparameterize(z_q_mean, z_q_logvar)
        
        # 计算熵正则化项
        entropy = 0.5 * (1 + math.log(2*math.pi) + z_q_logvar).sum()
        
        # 计算最优先验的对数概率
        log_mean_q = self._compute_optimal_prior_logprob(z_sample)
        
        # 计算当前先验的对数概率
        log_p_z = self._log_gaussian_mixture(z_sample, pr_means, pr_logvars, pr_w)
        
        # BooVAE损失（参考自vae/model/boost.py）
        loss = -entropy - lbd * log_mean_q + lbd * log_p_z
        
        if initial_training:
            self.model.train()
        
        return loss, entropy, log_mean_q, log_p_z

    def _compute_optimal_weight(self, 
                              max_iter: int = 1000,
                              tol: float = 1e-4,
                              lr: float = 0.1) -> float:
        """
        计算最优权重
        参考自vae/model/boost.py中的get_opt_alpha方法
        """
        w = torch.tensor(0.5, device=self.device, requires_grad=False)
        
        print("计算最优权重...")
        
        for i in range(max_iter):
            grad = self._compute_alpha_gradient(w)
            w = w - lr / (i + 1.) * grad
            w = torch.clamp(w, 1e-4, 0.99)
            
            if i > 20 and abs(grad.item()) < tol:
                break
        
        return w.item()

    def _compute_alpha_gradient(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        计算权重梯度
        参考自vae/model/boost.py中的alpha_grad方法
        """
        with torch.no_grad():
            # 从新组件采样
            z_q_mean, z_q_logvar = self.model.encoder(self.h_mu)
            h_sample = self.model.reparameterize(z_q_mean, z_q_logvar)
            
            # 从现有先验采样
            if hasattr(self.model, 'pr_q_means') and len(self.model.pr_q_means) > 0:
                idx = np.random.randint(len(self.model.pr_q_means))
                p_mu = self.model.pr_q_means[idx]
                p_logvar = self.model.pr_q_logvars[idx]
                p_sample = self.model.reparameterize(p_mu, p_logvar)
            else:
                p_sample = torch.randn_like(h_sample)
            
            # 计算梯度项
            grad_h = self._compute_gradient_term(h_sample, alpha)
            grad_p = self._compute_gradient_term(p_sample, alpha)
        
        return grad_h - grad_p

    def _compute_gradient_term(self, z_sample: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        计算梯度项
        参考自vae/model/boost.py中的grad_weight方法
        """
        with torch.no_grad():
            log_q_z = self._compute_optimal_prior_logprob(z_sample)
            
            # 新组件概率
            z_q_mean, z_q_logvar = self.model.encoder(self.h_mu)
            log_h_z = self._gaussian_log_density(z_sample, z_q_mean, z_q_logvar, dim=1)
            log_h_z = log_h_z.mean()
            
            # 现有先验概率
            if hasattr(self.model, 'pr_q_means') and len(self.model.pr_q_means) > 0:
                current_means, current_logvars, current_weights = self._get_current_prior_params()
                log_p_z = self._log_gaussian_mixture(z_sample, current_means, current_logvars, current_weights)
            else:
                log_p_z = self._gaussian_log_density(z_sample, 
                                                   torch.zeros_like(z_sample), 
                                                   torch.zeros_like(z_sample), dim=1).mean()
            
            log_h_z += torch.log(alpha)
            log_p_z += torch.log(1. - alpha)
            
            comb_log_p = torch.logsumexp(torch.stack([log_p_z, log_h_z]), 0)
        
        return comb_log_p - log_q_z

    def _compute_optimal_prior_logprob(self, z_sample: torch.Tensor) -> torch.Tensor:
        """计算最优先验的对数概率"""
        return self._log_gaussian_mixture(
            z_sample,
            self._cached_optimal_means,
            self._cached_optimal_logvars,
            self._cached_optimal_weights
        )

    def _compute_weight_optimization_loss(self, 
                                        means: torch.Tensor,
                                        logvars: torch.Tensor, 
                                        weights: torch.Tensor,
                                        n_samples: int = 1000) -> torch.Tensor:
        """
        计算权重优化损失
        参考自vae/model/boost.py中的update_component_weigts方法
        """
        # 从当前先验采样
        indices = torch.multinomial(weights, n_samples, replacement=True)
        z_q_mean = means[indices]
        z_q_logvar = logvars[indices]
        z_sample = self.model.reparameterize(z_q_mean, z_q_logvar)
        
        # 计算当前先验的对数概率
        log_pr = self._log_gaussian_mixture(z_sample, means, logvars, weights)
        
        # 计算最优先验的对数概率
        log_opt = self._compute_optimal_prior_logprob(z_sample)
        
        # 对称KL散度的一半
        kl1 = (log_pr - log_opt).mean()
        
        # 从最优先验采样
        opt_indices = torch.randint(self._cached_optimal_means.shape[0], (n_samples,))
        z_q_mean_opt = self._cached_optimal_means[opt_indices]
        z_q_logvar_opt = self._cached_optimal_logvars[opt_indices]
        z_sample_opt = self.model.reparameterize(z_q_mean_opt, z_q_logvar_opt)
        
        log_pr_opt = self._log_gaussian_mixture(z_sample_opt, means, logvars, weights)
        log_opt_opt = self._compute_optimal_prior_logprob(z_sample_opt)
        
        kl2 = (log_opt_opt - log_pr_opt).mean()
        
        return 0.5 * kl1 + 0.5 * kl2

    def _log_gaussian_mixture(self, 
                            z_sample: torch.Tensor,
                            means: torch.Tensor, 
                            logvars: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
        """
        计算高斯混合分布的对数概率
        参考自vae/model/boost.py中的log_gaus_mixture方法
        """
        z_sample = z_sample.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means = means.unsqueeze(0)        # [1, n_components, latent_dim]
        logvars = logvars.unsqueeze(0)    # [1, n_components, latent_dim]
        
        weights = weights / (weights.sum() + 1e-10)
        log_w = torch.log(weights + 1e-10).unsqueeze(0).to(z_sample.device)
        
        # 计算高斯对数概率
        log_comps = self._gaussian_log_density(z_sample, means, logvars, dim=2)
        
        log_probs = torch.logsumexp(log_comps + log_w, dim=1)
        return log_probs.mean()

    def _gaussian_log_density(self, 
                            z: torch.Tensor,
                            mu: torch.Tensor, 
                            logvar: torch.Tensor,
                            dim: int = 1) -> torch.Tensor:
        """
        计算高斯分布的对数概率密度
        参考自vae/utils/distributions.py中的log_Normal_diag方法
        """
        log_density = -0.5 * (math.log(2.0 * math.pi) + logvar + 
                             torch.pow(z - mu, 2) / (torch.exp(logvar) + 1e-5))
        
        if dim is not None:
            return log_density.sum(dim=dim)
        else:
            return log_density

    def cleanup(self) -> None:
        """清理临时参数"""
        if self.h_mu is not None:
            del self.h_mu
            self.h_mu = None
        
        self._cached_optimal_means = None
        self._cached_optimal_logvars = None
        self._cached_optimal_weights = None