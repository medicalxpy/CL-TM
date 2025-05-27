# continual_learning/regularization.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

from loss.KL import gaussian_log_density
from loss.RL import get_reconstruction_loss

_logger = logging.getLogger(__name__)


class ContinualRegularizer:
    """
    持续学习正则化器
    
    实现BooVAE的防遗忘正则化机制，通过生成器正则化和编码器正则化防止灾难性遗忘。
    参考自: BooVAE vae/model/boost.py 的 generator_regularization 方法
    
    主要功能:
    - 生成器正则化: 确保新生成器能重构之前任务的数据
    - 编码器正则化: 保持编码器对之前任务数据的编码一致性  
    - 对称KL正则化: 平衡新旧知识的表示
    - 防遗忘约束: 防止学习新任务时完全覆盖旧知识
    
    Args:
        model: 主模型
        reg_weight: 正则化权重
        input_type: 输入数据类型 ('binary', 'continuous', 'count')
        device: 计算设备
    """
    
    def __init__(self,
        model: nn.Module,
        reg_weight: float = 1.0,
        input_type: str = 'continuous',
        device: torch.device = None
    ):
        self.model = model
        self.reg_weight = reg_weight
        self.input_type = input_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 存储之前任务的重构目标 (来自boost.py)
        self.previous_reconstructions = []
        self.previous_encodings = []
        
        _logger.info(f"初始化ContinualRegularizer: 权重={reg_weight}, 输入类型={input_type}")

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        计算持续学习正则化损失
        
        直接参考自: BooVAE vae/model/boost.py 的 generator_regularization 方法
        实现完整的防遗忘正则化机制
        
        Returns:
            正则化损失张量
        """
        if not self._has_previous_knowledge():
            return torch.tensor(0.0, device=self.device)
            
        # 获取之前任务的组件数量 (来自boost.py)
        n_previous = len(self.previous_reconstructions)
        if n_previous == 0:
            return torch.tensor(0.0, device=self.device)
            
        # 获取之前任务的权重 (来自boost.py)
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'weights'):
            weights = self.model.prior.weights[:n_previous].to(self.device)
            # 重新归一化权重
            if hasattr(self.model.prior, 'num_tasks') and self.model.prior.num_tasks > 1:
                weights = weights / (self.model.prior.num_tasks - 1)
        else:
            weights = torch.ones(n_previous, device=self.device) / n_previous
            
        # 计算生成器正则化 (来自boost.py)
        generator_reg = self._compute_generator_regularization(weights)
        
        # 计算编码器正则化 (来自boost.py)  
        encoder_reg = self._compute_encoder_regularization(weights)
        
        # 总正则化损失 (来自boost.py)
        total_reg = generator_reg + encoder_reg
        
        return total_reg

    def _has_previous_knowledge(self) -> bool:
        """检查是否有之前任务的知识"""
        return (len(self.previous_reconstructions) > 0 and
                len(self.previous_encodings) > 0 and
                hasattr(self.model, 'prior') and 
                hasattr(self.model.prior, 'weights'))

    def _compute_generator_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算生成器正则化损失
        
        参考自: BooVAE vae/model/boost.py 中生成器正则化的核心逻辑
        确保当前生成器能够重构之前任务学到的数据分布
        
        Args:
            weights: 之前任务组件的权重
            
        Returns:
            生成器正则化损失
        """
        n_previous = len(self.previous_reconstructions)
        batch_size = n_previous
        
        # 获取之前任务的正确潜在表示 (来自boost.py)
        z_mu_correct = torch.cat([enc['mu'] for enc in self.previous_encodings[:n_previous]], dim=0)
        z_logvar_correct = torch.cat([enc['logvar'] for enc in self.previous_encodings[:n_previous]], dim=0)
        
        # 从之前的分布中采样 (来自boost.py)
        z_samples = self._reparameterize(z_mu_correct, torch.clamp(z_logvar_correct, -5, -2))
        
        # 使用当前生成器重构 (来自boost.py)
        if hasattr(self.model, 'decoder'):
            # 对于有独立解码器的模型
            if hasattr(self.model.decoder, 'forward'):
                # 将z转换为theta (主题分布)
                theta = F.softmax(z_samples, dim=-1)
                recon_output = self.model.decoder(theta)
            else:
                recon_output = self.model.decoder(z_samples)
        else:
            # 对于集成解码器的模型
            theta = F.softmax(z_samples, dim=-1)
            recon_output = self.model.forward(z_samples, decode_only=True)
            
        # 获取重构结果
        if isinstance(recon_output, dict):
            x_mu = recon_output.get('recon_log', recon_output.get('mu'))
            x_logvar = recon_output.get('logvar', torch.zeros_like(x_mu))
        elif isinstance(recon_output, tuple):
            x_mu, x_logvar = recon_output
        else:
            x_mu = recon_output
            x_logvar = torch.zeros_like(x_mu)
            
        # 获取之前任务的目标重构 (来自boost.py)
        x_mu_correct = torch.cat([rec['mu'] for rec in self.previous_reconstructions[:n_previous]], dim=0)
        
        if self.input_type == 'continuous':
            x_logvar_correct = torch.cat([rec.get('logvar', torch.zeros_like(rec['mu'])) 
                                        for rec in self.previous_reconstructions[:n_previous]], dim=0)
        
        # 计算重构损失 (来自boost.py的核心正则化逻辑)
        if self.input_type == 'binary':
            # 对于二元数据使用伯努利KL散度
            generator_reg = self._bernoulli_kl(x_mu, x_mu_correct, dim=1)
        elif self.input_type == 'continuous':
            # 对于连续数据使用高斯KL散度
            generator_reg = self._gaussian_kl(
                x_mu.reshape(batch_size, -1), 
                x_logvar.reshape(batch_size, -1),
                x_mu_correct.reshape(batch_size, -1), 
                x_logvar_correct.reshape(batch_size, -1), 
                dim=1
            )
        else:
            # 对于计数数据使用MSE损失
            generator_reg = F.mse_loss(x_mu, x_mu_correct, reduction='none').sum(-1)
            
        # 应用权重 (来自boost.py)
        weighted_reg = generator_reg * weights.to(generator_reg.device)
        
        return weighted_reg.mean()

    def _compute_encoder_regularization(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算编码器正则化损失
        
        参考自: BooVAE vae/model/boost.py 中编码器正则化的逻辑
        确保编码器对之前任务的数据保持一致的编码
        
        Args:
            weights: 之前任务组件的权重
            
        Returns:
            编码器正则化损失
        """
        n_previous = len(self.previous_encodings)
        
        # 获取之前任务的正确编码 (来自boost.py)
        z_mu_correct = torch.cat([enc['mu'] for enc in self.previous_encodings[:n_previous]], dim=0)
        z_logvar_correct = torch.cat([enc['logvar'] for enc in self.previous_encodings[:n_previous]], dim=0)
        
        # 从正确编码中采样
        z_samples_correct = self._reparameterize(z_mu_correct, z_logvar_correct)
        
        # 获取对应的伪输入 (来自boost.py)
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'mu_list'):
            pseudo_inputs = torch.cat([self.model.prior.mu_list[i] 
                                     for i in range(min(n_previous, len(self.model.prior.mu_list)))], dim=0)
        else:
            # 如果没有伪输入，使用随机输入
            input_dim = z_mu_correct.shape[-1]  # 假设输入维度等于潜在维度
            pseudo_inputs = torch.randn(n_previous, input_dim, device=self.device)
            
        # 使用当前编码器重新编码 (来自boost.py)
        if hasattr(self.model, 'encoder'):
            z_mu_current, z_logvar_current = self.model.encoder(pseudo_inputs)
        else:
            # 对于集成编码器的模型
            fwd_dict = self.model.forward(pseudo_inputs)
            z_mu_current = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
            z_logvar_current = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
            
        # 从当前编码中采样
        z_samples_current = self._reparameterize(z_mu_current, z_logvar_current)
        
        # 计算对称KL散度 (来自boost.py的关键创新)
        # KL(current || correct)
        log_q_current1 = self._mixture_log_prob(z_samples_current, z_mu_current, z_logvar_current, weights)
        log_q_correct1 = self._mixture_log_prob(z_samples_current, z_mu_correct, z_logvar_correct, weights)
        
        # KL(correct || current)  
        log_q_current2 = self._mixture_log_prob(z_samples_correct, z_mu_current, z_logvar_current, weights)
        log_q_correct2 = self._mixture_log_prob(z_samples_correct, z_mu_correct, z_logvar_correct, weights)
        
        # 对称KL散度 (来自boost.py)
        encoder_reg = 0.5 * ((log_q_current1 - log_q_correct1) + (log_q_correct2 - log_q_current2))
        
        return encoder_reg.mean()

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _bernoulli_kl(self, q_mu: torch.Tensor, p_mu: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        计算伯努利分布之间的KL散度
        参考自: BooVAE vae/utils/distributions.py 的 bernoulli_kl 函数
        """
        res = q_mu * (torch.log(q_mu + 1e-5) - torch.log(p_mu + 1e-5))
        res += (1 - q_mu) * (torch.log(1 - q_mu + 1e-5) - torch.log(1 - p_mu + 1e-5))
        if dim is not None:
            return res.sum(dim=dim)
        else:
            return res

    def _gaussian_kl(self, q_mu: torch.Tensor, q_logsigma: torch.Tensor, 
                    p_mu: torch.Tensor, p_logsigma: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        计算高斯分布之间的KL散度
        参考自: BooVAE vae/utils/distributions.py 的 gaus_kl 函数
        """
        res = p_logsigma - q_logsigma - 1 + torch.exp(q_logsigma - p_logsigma)
        res = res + (q_mu - p_mu).pow(2) / (torch.exp(p_logsigma) + 1e-5)
        if dim is not None:
            return 0.5 * res.sum(dim=dim)
        else:
            return 0.5 * res

    def _mixture_log_prob(self, z: torch.Tensor, means: torch.Tensor, 
                         logvars: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        计算混合高斯分布的对数概率
        参考自: BoostingOptimizer中的_mixture_log_prob方法
        """
        z_expand = z.unsqueeze(1)  # [batch_size, 1, latent_dim]
        means_expand = means.unsqueeze(0)  # [1, n_components, latent_dim]
        logvars_expand = logvars.unsqueeze(0)  # [1, n_components, latent_dim]
        
        # 计算各组件的对数概率
        log_comps = gaussian_log_density(z_expand, means_expand, logvars_expand, dim=2)
        
        # 应用权重
        log_w = torch.log(weights + 1e-10).unsqueeze(0).to(z.device)
        log_probs = log_comps + log_w
        
        # 混合分布的对数概率
        return torch.logsumexp(log_probs, dim=1)

    def update_previous_knowledge(self):
        """
        更新之前任务的知识
        
        参考自: BooVAE vae/model/boost.py 的 finish_training_task 和相关逻辑
        在完成当前任务训练后调用，保存当前任务的知识以用于未来的正则化
        """
        if not hasattr(self.model, 'prior') or not hasattr(self.model.prior, 'mu_list'):
            _logger.warning("模型没有混合先验，无法更新之前的知识")
            return
            
        _logger.info("更新之前任务的知识...")
        
        # 获取当前任务的组件 (来自boost.py)
        if hasattr(self.model.prior, 'task_weight') and hasattr(self.model.prior, 'num_tasks'):
            current_task_mask = self.model.prior.task_weight == self.model.prior.num_tasks
            current_task_indices = torch.where(current_task_mask)[0]
        else:
            # 如果没有任务权重，假设最后添加的组件是当前任务的
            current_task_indices = torch.tensor([len(self.model.prior.mu_list) - 1])
            
        # 保存当前任务的编码和重构 (来自boost.py的update_optimal_prior逻辑)
        self.model.eval()
        with torch.no_grad():
            for idx in current_task_indices:
                if idx < len(self.model.prior.mu_list):
                    pseudo_input = self.model.prior.mu_list[idx]
                    
                    # 编码伪输入 (来自boost.py)
                    if hasattr(self.model, 'encoder'):
                        z_mu, z_logvar = self.model.encoder(pseudo_input)
                    else:
                        fwd_dict = self.model.forward(pseudo_input)
                        z_mu = fwd_dict.get('mu_q_delta', fwd_dict.get('mu'))
                        z_logvar = fwd_dict.get('logsigma_q_delta', fwd_dict.get('logvar'))
                        
                    # 解码获取重构 (来自boost.py)
                    if hasattr(self.model, 'decoder'):
                        theta = F.softmax(z_mu, dim=-1)
                        if hasattr(self.model.decoder, 'forward'):
                            recon_output = self.model.decoder(theta)
                        else:
                            recon_output = self.model.decoder(z_mu)
                    else:
                        theta = F.softmax(z_mu, dim=-1)
                        recon_output = self.model.forward(z_mu, decode_only=True)
                        
                    # 保存编码信息
                    encoding_info = {
                        'mu': z_mu.clone(),
                        'logvar': z_logvar.clone()
                    }
                    self.previous_encodings.append(encoding_info)
                    
                    # 保存重构信息
                    if isinstance(recon_output, dict):
                        recon_mu = recon_output.get('recon_log', recon_output.get('mu'))
                        recon_logvar = recon_output.get('logvar')
                    elif isinstance(recon_output, tuple):
                        recon_mu, recon_logvar = recon_output
                    else:
                        recon_mu = recon_output
                        recon_logvar = None
                        
                    reconstruction_info = {'mu': recon_mu.clone()}
                    if recon_logvar is not None:
                        reconstruction_info['logvar'] = recon_logvar.clone()
                        
                    self.previous_reconstructions.append(reconstruction_info)
                    
        _logger.info(f"已保存 {len(current_task_indices)} 个组件的知识，"
                    f"总共 {len(self.previous_encodings)} 个历史编码")

    def clear_previous_knowledge(self):
        """清除之前任务的知识（用于重新开始训练）"""
        self.previous_reconstructions.clear()
        self.previous_encodings.clear()
        _logger.info("已清除所有之前任务的知识")

    def get_regularization_stats(self) -> Dict[str, Union[int, float]]:
        """获取正则化统计信息"""
        return {
            'n_previous_encodings': len(self.previous_encodings),
            'n_previous_reconstructions': len(self.previous_reconstructions),
            'reg_weight': self.reg_weight,
            'input_type': self.input_type,
            'has_previous_knowledge': self._has_previous_knowledge()
        }

    def set_regularization_weight(self, weight: float):
        """设置正则化权重"""
        self.reg_weight = weight
        _logger.info(f"正则化权重已更新为: {weight}")

    def get_memory_usage(self) -> Dict[str, int]:
        """获取内存使用情况"""
        encoding_memory = sum(enc['mu'].numel() + enc['logvar'].numel() 
                            for enc in self.previous_encodings)
        reconstruction_memory = sum(rec['mu'].numel() + 
                                  (rec.get('logvar', torch.tensor(0)).numel() if 'logvar' in rec else 0)
                                  for rec in self.previous_reconstructions)
        
        return {
            'encoding_tensors': encoding_memory,
            'reconstruction_tensors': reconstruction_memory,
            'total_tensors': encoding_memory + reconstruction_memory
        }