import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import anndata
import logging
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

# 导入现有模块
from models.cl_scetm import CL_scETM
from data.scETMdataset import create_data_loader
from trainers.trainer_utils import set_seed, StatsRecorder

_logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """
    基于增量高斯先验的scETM增量学习训练器
    
    该训练器实现了一种新的增量学习方法：
    1. 使用标准scETM架构
    2. 保存上一次训练的编码器输出的均值和方差统计
    3. 将这些统计信息作为下一次训练的先验分布
    """
    
    def __init__(self,
                 n_genes: int,
                 n_topics: int = 50,
                 hidden_sizes: List[int] = [128],
                 gene_emb_dim: int = 400,
                 learning_rate: float = 5e-3,
                 weight_decay: float = 0.0,
                 prior_strength: float = 1.0,
                 adaptive_strength: bool = True,
                 device: torch.device = None,
                 seed: int = 42):
        """
        初始化增量训练器
        
        参数:
            n_genes: 基因数量
            n_topics: 主题数量
            hidden_sizes: 编码器隐藏层大小
            gene_emb_dim: 基因嵌入维度
            learning_rate: 学习率
            weight_decay: 权重衰减
            prior_strength: 先验强度
            adaptive_strength: 是否自适应调整先验强度
            device: 训练设备
            seed: 随机种子
        """
        self.n_genes = n_genes
        self.n_topics = n_topics
        self.hidden_sizes = hidden_sizes
        self.gene_emb_dim = gene_emb_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prior_strength = prior_strength
        self.adaptive_strength = adaptive_strength
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # 设置随机种子
        set_seed(seed)
        
        # 初始化模型为None，在首次训练时创建
        self.model = None
        
        # 训练历史记录
        self.training_history = []
        self.dataset_count = 0
        
        _logger.info(f"IncrementalTrainer初始化完成，设备: {self.device}")
    
    def _create_model(self, prior_type: str = 'standard') -> CL_scETM:
        """
        创建模型实例
        
        参数:
            prior_type: 先验类型 ('standard', 'incremental')
            
        返回:
            CL_scETM模型实例
        """
        model = CL_scETM(
            n_genes=self.n_genes,
            n_topics=self.n_topics,
            hidden_sizes=self.hidden_sizes,
            gene_emb_dim=self.gene_emb_dim,
            prior_type=prior_type,
            prior_strength=self.prior_strength,
            adaptive_strength=self.adaptive_strength,
            device=self.device
        )
        return model
    
    def train_first_dataset(self,
                           adata: anndata.AnnData,
                           dataset_name: str,
                           n_epochs: int = 100,
                           batch_size: int = 1024,
                           batch_col: str = 'batch_indices',
                           eval_every: int = 10,
                           save_path: Optional[str] = None) -> Dict:
        """
        训练第一个数据集（使用标准先验）
        
        参数:
            adata: 数据集
            dataset_name: 数据集名称
            n_epochs: 训练轮数
            batch_size: 批大小
            batch_col: 批次列名
            eval_every: 评估间隔
            save_path: 保存路径
            
        返回:
            训练历史
        """
        _logger.info(f"训练第一个数据集: {dataset_name}, 大小: {adata.n_obs}x{adata.n_vars}")
        
        # 创建使用标准先验的模型
        self.model = self._create_model(prior_type='standard')
        
        # 训练模型
        history = self._train_dataset(
            adata=adata,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            batch_col=batch_col,
            eval_every=eval_every,
            is_first_dataset=True
        )
        
        # 转换到增量模式
        self._convert_to_incremental_model()
        
        # 保存模型
        if save_path:
            self.save_model(save_path)
        
        self.dataset_count = 1
        _logger.info(f"第一个数据集训练完成，模型已转换为增量模式")
        
        return history
    
    def train_incremental_dataset(self,
                                 adata: anndata.AnnData,
                                 dataset_name: str,
                                 n_epochs: int = 100,
                                 batch_size: int = 1024,
                                 batch_col: str = 'batch_indices',
                                 eval_every: int = 10,
                                 save_path: Optional[str] = None) -> Dict:
        """
        增量训练新数据集
        
        参数:
            adata: 新数据集
            dataset_name: 数据集名称
            n_epochs: 训练轮数
            batch_size: 批大小
            batch_col: 批次列名
            eval_every: 评估间隔
            save_path: 保存路径
            
        返回:
            训练历史
        """
        if self.model is None:
            raise ValueError("请先使用train_first_dataset训练第一个数据集")
        
        if self.model.prior_type != 'incremental':
            raise ValueError("模型必须处于增量模式")
        
        _logger.info(f"增量训练数据集: {dataset_name}, 大小: {adata.n_obs}x{adata.n_vars}")
        
        # 打印先验信息
        prior_info = self.model.get_prior_info()
        _logger.info(f"当前先验信息: {prior_info}")
        
        # 训练模型
        history = self._train_dataset(
            adata=adata,
            dataset_name=dataset_name,
            n_epochs=n_epochs,
            batch_size=batch_size,
            batch_col=batch_col,
            eval_every=eval_every,
            is_first_dataset=False
        )
        
        # 保存模型
        if save_path:
            self.save_model(save_path)
        
        self.dataset_count += 1
        _logger.info(f"增量训练完成，总数据集数: {self.dataset_count}")
        
        return history
    
    def _train_dataset(self,
                      adata: anndata.AnnData,
                      dataset_name: str,
                      n_epochs: int,
                      batch_size: int,
                      batch_col: str,
                      eval_every: int,
                      is_first_dataset: bool) -> Dict:
        """
        训练单个数据集的核心逻辑
        """
        start_time = time.time()
        
        # 创建数据加载器
        train_loader = create_data_loader(
            adata, 
            batch_size=batch_size, 
            shuffle=True, 
            batch_col=batch_col,
            num_workers=0
        )
        
        # 设置优化器
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 训练循环
        epoch_history = []
        self.model.train()
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_idx, data_dict in enumerate(train_loader):
                # 将数据移到设备
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # 计算KL权重（warmup策略）
                kl_weight = min(1.0, epoch / (n_epochs * 0.3))
                hyper_param_dict = {'kl_weight': kl_weight}
                
                # 训练步骤
                record = self.model.train_step(optimizer, data_dict, hyper_param_dict)
                epoch_losses.append(record)
            
            # 记录epoch统计
            avg_loss = np.mean([r['loss'] for r in epoch_losses])
            avg_nll = np.mean([r['nll'] for r in epoch_losses])
            avg_kl = np.mean([r['kl'] for r in epoch_losses])
            
            epoch_record = {
                'epoch': epoch,
                'loss': avg_loss,
                'nll': avg_nll,
                'kl': avg_kl,
                'kl_weight': kl_weight
            }
            epoch_history.append(epoch_record)
            
            # 打印进度
            if epoch % eval_every == 0:
                _logger.info(f"  Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}, "
                           f"NLL: {avg_nll:.4f}, KL: {avg_kl:.4f}")
        
        # 完成数据集训练
        if not is_first_dataset:
            self.model.finalize_dataset_training()
        
        # 记录训练历史
        dataset_record = {
            'dataset_name': dataset_name,
            'dataset_count': self.dataset_count + (0 if is_first_dataset else 1),
            'is_first_dataset': is_first_dataset,
            'n_epochs': n_epochs,
            'n_samples': adata.n_obs,
            'n_genes': adata.n_vars,
            'duration': time.time() - start_time,
            'epoch_history': epoch_history,
            'final_loss': avg_loss,
            'prior_info': self.model.get_prior_info()
        }
        
        self.training_history.append(dataset_record)
        
        return dataset_record
    
    def _convert_to_incremental_model(self):
        """
        将标准模型转换为增量模型
        """
        if self.model is None or self.model.prior_type != 'standard':
            return
        
        _logger.info("转换模型到增量模式...")
        
        # 保存当前模型状态
        model_state = self.model.state_dict()
        
        # 创建增量模型
        incremental_model = self._create_model(prior_type='incremental')
        
        # 加载权重
        incremental_model.load_state_dict(model_state)
        
        # 替换模型
        self.model = incremental_model
        
        # 使用第一个数据集的统计信息初始化先验
        self.model.finalize_dataset_training()
        
        _logger.info("模型已转换为增量模式")
    
    def evaluate_model(self, 
                      adata: anndata.AnnData,
                      batch_size: int = 1024,
                      batch_col: str = 'batch_indices') -> Dict:
        """
        评估模型性能
        
        参数:
            adata: 测试数据集
            batch_size: 批大小
            batch_col: 批次列名
            
        返回:
            评估结果
        """
        if self.model is None:
            raise ValueError("模型尚未初始化")
        
        self.model.eval()
        
        # 创建数据加载器
        test_loader = create_data_loader(
            adata, 
            batch_size=batch_size, 
            shuffle=False, 
            batch_col=batch_col,
            num_workers=0
        )
        
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for data_dict in test_loader:
                # 将数据移到设备
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # 计算损失
                loss, loss_dict = self.model.calculate_loss(
                    x=data_dict['X'],
                    batch_indices=data_dict.get('batch_indices', None),
                    beta=1.0,
                    average=False
                )
                
                batch_size = data_dict['X'].size(0)
                total_loss += loss.sum().item()
                total_nll += loss_dict['nll'].sum().item()
                total_kl += loss_dict['kl'].sum().item()
                total_samples += batch_size
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        avg_nll = total_nll / total_samples
        avg_kl = total_kl / total_samples
        
        eval_result = {
            'total_samples': total_samples,
            'avg_loss': avg_loss,
            'avg_nll': avg_nll,
            'avg_kl': avg_kl,
            'perplexity': np.exp(avg_nll)
        }
        
        _logger.info(f"评估结果: Loss={avg_loss:.4f}, NLL={avg_nll:.4f}, "
                   f"KL={avg_kl:.4f}, Perplexity={eval_result['perplexity']:.4f}")
        
        return eval_result
    
    def get_topic_representations(self, adata: anndata.AnnData, 
                                batch_size: int = 1024,
                                batch_col: str = 'batch_indices') -> Dict[str, np.ndarray]:
        """
        获取主题表示
        
        参数:
            adata: 数据集
            batch_size: 批大小
            batch_col: 批次列名
            
        返回:
            包含'theta'和'delta'的字典
        """
        if self.model is None:
            raise ValueError("模型尚未初始化")
        
        self.model.eval()
        
        # 创建数据加载器
        data_loader = create_data_loader(
            adata, 
            batch_size=batch_size, 
            shuffle=False, 
            batch_col=batch_col,
            num_workers=0
        )
        
        theta_list = []
        delta_list = []
        
        with torch.no_grad():
            for data_dict in data_loader:
                # 将数据移到设备
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # 前向传播
                output = self.model.forward(
                    x=data_dict['X'],
                    batch_indices=data_dict.get('batch_indices', None)
                )
                
                theta_list.append(output['theta'].cpu().numpy())
                delta_list.append(output['delta'].cpu().numpy())
        
        # 合并结果
        theta = np.concatenate(theta_list, axis=0)
        delta = np.concatenate(delta_list, axis=0)
        
        return {'theta': theta, 'delta': delta}
    
    def save_model(self, save_path: str):
        """
        保存模型和训练历史
        """
        if self.model is None:
            raise ValueError("模型尚未初始化")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存状态
        save_dict = {
            'incremental_state': self.model.save_incremental_state(),
            'trainer_config': {
                'n_genes': self.n_genes,
                'n_topics': self.n_topics,
                'hidden_sizes': self.hidden_sizes,
                'gene_emb_dim': self.gene_emb_dim,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'prior_strength': self.prior_strength,
                'adaptive_strength': self.adaptive_strength,
                'seed': self.seed,
            },
            'training_history': self.training_history,
            'dataset_count': self.dataset_count,
        }
        
        torch.save(save_dict, save_path)
        _logger.info(f"模型已保存到: {save_path}")
    
    def load_model(self, load_path: str):
        """
        加载模型和训练历史
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 恢复训练器配置
        trainer_config = checkpoint['trainer_config']
        self.n_genes = trainer_config['n_genes']
        self.n_topics = trainer_config['n_topics']
        self.hidden_sizes = trainer_config['hidden_sizes']
        self.gene_emb_dim = trainer_config['gene_emb_dim']
        self.learning_rate = trainer_config['learning_rate']
        self.weight_decay = trainer_config['weight_decay']
        self.prior_strength = trainer_config['prior_strength']
        self.adaptive_strength = trainer_config['adaptive_strength']
        self.seed = trainer_config['seed']
        
        # 创建模型
        incremental_state = checkpoint['incremental_state']
        prior_type = incremental_state['prior_type']
        self.model = self._create_model(prior_type=prior_type)
        
        # 加载模型状态
        self.model.load_incremental_state(incremental_state)
        
        # 恢复训练历史
        self.training_history = checkpoint['training_history']
        self.dataset_count = checkpoint['dataset_count']
        
        _logger.info(f"模型已从 {load_path} 加载，已训练 {self.dataset_count} 个数据集")
    
    def get_training_summary(self) -> Dict:
        """
        获取训练摘要
        """
        if not self.training_history:
            return {'total_datasets': 0, 'total_training_time': 0.0}
        
        total_time = sum(record['duration'] for record in self.training_history)
        total_samples = sum(record['n_samples'] for record in self.training_history)
        
        summary = {
            'total_datasets': len(self.training_history),
            'total_training_time': total_time,
            'total_samples': total_samples,
            'average_time_per_dataset': total_time / len(self.training_history),
            'datasets': []
        }
        
        for record in self.training_history:
            dataset_summary = {
                'name': record['dataset_name'],
                'is_first': record['is_first_dataset'],
                'n_samples': record['n_samples'],
                'n_epochs': record['n_epochs'],
                'duration': record['duration'],
                'final_loss': record['final_loss']
            }
            summary['datasets'].append(dataset_summary)
        
        return summary