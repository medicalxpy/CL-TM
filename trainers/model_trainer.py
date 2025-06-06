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
from trainers.component_trainer import ComponentTrainer
from data.scETMdataset import create_data_loader
from trainers.trainer_utils import set_seed, StatsRecorder

_logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    结合BooVAE和scETM的增量学习训练器
    
    实现功能：
    1. 增量训练：基于BooVAE的动态先验方法
    2. 单独训练：常规的单数据集训练
    3. 伪输入的保存与加载
    """
    
    def __init__(self,
                 model: CL_scETM,
                 device: torch.device = None,
                 learning_rate: float = 5e-3,
                 weight_decay: float = 0.0,
                 component_threshold: float = 0.1,  # 伪输入权重阈值
                 max_components_per_dataset: int = 10,  # 每个数据集最大组件数
                 seed: int = 42):
        """
        初始化训练器
        
        参数:
            model: CL_scETM模型
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            component_threshold: 伪输入权重阈值，低于此值停止训练新组件
            max_components_per_dataset: 每个数据集最大组件数
            seed: 随机种子
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.component_threshold = component_threshold
        self.max_components_per_dataset = max_components_per_dataset
        self.seed = seed
        
        # 设置随机种子
        set_seed(seed)
        
        # 初始化组件训练器（参考BooVAE的组件训练逻辑）
        self.component_trainer = ComponentTrainer(
            model=model,
            device=self.device,
            pseudoinputs_mean=0.0,
            pseudoinputs_std=0.1
        )
        
        # 训练历史
        self.training_history = []
        self.dataset_history = []
        
        _logger.info(f"ModelTrainer初始化完成，设备: {self.device}")

    def train_single_dataset(self,
                           adata: anndata.AnnData,
                           n_epochs: int = 100,
                           batch_size: int = 1024,
                           batch_col: str = 'batch_indices',
                           eval_every: int = 10,
                           save_path: Optional[str] = None) -> Dict:
        """
        单数据集训练（非增量训练）
        参考scETM的常规训练流程
        
        参数:
            adata: 数据集
            n_epochs: 训练轮数
            batch_size: 批大小
            batch_col: 批次列名
            eval_every: 评估间隔
            save_path: 保存路径
            
        返回:
            训练历史
        """
        _logger.info(f"开始单数据集训练，数据集大小: {adata.n_obs}x{adata.n_vars}")
        
        # 确保模型使用标准先验
        if self.model.prior_type != 'standard':
            _logger.warning("单数据集训练建议使用标准先验")
        
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
        training_history = []
        self.model.train()
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_idx, data_dict in enumerate(train_loader):
                # 将数据移到设备
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # 计算KL权重（参考scETM的warmup策略）
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
            training_history.append(epoch_record)
            
            # 打印进度
            if epoch % eval_every == 0:
                _logger.info(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}, "
                           f"NLL: {avg_nll:.4f}, KL: {avg_kl:.4f}")
        
        # 保存模型
        if save_path:
            self.save_model(save_path, training_history)
            
        _logger.info("单数据集训练完成")
        return training_history

    def train_incremental(self,
                         new_adata: anndata.AnnData,
                         dataset_name: str,
                         epochs_per_cycle: int = 20,  # 每轮编码器训练的epoch数
                         total_epochs: int = 200,    # 总训练轮数
                         batch_size: int = 1024,
                         batch_col: str = 'batch_indices',
                         save_dir: Optional[str] = None) -> Dict:
        """
        增量训练新数据集
        
        训练逻辑：
        1. 如果有已训练的伪输入：先更新权重，然后交替进行编码器训练和伪输入添加
        2. 如果没有伪输入（第一次）：先用标准先验训练编码器，然后开始添加伪输入
        
        参数:
            new_adata: 新数据集
            dataset_name: 数据集名称
            epochs_per_cycle: 每轮编码器训练的epoch数
            total_epochs: 总训练轮数
            batch_size: 批大小
            batch_col: 批次列名
            save_dir: 保存目录
            
        返回:
            训练历史
        """
        _logger.info(f"开始增量训练，新数据集: {dataset_name}, "
                   f"大小: {new_adata.n_obs}x{new_adata.n_vars}")
        
        training_start_time = time.time()
        dataset_history = {
            'dataset_name': dataset_name,
            'start_time': time.time(),
            'is_first_dataset': False,
            'encoder_training_cycles': [],
            'component_training': [],
            'final_components': 0
        }
        
        # 检查是否是第一次训练（没有伪输入）
        has_existing_components = (hasattr(self.model.prior, 'mu_list') and 
                                 len(self.model.prior.mu_list) > 0)
        
        if not has_existing_components:
            _logger.info("检测到第一次训练，将先使用标准先验训练编码器")
            dataset_history['is_first_dataset'] = True
            # 暂时切换到标准先验进行初始训练
            original_prior_type = self.model.prior_type
            self._switch_to_standard_prior()
        else:
            _logger.info(f"检测到已有 {len(self.model.prior.mu_list)} 个伪输入，开始更新权重")
            # 更新现有伪输入权重
            self._update_existing_weights(new_adata, batch_size)
        
        # 创建数据加载器
        train_loader = create_data_loader(
            new_adata, 
            batch_size=batch_size, 
            shuffle=True, 
            batch_col=batch_col,
            num_workers=0
        )
        
        # 获取优化数据
        X_opt = self._prepare_optimization_data(new_adata, batch_size)
        
        # 设置优化器
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # 交替训练循环
        epoch = 0
        component_count = 0
        
        while epoch < total_epochs:
            # 编码器训练阶段
            _logger.info(f"编码器训练周期 {epoch//epochs_per_cycle + 1}，Epoch {epoch}-{min(epoch+epochs_per_cycle-1, total_epochs-1)}")
            
            encoder_history = self._train_encoder_cycle(
                train_loader, optimizer, epochs_per_cycle, epoch, total_epochs
            )
            dataset_history['encoder_training_cycles'].append(encoder_history)
            
            epoch += epochs_per_cycle
            
            # 如果是第一次训练且完成了初始编码器训练，切换到vamp先验
            if not has_existing_components and epoch >= epochs_per_cycle:
                _logger.info("切换到vamp先验，开始添加伪输入")
                self._switch_to_vamp_prior()
                has_existing_components = True
            
            # 伪输入添加阶段（如果已经有vamp先验）
            if has_existing_components and epoch < total_epochs:
                _logger.info(f"尝试添加第 {component_count + 1} 个伪输入...")
                
                new_weight = self.component_trainer.train_new_component(
                    X_opt=X_opt,
                    max_steps=20000,  # 减少步数，因为要多次添加
                    lbd=1.0,
                    lr=0.003
                )
                
                component_record = {
                    'component_id': component_count,
                    'epoch': epoch,
                    'weight': new_weight,
                    'accepted': new_weight >= self.component_threshold
                }
                dataset_history['component_training'].append(component_record)
                
                # 检查权重阈值
                if new_weight < self.component_threshold:
                    _logger.info(f"新伪输入权重 {new_weight:.6f} 低于阈值 {self.component_threshold}，停止添加")
                    break
                
                # 接受新伪输入
                _logger.info(f"接受新伪输入，权重: {new_weight:.6f}")
                self.component_trainer.accept_new_component_with_weight(new_weight)
                component_count += 1
                
                # 检查最大组件数
                if component_count >= self.max_components_per_dataset:
                    _logger.info(f"达到最大组件数 {self.max_components_per_dataset}，停止添加")
                    break
                
                # 清理临时参数
                self.component_trainer.cleanup()
        
        # 完成训练
        dataset_history['final_components'] = component_count
        dataset_history['end_time'] = time.time()
        dataset_history['duration'] = time.time() - training_start_time
        
        # 记录到历史
        self.dataset_history.append(dataset_history)
        
        # 保存模型和组件
        if save_dir:
            self.save_incremental_state(save_dir, dataset_name, dataset_history)
        
        _logger.info(f"增量训练完成，共添加 {component_count} 个伪输入，"
                   f"耗时 {dataset_history['duration']:.1f} 秒")
        
        return dataset_history

    def _train_encoder_cycle(self,
                           train_loader: DataLoader,
                           optimizer: optim.Optimizer,
                           epochs_per_cycle: int,
                           start_epoch: int,
                           total_epochs: int) -> List[Dict]:
        """
        训练编码器一个周期
        """
        cycle_history = []
        self.model.train()
        
        for cycle_epoch in range(epochs_per_cycle):
            current_epoch = start_epoch + cycle_epoch
            if current_epoch >= total_epochs:
                break
                
            epoch_losses = []
            
            for data_dict in train_loader:
                # 将数据移到设备
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # KL权重调度
                kl_weight = min(1.0, current_epoch / (total_epochs * 0.3))
                hyper_param_dict = {'kl_weight': kl_weight}
                
                # 根据当前先验类型设置训练参数
                if self.model.prior_type == 'vamp' and hasattr(self.model.prior, 'mu_list'):
                    # 使用vamp先验时，需要传递伪输入信息
                    hyper_param_dict['pseudoinputs'] = self.model.prior.mu_list
                    hyper_param_dict['weights'] = getattr(self.model.prior, 'weights', None)
                
                # 训练步骤
                record = self.model.train_step(optimizer, data_dict, hyper_param_dict)
                epoch_losses.append(record)
            
            # 记录epoch统计
            avg_loss = np.mean([r['loss'] for r in epoch_losses])
            avg_nll = np.mean([r['nll'] for r in epoch_losses])
            avg_kl = np.mean([r['kl'] for r in epoch_losses])
            
            epoch_record = {
                'epoch': current_epoch,
                'loss': avg_loss,
                'nll': avg_nll,
                'kl': avg_kl,
                'kl_weight': kl_weight,
                'prior_type': self.model.prior_type
            }
            cycle_history.append(epoch_record)
            
            # 打印进度
            if current_epoch % 5 == 0:
                _logger.info(f"  Epoch {current_epoch}, Loss: {avg_loss:.4f}, "
                           f"NLL: {avg_nll:.4f}, KL: {avg_kl:.4f}, Prior: {self.model.prior_type}")
        
        return cycle_history

    def _update_existing_weights(self, adata: anndata.AnnData, batch_size: int) -> None:
        """
        更新现有伪输入的权重
        """
        if not (hasattr(self.model.prior, 'weights') and len(self.model.prior.weights) > 0):
            return
            
        _logger.info("更新现有伪输入权重...")
        X_opt = self._prepare_optimization_data(adata, batch_size)
        
        self.component_trainer.update_existing_component_weights(
            X_opt, n_steps=300, lr=0.001
        )
        
        _logger.info("伪输入权重更新完成")

    def _switch_to_standard_prior(self) -> None:
        """
        切换到标准先验
        """
        self.model.prior_type = 'standard'
        _logger.info("已切换到标准先验")

    def _switch_to_vamp_prior(self) -> None:
        """
        切换到vamp先验
        """
        from priors.mixture_prior import VampMixture
        
        self.model.prior_type = 'vamp'
        # 初始化空的vamp先验
        if not hasattr(self.model, 'prior') or not hasattr(self.model.prior, 'mu_list'):
            self.model.prior = VampMixture(pseudoinputs=[], alpha=[])
            self.model.pr_q_means = []
            self.model.pr_q_logvars = []
        
        _logger.info("已切换到vamp先验")

    def _prepare_optimization_data(self,
                                 adata: anndata.AnnData,
                                 batch_size: int,
                                 max_samples: int = 1000) -> torch.Tensor:
        """
        准备用于组件优化的数据（参考BooVAE的X_opt准备）
        """
        # 随机采样部分数据用于优化
        n_samples = min(max_samples, adata.n_obs)
        indices = np.random.choice(adata.n_obs, n_samples, replace=False)
        
        # 提取数据
        X_opt = torch.FloatTensor(adata.X[indices].toarray() if hasattr(adata.X, 'toarray') else adata.X[indices])
        X_opt = X_opt.to(self.device)
        
        _logger.info(f"准备优化数据，样本数: {n_samples}")
        return X_opt

    def save_model(self, save_path: str, history: Optional[List] = None) -> None:
        """
        保存模型和训练历史
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_genes': self.model.n_genes,
                'n_topics': self.model.n_topics,
                'prior_type': self.model.prior_type,
            },
            'trainer_config': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'component_threshold': self.component_threshold,
                'seed': self.seed,
            },
            'training_history': history or [],
            'dataset_history': self.dataset_history,
        }
        
        torch.save(save_dict, save_path)
        _logger.info(f"模型已保存到: {save_path}")

    def load_model(self, load_path: str) -> None:
        """
        加载模型和训练历史
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        if 'dataset_history' in checkpoint:
            self.dataset_history = checkpoint['dataset_history']
        
        _logger.info(f"模型已从 {load_path} 加载")

    def save_incremental_state(self,
                              save_dir: str,
                              dataset_name: str,
                              dataset_history: Dict) -> None:
        """
        保存增量训练状态
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型
        model_path = os.path.join(save_dir, f"{dataset_name}_model.pt")
        self.save_model(model_path, dataset_history)
        
        # 保存伪输入信息
        pseudoinputs_path = os.path.join(save_dir, f"{dataset_name}_pseudoinputs.json")
        pseudoinputs_info = self._extract_pseudoinputs_info()
        
        with open(pseudoinputs_path, 'w') as f:
            json.dump(pseudoinputs_info, f, indent=2)
        
        _logger.info(f"增量训练状态已保存到: {save_dir}")

    def load_incremental_state(self, load_dir: str, dataset_name: str) -> None:
        """
        加载增量训练状态
        """
        # 加载模型
        model_path = os.path.join(load_dir, f"{dataset_name}_model.pt")
        if os.path.exists(model_path):
            self.load_model(model_path)
        
        # 加载伪输入信息
        pseudoinputs_path = os.path.join(load_dir, f"{dataset_name}_pseudoinputs.json")
        if os.path.exists(pseudoinputs_path):
            with open(pseudoinputs_path, 'r') as f:
                pseudoinputs_info = json.load(f)
            _logger.info(f"伪输入信息已加载: {len(pseudoinputs_info.get('components', []))} 个组件")

    def _extract_pseudoinputs_info(self) -> Dict:
        """
        提取伪输入信息用于保存
        """
        info = {
            'n_components': 0,
            'components': [],
            'total_weight': 0.0
        }
        
        if (hasattr(self.model.prior, 'mu_list') and 
            hasattr(self.model.prior, 'weights')):
            
            info['n_components'] = len(self.model.prior.mu_list)
            info['total_weight'] = float(self.model.prior.weights.sum())
            
            for i, (pseudo_input, weight) in enumerate(zip(
                self.model.prior.mu_list, 
                self.model.prior.weights
            )):
                component_info = {
                    'id': i,
                    'weight': float(weight),
                    'input_stats': {
                        'mean': float(pseudo_input.mean()),
                        'std': float(pseudo_input.std()),
                        'shape': list(pseudo_input.shape)
                    }
                }
                info['components'].append(component_info)
        
        return info

    def get_training_summary(self) -> Dict:
        """
        获取训练摘要
        """
        summary = {
            'total_datasets': len(self.dataset_history),
            'total_components': 0,
            'training_time': 0.0,
            'datasets': []
        }
        
        for dataset_record in self.dataset_history:
            dataset_summary = {
                'name': dataset_record['dataset_name'],
                'components_added': dataset_record['final_components'],
                'duration': dataset_record.get('duration', 0.0),
                'component_weights': [c['weight'] for c in dataset_record['component_training'] if c['accepted']]
            }
            summary['datasets'].append(dataset_summary)
            summary['total_components'] += dataset_record['final_components']
            summary['training_time'] += dataset_record.get('duration', 0.0)
        
        return summary