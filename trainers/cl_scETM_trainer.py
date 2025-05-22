# trainers/cl_scETM_trainer.py
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Mapping
import logging
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import anndata

# 导入自定义模块
from models.cl_scetm import CL_scETM
from data.scETMdataset import create_data_loader
from .trainer_utils import (
    StatsRecorder, 
    train_test_split, 
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_optimizer,
    get_scheduler,
    prepare_for_continual_learning
)

# 设置日志
_logger = logging.getLogger(__name__)


class CL_scETM_Trainer:
    """CL-scETM的持续学习训练器。
    
    结合了scETM的UnsupervisedTrainer设计和BooVAE的持续学习训练逻辑。
    支持标准先验和混合先验的训练。
    
    属性:
        model: CL_scETM模型
        adata: 当前训练的数据集
        train_adata: 训练数据
        test_adata: 测试数据
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 设备
        ckpt_dir: 检查点目录
        current_task: 当前任务编号（用于持续学习）
        trained_genes: 已训练过的基因列表
    """
    
    def __init__(self,
        model: CL_scETM,
        adata: anndata.AnnData,
        ckpt_dir: Optional[str] = None,
        test_ratio: float = 0.1,
        data_split_seed: int = 1,
        learning_rate: float = 5e-3,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        train_instance_name: str = "cl_scETM",
        seed: int = -1,
        device: Optional[torch.device] = None
    ) -> None:
        """初始化训练器。
        
        参考自 scETM/trainers/UnsupervisedTrainer.py 的 __init__ 方法。
        
        参数:
            model: CL_scETM模型
            adata: 完整的单细胞数据集
            ckpt_dir: 保存检查点的目录
            test_ratio: 测试集比例
            data_split_seed: 数据分割随机种子
            learning_rate: 学习率
            weight_decay: 权重衰减
            batch_size: 批次大小
            train_instance_name: 训练实例名称
            seed: 随机种子
            device: 训练设备
        """
        # 设置随机种子
        if seed >= 0:
            set_seed(seed)
            
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # 模型和数据
        self.model = model.to(device)
        self.adata = adata
        
        # 数据集分割
        if test_ratio > 0:
            self.train_adata, self.test_adata = train_test_split(
                adata, test_ratio, seed=data_split_seed
            )
        else:
            self.train_adata = self.test_adata = adata
            
        # 优化器和调度器
        self.optimizer = get_optimizer(
            self.model, 
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = get_scheduler(self.optimizer)
        
        # 训练参数
        self.batch_size = batch_size
        self.train_instance_name = train_instance_name
        self.seed = seed
        
        # 持续学习相关
        self.current_task = 0
        self.trained_genes = list(adata.var_names) if adata is not None else []
        
        # 检查点目录
        if ckpt_dir is not None:
            self.ckpt_dir = os.path.join(
                ckpt_dir, 
                f"{train_instance_name}_{time.strftime('%m_%d-%H_%M_%S')}"
            )
            os.makedirs(self.ckpt_dir, exist_ok=True)
            _logger.info(f'检查点目录: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None
            
        # 训练状态
        self.step = 0
        self.epoch = 0

    def train(self,
        n_epochs: int = 100,
        eval_every: int = 10,
        save_every: int = 20,
        kl_warmup_epochs: int = 30,
        min_kl_weight: float = 0.0,
        max_kl_weight: float = 1.0,
        eval_metric: str = 'nll',
        batch_col: str = "batch_indices",
        num_workers: int = 0,
        writer: Optional[SummaryWriter] = None,
        add_components_every: Optional[int] = None,
        prune_components: bool = False,
        **kwargs
    ) -> Dict[str, List[float]]:
        """训练模型。
        
        结合了 scETM/trainers/UnsupervisedTrainer.py 的 train 方法
        和 BooVAE/utils/trainer.py 的 train_vae 函数。
        
        参数:
            n_epochs: 训练轮数
            eval_every: 评估间隔
            save_every: 保存间隔
            kl_warmup_epochs: KL权重预热轮数
            min_kl_weight: 最小KL权重
            max_kl_weight: 最大KL权重
            eval_metric: 评估指标
            batch_col: 批次列名
            num_workers: 数据加载线程数
            writer: TensorBoard写入器
            add_components_every: 添加新组件的间隔（仅用于混合先验）
            prune_components: 是否修剪组件（仅用于混合先验）
            **kwargs: 其他参数
            
        返回:
            训练历史字典
        """
        # 创建数据加载器
        train_loader = create_data_loader(
            self.train_adata,
            batch_size=self.batch_size,
            shuffle=True,
            batch_col=batch_col,
            num_workers=num_workers
        )
        
        test_loader = create_data_loader(
            self.test_adata,
            batch_size=self.batch_size,
            shuffle=False,
            batch_col=batch_col,
            num_workers=num_workers
        ) if self.test_adata is not self.train_adata else None
        
        # 设置统计记录器
        record_log_path = os.path.join(self.ckpt_dir, 'train_log.txt') if self.ckpt_dir else None
        recorder = StatsRecorder(
            record_log_path=record_log_path,
            writer=writer,
            metadata=self.adata.obs if self.adata is not None else None
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_nll': [],
            'train_kl': [],
            'test_nll': []
        }
        
        # 最佳验证损失
        best_test_loss = float('inf')
        early_stop_counter = 0
        
        _logger.info(f"开始训练任务 {self.current_task}")
        
        for epoch in range(n_epochs):
            # 计算KL权重（参考scETM的_calc_weight方法）
            kl_weight = self._calc_kl_weight(
                epoch, n_epochs, kl_warmup_epochs, 
                min_kl_weight, max_kl_weight
            )
            
            # 训练一个epoch
            train_record = self._train_epoch(
                train_loader, kl_weight, epoch, n_epochs,
                add_components_every, batch_col
            )
            
            # 更新历史记录
            history['train_loss'].append(train_record['loss'])
            history['train_nll'].append(train_record['nll'])
            history['train_kl'].append(train_record['kl'])
            
            # 记录训练统计
            recorder.update(train_record, self.epoch, n_epochs, 
                          min((epoch // eval_every + 1) * eval_every, n_epochs))
            
            # 评估
            if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
                _logger.info(f"\n{'='*20} Epoch {epoch+1} {'='*20}")
                
                # 记录当前学习率和KL权重
                current_lr = self.optimizer.param_groups[0]['lr']
                _logger.info(f"学习率: {current_lr:.6f}, KL权重: {kl_weight:.6f}")
                
                # 记录训练统计
                recorder.log_and_clear_record()
                
                # 测试集评估
                if test_loader is not None:
                    test_nll = self._evaluate(test_loader, batch_col)
                    history['test_nll'].append(test_nll)
                    _logger.info(f"测试NLL: {test_nll:.4f}")
                    
                    # 学习率调度
                    if self.scheduler is not None:
                        self.scheduler.step(test_nll)
                    
                    # 早停检查
                    if test_nll < best_test_loss:
                        best_test_loss = test_nll
                        early_stop_counter = 0
                        
                        # 保存最佳模型
                        if self.ckpt_dir is not None:
                            self._save_best_model()
                    else:
                        early_stop_counter += 1
                        
                # 组件修剪（仅用于混合先验）
                if prune_components and self.model.prior_type == 'mixture':
                    if (epoch + 1) % (eval_every * 2) == 0:
                        self.model.update_component_weights()
                        
            # 定期保存
            if (epoch + 1) % save_every == 0 and self.ckpt_dir is not None:
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, self.ckpt_dir,
                    additional_info={
                        'current_task': self.current_task,
                        'trained_genes': self.trained_genes,
                        'epoch': self.epoch,
                        'step': self.step
                    }
                )
                
            self.epoch += 1
            
        _logger.info(f"训练完成！最佳测试NLL: {best_test_loss:.4f}")
        
        return history

    def _train_epoch(self, 
                    train_loader, 
                    kl_weight: float,
                    epoch: int,
                    n_epochs: int,
                    add_components_every: Optional[int],
                    batch_col: str) -> Dict[str, float]:
        """训练一个epoch。
        
        参考自 scETM/trainers/UnsupervisedTrainer.py 的 do_train_step 方法
        和 BooVAE/utils/trainer.py 的 train_epoch 函数。
        
        参数:
            train_loader: 训练数据加载器
            kl_weight: KL散度权重
            epoch: 当前epoch
            n_epochs: 总epoch数
            add_components_every: 添加组件间隔
            batch_col: 批次列名
            
        返回:
            训练统计字典
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_kl = 0.0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # 准备数据
            data_dict = self._prepare_batch_data(batch_data, batch_col)
            
            # 构建超参数字典
            hyper_param_dict = {'kl_weight': kl_weight}
            
            # 训练步骤
            record = self.model.train_step(
                self.optimizer, 
                data_dict, 
                hyper_param_dict
            )
            
            # 累积统计
            epoch_loss += record['loss']
            epoch_nll += record['nll']
            epoch_kl += record['kl']
            n_batches += 1
            
            # 添加新组件（仅用于混合先验）
            if (self.model.prior_type == 'mixture' and 
                add_components_every is not None and
                self.step % add_components_every == 0 and
                self.step > 0):
                
                # 使用当前批次的平均值作为新伪输入
                with torch.no_grad():
                    batch_mean = data_dict['cells'].mean(0, keepdim=True)
                self.model.add_component(batch_mean)
                _logger.info(f"在步骤 {self.step} 添加了新组件")
                
            self.step += 1
            
        # 返回平均值
        return {
            'loss': epoch_loss / n_batches,
            'nll': epoch_nll / n_batches,
            'kl': epoch_kl / n_batches
        }

    def _evaluate(self, test_loader, batch_col: str) -> float:
        """评估模型。
        
        参考自 scETM/trainers/UnsupervisedTrainer.py 的评估逻辑。
        
        参数:
            test_loader: 测试数据加载器
            batch_col: 批次列名
            
        返回:
            测试负对数似然
        """
        self.model.eval()
        
        total_nll = 0.0
        n_cells = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                # 准备数据
                data_dict = self._prepare_batch_data(batch_data, batch_col)
                
                # 前向传播
                fwd_dict = self.model.forward(
                    data_dict['cells'],
                    data_dict.get('batch_indices', None),
                    {'decode': True}
                )
                
                # 计算NLL
                if 'nll' in fwd_dict:
                    total_nll += fwd_dict['nll'].item()
                    n_cells += data_dict['cells'].shape[0]
                    
        return total_nll / n_cells if n_cells > 0 else 0.0

    def _prepare_batch_data(self, batch_data: Dict[str, torch.Tensor], 
                           batch_col: str) -> Dict[str, torch.Tensor]:
        """准备批次数据。
        
        参数:
            batch_data: 原始批次数据
            batch_col: 批次列名
            
        返回:
            准备好的数据字典
        """
        # 将数据移到设备上
        data_dict = {}
        for key, val in batch_data.items():
            if isinstance(val, torch.Tensor):
                data_dict[key] = val.to(self.device)
            else:
                data_dict[key] = val
                
        # 计算库大小（如果需要）
        if 'library_size' not in data_dict and 'cells' in data_dict:
            data_dict['library_size'] = data_dict['cells'].sum(1, keepdim=True)
            
        return data_dict

    def _calc_kl_weight(self, epoch: int, n_epochs: int, 
                       warmup_epochs: int, min_weight: float, 
                       max_weight: float) -> float:
        """计算KL权重。
        
        直接参考自 scETM/trainers/UnsupervisedTrainer.py 的 _calc_weight 方法。
        
        参数:
            epoch: 当前epoch
            n_epochs: 总epoch数
            warmup_epochs: 预热epoch数
            min_weight: 最小权重
            max_weight: 最大权重
            
        返回:
            当前KL权重
        """
        if epoch < warmup_epochs:
            # 线性预热
            return min_weight + (max_weight - min_weight) * epoch / warmup_epochs
        else:
            return max_weight

    def _save_best_model(self) -> None:
        """保存最佳模型。"""
        if self.ckpt_dir is not None:
            best_model_path = os.path.join(self.ckpt_dir, 'best_model.pth')
            self.model.save(best_model_path)
            _logger.info(f"最佳模型已保存到 {best_model_path}")

    def continue_training(self, 
                         new_adata: anndata.AnnData,
                         n_epochs: int = 100,
                         prepare_data: bool = True,
                         **train_kwargs) -> Dict[str, List[float]]:
        """使用新数据继续训练（持续学习）。
        
        这是实现持续学习的核心方法，参考了BooVAE的任务切换逻辑。
        
        参数:
            new_adata: 新的数据集
            n_epochs: 训练轮数
            prepare_data: 是否准备数据（处理基因对齐等）
            **train_kwargs: 传递给train方法的其他参数
            
        返回:
            训练历史
        """
        _logger.info(f"开始持续学习，当前任务: {self.current_task + 1}")
        
        # 准备新数据
        if prepare_data:
            self.model, new_adata = prepare_for_continual_learning(
                self.model, 
                new_adata,
                prev_genes=self.trained_genes,
                keep_new_unique_genes=True
            )
            
        # 更新数据集
        self.adata = new_adata
        test_ratio = train_kwargs.pop('test_ratio', 0.1)
        if test_ratio > 0:
            self.train_adata, self.test_adata = train_test_split(
                new_adata, test_ratio
            )
        else:
            self.train_adata = self.test_adata = new_adata
            
        # 如果使用混合先验，完成上一个任务
        if self.model.prior_type == 'mixture' and self.current_task > 0:
            self.model.finish_training_task()
            
        # 更新任务计数
        self.current_task += 1
        
        # 更新已训练基因列表
        self.trained_genes = list(set(self.trained_genes) | set(new_adata.var_names))
        
        # 开始新任务的训练
        return self.train(n_epochs=n_epochs, **train_kwargs)

    def load_checkpoint(self, epoch: int) -> None:
        """加载检查点。
        
        参数:
            epoch: 要加载的epoch
        """
        if self.ckpt_dir is None:
            raise ValueError("未设置检查点目录")
            
        additional_info = load_checkpoint(
            self.model, self.optimizer, epoch, self.ckpt_dir
        )
        
        # 恢复训练状态
        if 'current_task' in additional_info:
            self.current_task = additional_info['current_task']
        if 'trained_genes' in additional_info:
            self.trained_genes = additional_info['trained_genes']
        if 'epoch' in additional_info:
            self.epoch = additional_info['epoch']
        if 'step' in additional_info:
            self.step = additional_info['step']
            
        _logger.info(f"从epoch {epoch} 恢复训练状态")