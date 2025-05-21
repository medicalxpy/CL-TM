# trainers/cl_scETM_trainer.py
import os
import time
import logging
import psutil
from typing import Dict, List, Mapping, Optional, Tuple, Union, Any, Callable

import numpy as np
import anndata
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

# 导入自定义模块
from .trainer_utils import StatsRecorder, train_test_split, set_seed, initialize_logger, calc_weight
# 导入您的数据处理组件
from data.scETM_dataset import SingleCellDataset, create_data_loader

# 设置日志
_logger = logging.getLogger(__name__)


class CL_scETMTrainer:
    """CL-scETM模型的训练器。"""

    attr_fname: Mapping[str, str] = dict(
        model = 'model',
        optimizer = 'opt'
    )

    def __init__(self,
        model: Any,  # CL_scETM类型
        adata: anndata.AnnData,
        ckpt_dir: Optional[str] = None,
        test_ratio: float = 0.,
        data_split_seed: int = 1,
        init_lr: float = 5e-3,
        lr_decay: float = 6e-5,
        batch_size: int = 2000,
        instance_name: str = "CL-scETM",
        restore_epoch: int = 0,
        seed: int = -1,
    ) -> None:
        """初始化CL_scETMTrainer对象。"""
        # 如果指定了随机种子，则设置
        if seed >= 0:
            set_seed(seed)

        self.model = model
        
        # 分割数据集为训练集和测试集
        self.train_adata = self.test_adata = self.adata = adata
        if test_ratio > 0:
            self.train_adata, self.test_adata = train_test_split(adata, test_ratio, seed=data_split_seed)
        
        # 设置优化器和学习率
        self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        self.lr = self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.steps_per_epoch = max(self.train_adata.n_obs / self.batch_size, 1)
        self.device = model.device
        self.step = self.epoch = 0
        self.seed = seed

        # 设置训练实例名称和检查点目录
        self.instance_name = instance_name
        if restore_epoch > 0:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)
        elif ckpt_dir is not None and restore_epoch == 0:
            self.ckpt_dir = os.path.join(ckpt_dir, f"{self.instance_name}_{time.strftime('%m_%d-%H_%M_%S')}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            initialize_logger(self.ckpt_dir)
            _logger.info(f'检查点目录: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None

    def load_ckpt(self, restore_epoch: int, ckpt_dir: Optional[str] = None) -> None:
        """加载模型检查点。"""
        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir
            
        assert ckpt_dir is not None and os.path.exists(ckpt_dir), f"检查点目录 {ckpt_dir} 不存在。"
        
        # 加载所有属性
        for attr, fname in self.attr_fname.items():
            fpath = os.path.join(ckpt_dir, f'{fname}-{restore_epoch}')
            getattr(self, attr).load_state_dict(torch.load(fpath))
            
        _logger.info(f'参数和优化器已从 {ckpt_dir} 恢复。')
        initialize_logger(self.ckpt_dir)
        _logger.info(f'检查点目录: {self.ckpt_dir}')
        self.update_step(restore_epoch * self.steps_per_epoch)

    def update_step(self, jump_to_step: Optional[int] = None) -> None:
        """更新当前步数、epoch和学习率。"""
        if jump_to_step is None:
            self.step += 1
        else:
            self.step = jump_to_step
            
        self.epoch = self.step / self.steps_per_epoch
        
        if self.lr_decay:
            if jump_to_step is None:
                self.lr *= np.exp(-self.lr_decay)
            else:
                self.lr = self.init_lr * np.exp(-jump_to_step * self.lr_decay)
                
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def train(self,
        new_adata: Optional[anndata.AnnData] = None,  # 如果提供则进行持续学习
        n_epochs: int = 800,
        eval_every: int = 200,
        kl_warmup_ratio: Optional[float] = None,  # 如果为None，则根据是否持续学习自动设置
        min_kl_weight: float = 0.,
        max_kl_weight: float = 1e-7,
        test_ratio: float = 0.,
        data_split_seed: int = 1,
        eval: bool = True,
        batch_col: str = "batch_indices",
        save_model_ckpt: bool = True,
        record_log_path: Optional[str] = None,
        writer: Optional[SummaryWriter] = None,
        eval_result_log_path: Optional[str] = None,
        eval_kwargs: Optional[dict] = None,
        # 优化器参数
        init_lr: Optional[float] = None,  # 如果提供，则重置学习率
        # 数据加载参数
        num_workers: int = 4,  # 数据加载的工作线程数
        # 混合先验参数
        boost_params: Optional[Dict[str, Any]] = None,
        **train_kwargs
    ) -> None:
        """训练模型，支持常规训练和持续学习。"""
        # 确定是否进行持续学习
        continue_learning = new_adata is not None
        
        # 保存原始数据集引用，以便在持续学习后恢复
        original_adata = self.adata
        original_train_adata = self.train_adata
        original_test_adata = self.test_adata
        
        # 处理数据集
        if continue_learning:
            # 更新为新数据集
            self.adata = new_adata
            self.train_adata = self.test_adata = self.adata
            if test_ratio > 0:
                self.train_adata, self.test_adata = train_test_split(self.adata, test_ratio, seed=data_split_seed)
        
            # 如果是持续学习且模型支持混合先验，完成当前任务并准备下一个任务
            if hasattr(self.model, 'prior_type') and getattr(self.model, 'prior_type') == 'mixture':
                _logger.info("完成当前任务并准备下一个任务")
                if hasattr(self.model, 'finish_training_task'):
                    self.model.finish_training_task()
        
        # 处理学习率
        if init_lr is not None:
            self.lr = self.init_lr = init_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        
        # 更新每个epoch的步数
        self.steps_per_epoch = max(self.train_adata.n_obs / self.batch_size, 1)
        
        # 设置KL预热比例
        if kl_warmup_ratio is None:
            # 持续学习时默认不使用KL预热，常规训练时使用1/3的预热比例
            kl_warmup_ratio = 0.0 if continue_learning else 1/3
        
        # 处理boost参数
        if boost_params is None:
            boost_params = {}
        
        # 设置默认评估参数
        default_eval_kwargs = dict(
            batch_col = batch_col,
            plot_fname = f'{self.instance_name}_{self.model.clustering_input}',
            plot_dir = self.ckpt_dir,
            writer = writer
        )
        
        if eval_kwargs is not None:
            default_eval_kwargs.update(eval_kwargs)
        eval_kwargs = default_eval_kwargs
        
        # 使用CL-TM中的数据加载组件
        train_loader = create_data_loader(
            self.train_adata,
            batch_size=self.batch_size,
            shuffle=True,
            batch_col=batch_col,
            num_workers=num_workers
        )
        
        # 设置统计记录器
        recorder = StatsRecorder(record_log_path=record_log_path, writer=writer, metadata=self.adata.obs)
        next_ckpt_epoch = min(int(np.ceil(self.epoch / eval_every) * eval_every), n_epochs)

        # 主训练循环
        global_step = 0
        steps_this_epoch = 0
        epoch_start_step = self.step
        
        while self.epoch < n_epochs:
            # 在每个epoch开始时重置计数器
            if steps_this_epoch == 0:
                epoch_start = time.time()
            
            # 计算KL权重（预热过程）
            kl_weight = calc_weight(
                self.epoch, 
                n_epochs, 
                0, 
                kl_warmup_ratio, 
                min_kl_weight, 
                max_kl_weight
            )
            
            # 构建超参数字典
            hyper_param_dict = {'kl_weight': kl_weight}
            
            epoch_loss = 0.0
            batch_count = 0
            
            # 对一个epoch的数据进行训练
            for batch_data in train_loader:
                # 将数据移动到设备上
                data_dict = {k: v.to(self.device) for k, v in batch_data.items()}
                
                # 执行训练步骤
                new_record = self.model.train_step(self.optimizer, data_dict, hyper_param_dict)
                
                # 累积损失
                epoch_loss += new_record['loss']
                batch_count += 1
                
                # 检查是否需要添加新组件（如果使用mixture先验）
                global_step += 1
                steps_this_epoch += 1
                
                # 获取boost相关参数
                comp_ep = boost_params.get('comp_ep', 1.0)
                number_components = boost_params.get('number_components', 500)
                lbd = boost_params.get('lbd', 1.0)
                
                if (hasattr(self.model, 'prior_type') and 
                    getattr(self.model, 'prior_type') == 'mixture' and 
                    boost_params.get('use_boost', False) and 
                    self.epoch <= comp_ep * number_components + 1):
                    
                    # 获取当前组件数
                    if hasattr(self.model.prior, 'mu_list'):
                        comp = len(self.model.prior.mu_list)
                    elif hasattr(self.model.prior, 'num_comp'):
                        comp = self.model.prior.num_comp
                    else:
                        comp = 0
                    
                    n_steps_per_epoch = int(self.steps_per_epoch)
                    
                    if (global_step % int(comp_ep * n_steps_per_epoch) == 0 and 
                        global_step > 0 and 
                        comp < number_components):
                        
                        # 添加新组件
                        _logger.info(f"添加新组件 {comp+1}/{number_components}")
                        if hasattr(self.model, 'add_component'):
                            component_history = self.model.add_component(lbd=lbd)
                            # 记录组件历史（如果需要）
                            if writer is not None and isinstance(component_history, dict):
                                for k, v in component_history.items():
                                    writer.add_scalar(f'component/{k}', v, global_step)
                
                # 更新步数
                self.update_step()
            
            # 计算epoch平均损失
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_time = time.time() - epoch_start
            
            # 更新记录
            current_epoch_record = {
                'loss': avg_epoch_loss,
                'epoch_time': epoch_time
            }
            # 合并新记录
            current_epoch_record.update(new_record)
            
            recorder.update(current_epoch_record, self.epoch, n_epochs, next_ckpt_epoch)
            
            # 重置epoch计数器
            steps_this_epoch = 0

            # 日志和评估
            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {next_ckpt_epoch:.0f}' + '=' * 10)

                # 记录内存使用情况
                _logger.info(repr(psutil.Process().memory_info()))
                
                # 记录当前学习率和KL权重
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr:12.4g}')
                _logger.info(f'{"kl_weight":12s}: {kl_weight:12.4g}')

                # 记录跟踪项的统计信息
                recorder.log_and_clear_record()
                
                # 如果有测试数据，计算测试NLL
                if self.test_adata is not self.adata:
                    test_nll = self.model.get_cell_embeddings_and_nll(
                        self.test_adata, 
                        self.batch_size, 
                        batch_col=batch_col, 
                        emb_names=[]
                    )
                    if test_nll is not None:
                        _logger.info(f'测试NLL: {test_nll:7.4f}')
                else:
                    test_nll = None
                
                # 如果需要评估，则获取嵌入、评估并记录结果
                if eval:
                    from scETM.eval_utils import evaluate
                    
                    # 准备评估参数
                    current_eval_kwargs = eval_kwargs.copy()
                    current_eval_kwargs['plot_fname'] = current_eval_kwargs['plot_fname'] + f'_epoch{int(next_ckpt_epoch)}'
                    
                    # 获取cell embeddings
                    self.model.get_cell_embeddings_and_nll(
                        self.adata, 
                        self.batch_size, 
                        batch_col=batch_col, 
                        emb_names=[self.model.clustering_input]
                    )
                    
                    # 评估
                    result = evaluate(
                        adata=self.adata, 
                        embedding_key=self.model.clustering_input, 
                        **current_eval_kwargs
                    )
                    result['test_nll'] = test_nll
                    
                    # 记录评估结果
                    self._log_eval_result(result, next_ckpt_epoch, writer, eval_result_log_path)

                # 保存检查点
                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    self.save_model_and_optimizer(next_ckpt_epoch)

                _logger.info('=' * 10 + f'评估结束' + '=' * 10)
                next_ckpt_epoch = min(eval_every + next_ckpt_epoch, n_epochs)

        # 清理资源
        del recorder
        _logger.info(f"优化完成: {self.ckpt_dir}")
        
        # 如果是持续学习，恢复原始数据集引用（但保留训练后的模型状态）
        if continue_learning:
            self.adata = original_adata
            self.train_adata = original_train_adata
            self.test_adata = original_test_adata

    def save_model_and_optimizer(self, epoch: int) -> None:
        """保存模型和优化器。"""
        for attr, fname in self.attr_fname.items():
            torch.save(getattr(self, attr).state_dict(), os.path.join(self.ckpt_dir, f'{fname}-{epoch}'))

    def _log_eval_result(self,
        result: Mapping[str, Union[float, None, np.ndarray]],
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        eval_result_log_path: Optional[str] = None
    ) -> None:
        """记录评估结果。"""
        # 记录到tensorboard
        if writer is not None:
            for k, v in result.items():
                if isinstance(v, float):
                    writer.add_scalar(k, v, epoch)
        
        # 记录到文件
        if eval_result_log_path is not None:
            with open(eval_result_log_path, 'a+') as f:
                # ckpt_dir, epoch, test_nll, ari, nmi, k_bet, ebm, time, seed
                f.write(f'{Path(self.ckpt_dir).name}\t'
                        f'{epoch}\t'
                        f'{result["test_nll"]}\t'
                        f'{result["ari"]}\t'
                        f'{result["nmi"]}\t'
                        f'{result["k_bet"]}\t'
                        f'{result["ebm"]}\t'
                        f'{time.strftime("%m_%d-%H_%M_%S")}\t'
                        f'{self.seed}\n')