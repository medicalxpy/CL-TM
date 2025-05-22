# trainers/trainer_utils.py
import os
import copy
import pandas as pd
import random
from typing import DefaultDict, IO, List, Sequence, Union, Tuple, Dict, Any, Optional
import logging
from collections import defaultdict

import numpy as np
import anndata
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 设置日志
_logger = logging.getLogger(__name__)


class StatsRecorder:
    """训练统计记录器类。
    
    直接参考自 scETM/trainers/trainer_utils.py 中的 _stats_recorder 类。
    用于记录训练过程中的各种统计信息。
    
    属性:
        record: 训练统计记录
        fmt: 训练统计的打印格式
        log_file: 写入日志的文件流
        writer: 用于tensorboard日志记录的SummaryWriter
    """
    
    def __init__(self,
        record_log_path: Optional[str] = None,
        fmt: str = "10.4g",
        writer: Optional[SummaryWriter] = None,
        metadata: Optional[pd.DataFrame] = None
    ) -> None:
        """初始化统计记录器。
        
        参数:
            record_log_path: 写入日志的文件路径
            fmt: 训练统计的打印格式
            writer: tensorboard的SummaryWriter
            metadata: 元数据DataFrame
        """
        self.record: DefaultDict[List] = defaultdict(list)
        self.fmt: str = fmt
        self.log_file: Optional[IO] = None
        self.writer: Optional[SummaryWriter] = writer
        
        if writer is not None and metadata is not None:
            metadata.to_csv(os.path.join(writer.get_logdir(), 'metadata.tsv'), sep='\t')
            
        if record_log_path is not None:
            self.log_file = open(record_log_path, 'w')
            self._header_logged: bool = False

    def update(self, new_record: dict, epoch: float, total_epochs: int, next_ckpt_epoch: int) -> None:
        """更新记录并打印到控制台。
        
        参考自 scETM/trainers/trainer_utils.py 中的 _stats_recorder.update 方法。
        
        参数:
            new_record: 最新的训练统计
            epoch: 当前epoch
            total_epochs: 总epoch数
            next_ckpt_epoch: 下一个检查点epoch
        """
        if self.log_file is not None:
            if not self._header_logged:
                self._header_logged = True
                self.log_file.write('epoch\t' + '\t'.join(new_record.keys()) + '\n')
            self.log_file.write(f'{epoch}\t' + '\t'.join(map(str, new_record.values())) + '\n')
            
        for key, val in new_record.items():
            print(f'{key}: {val:{self.fmt}}', end='\t')
            self.record[key].append(val)
            if self.writer is not None:
                self.writer.add_scalar(key, val, epoch)
                
        print(f'Epoch {int(epoch):5d}/{total_epochs:5d}\tNext ckpt: {next_ckpt_epoch:7d}', end='\r', flush=True)

    def log_and_clear_record(self) -> None:
        """记录到logger并重置record。
        
        参考自 scETM/trainers/trainer_utils.py 中的 _stats_recorder.log_and_clear_record 方法。
        """
        for key, val in self.record.items():
            _logger.info(f'{key:12s}: {np.mean(val):{self.fmt}}')
        self.record = defaultdict(list)

    def __del__(self) -> None:
        """析构函数，关闭日志文件。"""
        if self.log_file is not None:
            self.log_file.close()


def train_test_split(
    adata: anndata.AnnData,
    test_ratio: float = 0.1,
    seed: int = 1
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """将数据集分割为训练集和测试集。
    
    直接复用自 scETM/trainers/trainer_utils.py 中的 train_test_split 函数。
    
    参数:
        adata: 要分割的数据集
        test_ratio: 测试数据在adata中的比例
        seed: 随机种子
        
    返回:
        训练集和测试集，都是AnnData格式
    """
    rng = np.random.default_rng(seed=seed)
    test_indices = rng.choice(adata.n_obs, size=int(test_ratio * adata.n_obs), replace=False)
    train_indices = list(set(range(adata.n_obs)).difference(test_indices))
    train_adata = adata[adata.obs_names[train_indices], :]
    test_adata = adata[adata.obs_names[test_indices], :]
    _logger.info(f'保留 {test_adata.n_obs} 个细胞 ({test_ratio:g}) 作为测试数据。')
    return train_adata, test_adata


def set_seed(seed: int) -> None:
    """设置随机种子。
    
    直接复用自 scETM/trainers/trainer_utils.py 中的 set_seed 函数。
    
    参数:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    _logger.info(f'设置种子为 {seed}。')


def prepare_for_continual_learning(
    model: nn.Module,
    new_adata: anndata.AnnData,
    prev_genes: Optional[Sequence[str]] = None,
    keep_new_unique_genes: bool = True,
    batch_col: Optional[str] = "batch_indices"
) -> Tuple[nn.Module, anndata.AnnData]:
    """为持续学习准备模型和数据。
    
    修改自 scETM/trainers/trainer_utils.py 中的 prepare_for_transfer 函数，
    适配持续学习场景。
    
    参数:
        model: 已训练的模型
        new_adata: 新数据集
        prev_genes: 之前数据集的基因列表
        keep_new_unique_genes: 是否保留新数据集特有的基因
        batch_col: 批次列名
        
    返回:
        准备好的模型和数据集
    """
    if prev_genes is not None:
        # 找出共享基因
        new_genes = new_adata.var_names
        shared_genes = set(prev_genes).intersection(new_genes)
        
        if not shared_genes:
            _logger.warning("之前的数据集和新数据集没有共享基因！")
            
        if not keep_new_unique_genes:
            # 只保留共享基因
            new_adata = new_adata[:, list(shared_genes)]
        else:
            # 保留所有基因，但可能需要扩展模型
            # 这里需要根据具体模型来处理
            _logger.info(f"新数据集有 {len(new_genes)} 个基因，其中 {len(shared_genes)} 个是共享的。")
    
    # 更新批次信息
    if batch_col is not None and batch_col in new_adata.obs:
        # 更新批次数量
        n_batches = new_adata.obs[batch_col].nunique()
        if hasattr(model, 'n_batches'):
            model.n_batches = max(model.n_batches, n_batches)
            _logger.info(f"更新模型批次数为 {model.n_batches}")
    
    return model, new_adata


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """保存模型检查点。
    
    参考自 scETM/trainers/UnsupervisedTrainer.py 中的 save_model_and_optimizer 方法。
    
    参数:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前epoch
        save_dir: 保存目录
        additional_info: 额外要保存的信息
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # 保存模型
    model_path = os.path.join(save_dir, f'model-{epoch}.pth')
    torch.save(model.state_dict(), model_path)
    
    # 保存优化器
    optimizer_path = os.path.join(save_dir, f'optimizer-{epoch}.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    
    # 保存额外信息
    if additional_info is not None:
        info_path = os.path.join(save_dir, f'info-{epoch}.pth')
        torch.save(additional_info, info_path)
        
    _logger.info(f"检查点已保存到 {save_dir}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    load_dir: str
) -> Dict[str, Any]:
    """加载模型检查点。
    
    参考自 scETM/trainers/UnsupervisedTrainer.py 中的 load_ckpt 方法。
    
    参数:
        model: 模型
        optimizer: 优化器
        epoch: 要加载的epoch
        load_dir: 加载目录
        
    返回:
        额外信息字典
    """
    # 加载模型
    model_path = os.path.join(load_dir, f'model-{epoch}.pth')
    model.load_state_dict(torch.load(model_path))
    
    # 加载优化器
    optimizer_path = os.path.join(load_dir, f'optimizer-{epoch}.pth')
    optimizer.load_state_dict(torch.load(optimizer_path))
    
    # 加载额外信息
    info_path = os.path.join(load_dir, f'info-{epoch}.pth')
    if os.path.exists(info_path):
        additional_info = torch.load(info_path)
    else:
        additional_info = {}
        
    _logger.info(f'参数已从 {load_dir} 恢复。')
    return additional_info


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-3,
    weight_decay: float = 0.0,
    optimizer_type: str = 'adam'
) -> torch.optim.Optimizer:
    """获取优化器。
    
    参考自 BooVAE/utils/trainer.py 中的 get_optimizer 函数。
    
    参数:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        optimizer_type: 优化器类型
        
    返回:
        优化器
    """
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'plateau',
    patience: int = 10,
    factor: float = 0.5,
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """获取学习率调度器。
    
    参考自 BooVAE/utils/trainer.py 中的调度器创建逻辑。
    
    参数:
        optimizer: 优化器
        scheduler_type: 调度器类型
        patience: ReduceLROnPlateau的patience参数
        factor: 学习率降低因子
        **kwargs: 其他参数
        
    返回:
        学习率调度器
    """
    if scheduler_type == 'none':
        return None
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=factor, 
            patience=patience, 
            verbose=True
        )
    elif scheduler_type == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=gamma
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
        
    return scheduler