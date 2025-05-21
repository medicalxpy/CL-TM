# trainers/trainer_utils.py
import os
import copy
import pandas as pd
import random
from typing import DefaultDict, IO, List, Sequence, Union, Tuple, Optional, Dict, Any
import logging
from collections import defaultdict
import time

import numpy as np
import anndata
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 设置日志
_logger = logging.getLogger(__name__)


class StatsRecorder:
    """用于记录训练统计信息的工具类。
    
    参考自scETM.trainers.trainer_utils._stats_recorder
    
    属性:
        record: 训练统计记录
        fmt: 训练统计信息的打印格式
        log_file: 写入日志的文件流
        writer: 用于tensorboard日志记录的SummaryWriter
    """

    def __init__(self,
        record_log_path: Optional[str] = None,
        fmt: str = "10.4g",
        writer: Optional[SummaryWriter] = None,
        metadata: Optional[pd.DataFrame] = None
    ) -> None:
        """初始化统计记录器。"""
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
        """更新记录并打印进度信息。"""
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
                
        print(f'Epoch {int(epoch):5d}/{total_epochs:5d}\t下一检查点: {next_ckpt_epoch:7d}', end='\r', flush=True)

    def log_and_clear_record(self) -> None:
        """记录累积的统计信息并重置record"""
        for key, val in self.record.items():
            _logger.info(f'{key:12s}: {np.mean(val):{self.fmt}}')
        self.record = defaultdict(list)

    def __del__(self) -> None:
        """析构函数，关闭log_file（如果存在）"""
        if self.log_file is not None:
            self.log_file.close()


def train_test_split(
    adata: anndata.AnnData,
    test_ratio: float = 0.1,
    seed: int = 1
) -> Tuple[anndata.AnnData, anndata.AnnData]:
    """将数据集分割为训练集和测试集。"""
    rng = np.random.default_rng(seed=seed)
    test_indices = rng.choice(adata.n_obs, size=int(test_ratio * adata.n_obs), replace=False)
    train_indices = list(set(range(adata.n_obs)).difference(test_indices))
    train_adata = adata[adata.obs_names[train_indices], :]
    test_adata = adata[adata.obs_names[test_indices], :]
    _logger.info(f'保留 {test_adata.n_obs} 个细胞 ({test_ratio:g}) 作为测试数据。')
    return train_adata, test_adata


def set_seed(seed: int) -> None:
    """设置随机种子。"""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    _logger.info(f'设置随机种子为 {seed}。')


def initialize_logger(ckpt_dir=None, level=logging.INFO, logger=None) -> None:
    """初始化logger。"""
    if logger is None:
        logger = logging.getLogger('CL-TM')
    logger.setLevel(level)
    
    if logger.hasHandlers():
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.warning(f'重新初始化... 文件处理程序 {handler} 将被关闭。')
                logger.removeHandler(handler)
                
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
    
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
        
    if ckpt_dir is not None:
        file_handler = logging.FileHandler(os.path.join(ckpt_dir, 'log.txt'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)


def calc_weight(
    epoch: int,
    n_epochs: int,
    cutoff_ratio: float = 0.,
    warmup_ratio: float = 1/3,
    min_weight: float = 0.,
    max_weight: float = 1e-7
) -> float:
    """计算权重（用于KL预热）。"""
    fully_warmup_epoch = n_epochs * warmup_ratio
    if cutoff_ratio > warmup_ratio:
        _logger.warning(f'cutoff_ratio {cutoff_ratio} 大于 warmup_ratio {warmup_ratio}。这可能不是预期行为。')
        
    if epoch < n_epochs * cutoff_ratio:
        return 0.
        
    if warmup_ratio:
        return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
    else:
        return max_weight