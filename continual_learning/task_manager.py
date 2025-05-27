# continual_learning/task_manager.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import numpy as np

_logger = logging.getLogger(__name__)


class TaskManager:
    """
    任务管理器
    
    管理持续学习中的任务切换、权重分配和知识保持。
    参考自: BooVAE vae/utils/mixture.py 和 vae/model/boost.py 的任务管理逻辑
    
    主要功能:
    - 跟踪当前任务和历史任务
    - 管理任务之间的权重分配
    - 协调各持续学习组件的工作
    - 提供任务切换的统一接口
    - 监控任务学习进度
    
    Args:
        model: 主模型
        max_tasks: 最大任务数量
        task_balance_strategy: 任务平衡策略 ('equal', 'weighted', 'decay')
        device: 计算设备
    """
    
    def __init__(self,
        model: nn.Module,
        max_tasks: int = 10,
        task_balance_strategy: str = 'equal',
        device: torch.device = None
    ):
        self.model = model
        self.max_tasks = max_tasks
        self.task_balance_strategy = task_balance_strategy
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 任务状态跟踪
        self.current_task_id = 0
        self.completed_tasks = []
        self.task_histories = {}
        
        # 任务权重管理 (来自mixture.py)
        self.task_weights = torch.ones(1, device=self.device)  # 初始化第一个任务权重为1
        
        # 组件管理
        self.task_components = {}  # 每个任务的组件映射
        self.component_task_mapping = {}  # 组件到任务的反向映射
        
        _logger.info(f"初始化TaskManager: 最大任务数={max_tasks}, 平衡策略={task_balance_strategy}")

    def start_new_task(self, task_id: Optional[int] = None, task_info: Dict[str, Any] = None) -> int:
        """
        开始新任务
        
        参考自: BooVAE vae/model/boost.py 的任务切换逻辑
        处理从一个任务切换到下一个任务的所有必要操作
        
        Args:
            task_id: 任务ID，如果为None则自动分配
            task_info: 任务相关信息
            
        Returns:
            新任务的ID
        """
        # 完成当前任务 (如果存在)
        if self.current_task_id is not None and self.current_task_id not in self.completed_tasks:
            self.finish_current_task()
            
        # 分配新任务ID
        if task_id is None:
            task_id = len(self.completed_tasks)
            
        if task_id >= self.max_tasks:
            raise ValueError(f"任务ID {task_id} 超过最大任务数 {self.max_tasks}")
            
        self.current_task_id = task_id
        
        # 初始化任务信息
        if task_info is None:
            task_info = {}
            
        self.task_histories[task_id] = {
            'start_time': torch.tensor(0.0),  # 可以用实际时间戳
            'components': [],
            'info': task_info,
            'status': 'active'
        }
        
        # 更新模型的任务状态 (来自mixture.py)
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'num_tasks'):
            self.model.prior.num_tasks = task_id + 1
            
        # 更新任务权重 (来自mixture.py的任务权重管理逻辑)
        self._update_task_weights(task_id)
        
        _logger.info(f"开始新任务 {task_id}, 当前总任务数: {len(self.task_histories)}")
        return task_id

    def finish_current_task(self):
        """
        完成当前任务
        
        参考自: BooVAE vae/model/boost.py 的 finish_training_task 方法
        保存当前任务的知识并准备任务切换
        """
        if self.current_task_id is None:
            _logger.warning("没有活跃的任务需要完成")
            return
            
        task_id = self.current_task_id
        _logger.info(f"完成任务 {task_id}")
        
        # 更新任务状态
        if task_id in self.task_histories:
            self.task_histories[task_id]['status'] = 'completed'
            self.task_histories[task_id]['end_time'] = torch.tensor(0.0)  # 实际时间戳
            
        # 保存任务的组件信息 (来自boost.py)
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'mu_list'):
            # 记录属于当前任务的组件
            if hasattr(self.model.prior, 'task_weight'):
                current_task_components = (self.model.prior.task_weight == (task_id + 1)).nonzero().flatten()
                self.task_components[task_id] = current_task_components.tolist()
                
                # 更新组件到任务的映射
                for comp_idx in current_task_components:
                    self.component_task_mapping[comp_idx.item()] = task_id
                    
        # 将任务标记为已完成
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
            
        _logger.info(f"任务 {task_id} 已完成，共有 {len(self.completed_tasks)} 个已完成任务")

    def _update_task_weights(self, new_task_id: int):
        """
        更新任务权重
        
        参考自: BooVAE vae/utils/mixture.py 的权重管理逻辑
        根据任务平衡策略重新分配任务权重
        
        Args:
            new_task_id: 新任务ID
        """
        n_tasks = new_task_id + 1
        
        if self.task_balance_strategy == 'equal':
            # 等权重策略 (来自mixture.py)
            self.task_weights = torch.ones(n_tasks, device=self.device) / n_tasks
            
        elif self.task_balance_strategy == 'weighted':
            # 加权策略，给新任务更高权重
            weights = torch.ones(n_tasks, device=self.device)
            weights[-1] = 2.0  # 新任务权重翻倍
            self.task_weights = weights / weights.sum()
            
        elif self.task_balance_strategy == 'decay':
            # 衰减策略，旧任务权重逐渐衰减
            weights = torch.pow(0.9, torch.arange(n_tasks, device=self.device))
            weights = torch.flip(weights, [0])  # 新任务权重更高
            self.task_weights = weights / weights.sum()
            
        else:
            raise ValueError(f"未知的任务平衡策略: {self.task_balance_strategy}")
            
        _logger.debug(f"更新任务权重: {self.task_weights}")

    def get_current_task_info(self) -> Dict[str, Any]:
        """获取当前任务信息"""
        if self.current_task_id is None:
            return {}
            
        task_info = self.task_histories.get(self.current_task_id, {}).copy()
        task_info['task_id'] = self.current_task_id
        task_info['is_current'] = True
        
        return task_info

    def get_task_statistics(self) -> Dict[str, Union[int, float, List]]:
        """
        获取任务统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        stats = {
            'total_tasks': len(self.task_histories),
            'completed_tasks': len(self.completed_tasks),
            'current_task_id': self.current_task_id,
            'max_tasks': self.max_tasks,
            'task_balance_strategy': self.task_balance_strategy
        }
        
        # 组件统计
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'mu_list'):
            stats['total_components'] = len(self.model.prior.mu_list)
            stats['components_per_task'] = {}
            
            for task_id, components in self.task_components.items():
                stats['components_per_task'][task_id] = len(components)
                
        # 权重统计
        if len(self.task_weights) > 0:
            stats['task_weights'] = self.task_weights.tolist()
            stats['min_task_weight'] = self.task_weights.min().item()
            stats['max_task_weight'] = self.task_weights.max().item()
            
        return stats

    def get_task_components(self, task_id: int) -> List[int]:
        """
        获取指定任务的组件列表
        
        Args:
            task_id: 任务ID
            
        Returns:
            组件索引列表
        """
        return self.task_components.get(task_id, [])

    def get_component_task(self, component_id: int) -> Optional[int]:
        """
        获取指定组件所属的任务
        
        Args:
            component_id: 组件ID
            
        Returns:
            任务ID，如果组件不存在则返回None
        """
        return self.component_task_mapping.get(component_id)

    def balance_task_weights(self, strategy: Optional[str] = None):
        """
        重新平衡任务权重
        
        Args:
            strategy: 平衡策略，如果为None则使用当前策略
        """
        if strategy is not None:
            self.task_balance_strategy = strategy
            
        if self.current_task_id is not None:
            self._update_task_weights(self.current_task_id)
            
        # 如果模型有混合先验，更新其权重
        if (hasattr(self.model, 'prior') and 
            hasattr(self.model.prior, 'weights') and
            hasattr(self.model.prior, 'task_weight')):
            
            # 重新计算组件权重 (来自mixture.py的权重重分配逻辑)
            self._rebalance_component_weights()
            
    def _rebalance_component_weights(self):
        """
        重新平衡组件权重
        参考自: BooVAE vae/utils/mixture.py 的权重重分配逻辑
        """
        if not (hasattr(self.model, 'prior') and 
                hasattr(self.model.prior, 'weights') and
                hasattr(self.model.prior, 'task_weight')):
            return
            
        n_tasks = len(self.task_weights)
        new_weights = []
        
        # 为每个任务重新分配权重 (来自mixture.py)
        for task_id in range(n_tasks):
            task_mask = self.model.prior.task_weight == (task_id + 1)
            task_component_count = task_mask.sum().item()
            
            if task_component_count > 0:
                # 任务权重平均分配给该任务的所有组件
                weight_per_component = self.task_weights[task_id] / task_component_count
                task_weights = torch.full((task_component_count,), weight_per_component.item())
                new_weights.append(task_weights)
                
        if new_weights:
            self.model.prior.weights = torch.cat(new_weights).to(self.device)
            
        _logger.info("重新平衡了组件权重")

    def should_add_component(self, 
                           current_loss: float, 
                           loss_threshold: float = 1e-2,
                           min_components: int = 1,
                           max_components_per_task: int = 50) -> bool:
        """
        判断是否应该添加新组件
        
        基于当前损失、组件数量等因素决定是否需要添加新组件
        
        Args:
            current_loss: 当前训练损失
            loss_threshold: 损失阈值
            min_components: 每个任务最小组件数
            max_components_per_task: 每个任务最大组件数
            
        Returns:
            是否应该添加组件
        """
        if self.current_task_id is None:
            return False
            
        # 获取当前任务的组件数
        current_task_components = len(self.get_task_components(self.current_task_id))
        
        # 基本条件检查
        if current_task_components < min_components:
            return True
            
        if current_task_components >= max_components_per_task:
            return False
            
        # 基于损失的判断
        if current_loss > loss_threshold:
            return True
            
        return False

    def should_prune_components(self,
                              prune_interval: int = 100,
                              min_components_per_task: int = 1) -> bool:
        """
        判断是否应该修剪组件
        
        Args:
            prune_interval: 修剪间隔（训练步数）
            min_components_per_task: 每个任务最小组件数
            
        Returns:
            是否应该修剪组件
        """
        if self.current_task_id is None:
            return False
            
        # 获取当前任务的组件数
        current_task_components = len(self.get_task_components(self.current_task_id))
        
        # 如果组件数太少，不修剪
        if current_task_components <= min_components_per_task:
            return False
            
        # 其他修剪条件可以在这里添加
        # 例如基于训练步数、模型复杂度等
        
        return True

    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """获取内存使用统计"""
        memory_stats = {
            'task_histories_count': len(self.task_histories),
            'task_components_count': len(self.task_components),
            'component_mappings_count': len(self.component_task_mapping),
            'task_weights_size': self.task_weights.numel() if len(self.task_weights) > 0 else 0
        }
        
        # 如果模型有先验，统计先验内存使用
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'mu_list'):
            total_pseudo_inputs = sum(x.numel() for x in self.model.prior.mu_list)
            memory_stats['pseudo_inputs_size'] = total_pseudo_inputs
            
        return memory_stats

    def reset(self, keep_current_task: bool = False):
        """
        重置任务管理器
        
        Args:
            keep_current_task: 是否保留当前任务
        """
        if not keep_current_task:
            self.current_task_id = None
            
        self.completed_tasks.clear()
        self.task_histories.clear()
        self.task_components.clear()
        self.component_task_mapping.clear()
        self.task_weights = torch.ones(1, device=self.device)
        
        _logger.info("任务管理器已重置")

    def export_task_history(self) -> Dict[str, Any]:
        """
        导出任务历史记录
        
        Returns:
            可序列化的任务历史字典
        """
        export_data = {
            'completed_tasks': self.completed_tasks.copy(),
            'current_task_id': self.current_task_id,
            'task_balance_strategy': self.task_balance_strategy,
            'max_tasks': self.max_tasks,
            'task_components': self.task_components.copy(),
            'component_task_mapping': self.component_task_mapping.copy(),
            'task_weights': self.task_weights.tolist() if len(self.task_weights) > 0 else [],
            'task_histories': {}
        }
        
        # 导出任务历史（排除不可序列化的张量）
        for task_id, history in self.task_histories.items():
            export_data['task_histories'][task_id] = {
                'info': history.get('info', {}),
                'status': history.get('status', 'unknown'),
                'components': history.get('components', [])
            }
            
        return export_data

    def import_task_history(self, import_data: Dict[str, Any]):
        """
        导入任务历史记录
        
        Args:
            import_data: 从export_task_history导出的数据
        """
        self.completed_tasks = import_data.get('completed_tasks', [])
        self.current_task_id = import_data.get('current_task_id')
        self.task_balance_strategy = import_data.get('task_balance_strategy', 'equal')
        self.max_tasks = import_data.get('max_tasks', 10)
        self.task_components = import_data.get('task_components', {})
        self.component_task_mapping = import_data.get('component_task_mapping', {})
        
        # 恢复任务权重
        weights_list = import_data.get('task_weights', [])
        if weights_list:
            self.task_weights = torch.tensor(weights_list, device=self.device)
        else:
            self.task_weights = torch.ones(1, device=self.device)
            
        # 恢复任务历史
        self.task_histories = {}
        for task_id, history in import_data.get('task_histories', {}).items():
            self.task_histories[int(task_id)] = {
                'info': history.get('info', {}),
                'status': history.get('status', 'unknown'),
                'components': history.get('components', []),
                'start_time': torch.tensor(0.0),
                'end_time': torch.tensor(0.0)
            }
            
        _logger.info(f"导入了 {len(self.task_histories)} 个任务的历史记录")