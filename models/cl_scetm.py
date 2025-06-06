import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Mapping, Any

# 导入核心组件
from models.scETM_core.encoder import EncoderETM
from models.scETM_core.decoder import DecoderETM

# 导入先验
from priors.standard_prior import log_Normal_standard, KL_scETM
from priors.mixture_prior import VampMixture

# 导入损失计算函数
from loss.RL import get_reconstruction_loss
from loss.KL import get_kl_divergence


class CL_scETM(nn.Module):
    """
    CL-scETM: 结合了scETM的单细胞分析能力和持续学习机制的模型
    
    这个模型继承了scETM的主题模型结构，并添加了持续学习能力。
    它可以使用标准先验或混合先验（用于持续学习）。
    
    参数:
        n_genes: 基因数量
        n_topics: 主题数量
        hidden_sizes: 编码器隐藏层大小
        gene_emb_dim: 基因嵌入维度
        bn: 是否使用批归一化
        dropout_prob: dropout概率
        n_batches: 批次数量
        normalize_beta: 是否标准化beta矩阵
        input_batch_id: 是否将批次ID作为输入
        enable_batch_bias: 是否添加批次特定的偏置
        enable_global_bias: 是否添加全局偏置
        prior_type: 先验类型，'standard'或'vamp'
        n_pseudoinputs: 为混合先验初始化的伪输入数量
        pseudoinputs_mean: 伪输入初始化均值
        pseudoinputs_std: 伪输入初始化标准差
        X_opt: 用于初始化伪输入的示例数据
        device: 模型使用的设备
    """
    
    # 用于聚类的输入变量名
    clustering_input = 'delta'
    
    # 可用作嵌入的变量
    emb_names = ['delta', 'theta']
    
    # logsigma的最大和最小值（防止数值问题）
    max_logsigma = 10
    min_logsigma = -10

    def __init__(self,
        n_genes: int,
        n_topics: int = 50,
        hidden_sizes: List[int] = [128],
        gene_emb_dim: int = 400,
        bn: bool = True,
        dropout_prob: float = 0.1,
        n_batches: int = 1,
        normalize_beta: bool = False,
        input_batch_id: bool = False,
        enable_batch_bias: bool = True,
        enable_global_bias: bool = False,
        prior_type: str = 'standard',
        n_pseudoinputs: int = 1,
        pseudoinputs_mean: float = 0.0,
        pseudoinputs_std: float = 0.1,
        X_opt: Optional[torch.Tensor] = None,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super(CL_scETM, self).__init__()
        
        # 保存参数
        self.n_genes = n_genes
        self.n_topics = n_topics
        self.gene_emb_dim = gene_emb_dim
        self.normalize_beta = normalize_beta
        self.input_batch_id = input_batch_id
        self.enable_batch_bias = enable_batch_bias
        self.enable_global_bias = enable_global_bias
        self.n_batches = n_batches
        self.prior_type = prior_type
        self.device = device
        
        # 定义编码器和解码器
        self.encoder = EncoderETM(
            input_size=n_genes,
            output_size=n_topics,
            hidden_sizes=hidden_sizes,
            bn=bn,
            dropout_prob=dropout_prob,
            n_batches=n_batches,
            input_batch_id=input_batch_id
        )
        
        self.decoder = DecoderETM(
            n_topics=n_topics,
            n_genes=n_genes,
            gene_emb_dim=gene_emb_dim,
            normalize_beta=normalize_beta,
            enable_batch_bias=enable_batch_bias,
            enable_global_bias=enable_global_bias,
            n_batches=n_batches
        )
        
        # 初始化先验
        if prior_type == 'vamp':
            # 初始化混合先验相关参数
            self.n_pseudoinputs = n_pseudoinputs
            self.pseudoinputs_mean = pseudoinputs_mean
            self.pseudoinputs_std = pseudoinputs_std
            
            # 获取示例数据，用于初始化伪输入
            self.X_opt = X_opt
            if X_opt is not None:
                mean_opt = X_opt.mean(0, keepdim=True)
                self.prior = VampMixture(pseudoinputs=[mean_opt], alpha=[1.0])
            else:
                # 否则随机初始化
                pseudoinput = torch.randn(1, n_genes) * pseudoinputs_std + pseudoinputs_mean
                self.prior = VampMixture(pseudoinputs=[pseudoinput], alpha=[1.0])
            
            # 存储编码后的伪输入分布
            self.pr_q_means = []
            self.pr_q_logvars = []
        
        # 将模型移动到指定设备
        self.to(device)

    def forward(self, 
                x: torch.Tensor, 
                batch_indices: Optional[torch.Tensor] = None,
                hyper_param_dict: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        模型的前向传播
        
        参数:
            x: 输入数据，形状为[batch_size, n_genes]
            batch_indices: 批次索引，形状为[batch_size]
            hyper_param_dict: 超参数字典，包含beta（KL权重）等
            
        返回:
            包含前向传播结果的字典
        """
        if hyper_param_dict is None:
            hyper_param_dict = {}
        
        # 使用编码器获取隐变量分布
        mu_q_delta, logsigma_q_delta = self.encoder(x, batch_indices)
        
        if self.training:
            # 训练模式下，使用重参数化采样
            delta = self._reparameterize(mu_q_delta, logsigma_q_delta)
        else:
            # 评估模式下，直接使用均值
            delta = mu_q_delta
        
        # 计算主题分布（正则化后的delta）
        theta = F.softmax(delta, dim=-1)
        
        # 使用解码器重构数据
        recon_log = self.decoder(theta, batch_indices)
        
        if not self.training:
            # 评估模式下的返回
            ret_dict = {
                'delta': mu_q_delta,  # 使用均值作为delta
                'theta': theta,  # 主题分布
                'recon_log': recon_log  # 重构的对数概率
            }
            
            # 如果需要，计算负对数似然
            if hyper_param_dict.get('decode', False):
                ret_dict['nll'] = -torch.sum(x * recon_log, dim=1).sum()
            
            return ret_dict
        
        # 训练模式下的返回
        return {
            'delta': delta,  # 采样的delta
            'theta': theta,  # 主题分布
            'recon_log': recon_log,  # 重构的对数概率
            'mu_q_delta': mu_q_delta,  # 均值
            'logsigma_q_delta': logsigma_q_delta  # 对数方差
        }

    def _reparameterize(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """
        使用重参数化技巧从正态分布中采样
        参考自vae/model/simple_vae.py中的reparameterize方法
        
        参数:
            mu: 均值
            logsigma: 对数标准差
            
        返回:
            采样结果
        """
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def calculate_loss(self, 
                      x: torch.Tensor, 
                      batch_indices: Optional[torch.Tensor] = None,
                      beta: float = 1.0,
                      pseudoinputs = None,
                      weights= None,
                      average: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算模型损失
        
        参数:
            x: 输入数据
            batch_indices: 批次索引
            beta: KL散度权重
            average: 是否对batch维度计算平均值
            
        返回:
            总损失和包含各组成部分的字典
        """
        # 前向传播
        fwd_dict = self.forward(x, batch_indices)
        
        # 重构损失（负对数似然）
        nll = -torch.sum(x * fwd_dict['recon_log'], dim=1)
        prior_params=None
        if self.prior_type=='vamp':
            prior_params={
                'encoder':self.encoder,
                'pseudoinputs': pseudoinputs,
                'weights': weights
            }
        # KL散度
        kl=get_kl_divergence(z=fwd_dict['delta'],
                             q_mu=fwd_dict['mu_q_delta'],
                             q_logvar=fwd_dict['logsigma_q_delta'],
                             prior_type=self.prior_type,
                            prior_params=prior_params)
        
        # 计算总损失
        loss = nll + beta * kl
        
        # 如果需要求平均
        if average:
            nll = nll.mean()
            kl = kl.mean()
            loss = loss.mean()
        
        # 返回总损失和各部分
        return loss, {'nll': nll, 'kl': kl}

    def train_step(self, 
                  optimizer: torch.optim.Optimizer, 
                  data_dict: Dict[str, torch.Tensor], 
                  hyper_param_dict: Dict[str, Any], 
                  loss_update_callback: Optional[callable] = None) -> Dict[str, float]:
        """
        执行一步训练
        参考自scETM/models/BaseCellModel.py中的train_step方法
        
        参数:
            optimizer: 优化器
            data_dict: 数据字典，包含'cells'和'batch_indices'等
            hyper_param_dict: 超参数字典，包含'kl_weight'等
            loss_update_callback: 损失更新回调函数
            
        返回:
            训练记录字典
        """
        self.train()
        optimizer.zero_grad()
        
        # 解包数据
        x = data_dict['cells']
        batch_indices = data_dict.get('batch_indices', None)
        
        # 计算损失
        loss, record = self.calculate_loss(
            x, 
            batch_indices=batch_indices, 
            beta=hyper_param_dict.get('kl_weight', 1.0)
        )
        
        # 如果有回调函数，使用回调更新损失
        if loss_update_callback is not None:
            loss, record = loss_update_callback(loss, record)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), 50)
        record['max_norm'] = norms.cpu().numpy()
        
        # 优化器步进
        optimizer.step()
        
        # 转换为Python标量
        record = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in record.items()}
        record['loss'] = loss.item()
        
        return record

    def add_component(self, new_pseudo_input: Optional[torch.Tensor] = None, alpha: Optional[float] = None) -> None:
        """
        向混合先验添加新组件
        参考自vae/model/boost.py中的add_component方法
        
        参数:
            new_pseudo_input: 新伪输入，如不提供则使用默认值
            alpha: 新组件的权重，如不提供则使用默认值
        """
        if self.prior_type != 'vamp':
            raise ValueError("只有使用混合先验时才能添加组件")
        
        # 如果未提供伪输入，则使用默认初始化
        if new_pseudo_input is None:
            new_pseudo_input = torch.randn(1, self.n_genes, device=self.device) * self.pseudoinputs_std + self.pseudoinputs_mean
        
        # 添加到先验
        self.prior.add_component(new_pseudo_input, alpha=alpha)
        
        # 编码新伪输入
        with torch.no_grad():
            mu, logvar = self.encoder(new_pseudo_input)
        
        # 存储编码结果
        self.pr_q_means.append(nn.Parameter(mu.data, requires_grad=False))
        self.pr_q_logvars.append(nn.Parameter(logvar.data, requires_grad=False))

    def update_component_weights(self) -> None:
        """
        更新混合先验中组件的权重
        参考自vae/model/boost.py中的update_component_weigts方法
        """
        if self.prior_type != 'vamp':
            raise ValueError("只有使用混合先验时才能更新组件权重")
        
        print('正在修剪组件...')
        # 获取当前任务的组件
        curr_task = self.prior.task_weight == self.prior.num_tasks
        
        # 获取当前权重
        w = self.prior.weights[curr_task].clone()
        
        # 创建新权重参数
        w_new = nn.Parameter(w)
        
        # 训练权重参数（简化版，实际中可能需要更复杂的优化）
        opt = torch.optim.Adam([w_new], lr=0.0005)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.1)
        
        # 训练循环
        for it in range(500):
            opt.zero_grad()
            
            # 计算损失（这里使用一个简单的示例损失）
            # 实际中应该基于数据计算更复杂的损失
            loss = -torch.sum(w_new * torch.log(w_new + 1e-10))
            
            loss.backward()
            opt.step()
            sched.step(loss)
            
            # 确保权重非负且和为1
            w_new.data = torch.clamp(w_new.data, 0, 1)
            w_new.data = w_new.data / w_new.data.sum()
        
        # 使用优化后的权重更新先验
        self.prior.prune(w_new.data)

    def finish_training_task(self) -> None:
        """
        完成当前任务的训练，准备下一个任务
        参考自vae/model/boost.py中的finish_training_task方法
        """
        if self.prior_type != 'vamp':
            return
        
        initial_train_mode = self.training
        if initial_train_mode:
            self.eval()
        
        # 更新先验的最优分布
        self.prior.update_optimal_prior(self.encoder, self.decoder)
        
        # 增加任务计数
        self.prior.num_tasks += 1
        
        if initial_train_mode:
            self.train()
        
        print(f"完成任务 {self.prior.num_tasks-1} 的训练")

    def get_cell_embeddings_and_nll(self, 
                                   adata, 
                                   batch_size: int = 2000, 
                                   emb_names: Union[str, List[str], None] = None, 
                                   batch_col: str = 'batch_indices', 
                                   inplace: bool = True) -> Union[Union[None, float], Tuple[Dict[str, np.ndarray], Union[None, float]]]:
        """
        获取细胞嵌入和负对数似然
        参考自scETM/models/BaseCellModel.py中的get_cell_embeddings_and_nll方法
        
        参数:
            adata: AnnData对象
            batch_size: 批处理大小
            emb_names: 要返回的嵌入名称
            batch_col: 批次列名
            inplace: 是否将嵌入保存到adata
            
        返回:
            如果inplace为True，返回负对数似然；否则返回嵌入和负对数似然
        """
        from scETM.batch_sampler import CellSampler
        
        assert adata.n_vars == self.n_genes, f"数据集特征数 ({adata.n_vars}) 与模型特征数 ({self.n_genes}) 不匹配"
        
        # 初始化返回值
        nlls = []
        embs = {}
        
        # 检查批次信息
        has_batch = batch_col in adata.obs
        if self.input_batch_id and has_batch and adata.obs[batch_col].nunique() != self.n_batches:
            print(f"警告：数据集包含 {adata.obs[batch_col].nunique()} 个批次，而模型期望 {self.n_batches} 个批次")
        
        # 准备嵌入名称
        if emb_names is None:
            emb_names = self.emb_names
        if isinstance(emb_names, str):
            emb_names = [emb_names]
            
        # 初始化嵌入字典
        for name in emb_names:
            embs[name] = []
        
        # 设置为评估模式
        self.eval()
        
        # 创建数据采样器
        sampler = CellSampler(adata, batch_size=batch_size, sample_batch_id=has_batch, n_epochs=1, batch_col=batch_col, shuffle=False)
        
        # 收集嵌入和NLL
        with torch.no_grad():
            for data_dict in sampler:
                data_dict = {k: v.to(self.device) for k, v in data_dict.items()}
                
                # 前向传播
                fwd_dict = self.forward(data_dict['cells'], data_dict.get('batch_indices', None), {'decode': True})
                
                # 收集嵌入
                for name in emb_names:
                    embs[name].append(fwd_dict[name].cpu())
                
                # 收集NLL
                if 'nll' in fwd_dict:
                    nlls.append(fwd_dict['nll'].cpu().item())
        
        # 合并嵌入
        embs = {name: torch.cat(embs[name], dim=0).numpy() for name in emb_names}
        
        # 计算总NLL
        nll = sum(nlls) / adata.n_obs if nlls else None
        
        # 返回结果
        if inplace:
            adata.obsm.update(embs)
            return nll
        else:
            return embs, nll

    def get_all_embeddings_and_nll(self, 
                                  adata, 
                                  batch_size: int = 2000, 
                                  emb_names: Union[str, List[str], None] = None, 
                                  batch_col: str = 'batch_indices', 
                                  inplace: bool = True) -> Union[Union[None, float], Tuple[Dict[str, np.ndarray], Union[None, float]]]:
        """
        获取所有嵌入（细胞、基因、主题）和负对数似然
        参考自scETM/models/scETM.py中的get_all_embeddings_and_nll方法
        
        参数:
            adata: AnnData对象
            batch_size: 批处理大小
            emb_names: 要返回的嵌入名称
            batch_col: 批次列名
            inplace: 是否将嵌入保存到adata
            
        返回:
            如果inplace为True，返回负对数似然；否则返回嵌入和负对数似然
        """
        # 获取细胞嵌入和NLL
        result = self.get_cell_embeddings_and_nll(
            adata, 
            batch_size=batch_size, 
            emb_names=emb_names, 
            batch_col=batch_col, 
            inplace=inplace
        )
        
        # 获取基因和主题嵌入
        if inplace:
            # 保存到adata
            adata.varm['rho'] = self.decoder.rho.T.detach().cpu().numpy()
            adata.uns['alpha'] = self.decoder.alpha.detach().cpu().numpy()
            return result
        else:
            # 返回所有嵌入
            embs, nll = result
            embs['rho'] = self.decoder.rho.T.detach().cpu().numpy()
            embs['alpha'] = self.decoder.alpha.detach().cpu().numpy()
            return embs, nll

    def save(self, save_path: str) -> None:
        """
        保存模型
        
        参数:
            save_path: 保存路径
        """
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型
        torch.save({
            'model_state_dict': self.state_dict(),
            'prior_type': self.prior_type,
            'n_genes': self.n_genes,
            'n_topics': self.n_topics,
            'gene_emb_dim': self.gene_emb_dim,
            'normalize_beta': self.normalize_beta,
            'input_batch_id': self.input_batch_id,
            'enable_batch_bias': self.enable_batch_bias,
            'enable_global_bias': self.enable_global_bias,
            'n_batches': self.n_batches,
            # 对于混合先验，保存额外信息
            'prior_num_tasks': self.prior.num_tasks if self.prior_type == 'vamp' else 1,
            'prior_weights': self.prior.weights.cpu() if self.prior_type == 'vamp' else None,
            'prior_task_weight': self.prior.task_weight.cpu() if self.prior_type == 'vamp' else None,
            'prior_mu_list': [mu.cpu() for mu in self.prior.mu_list] if self.prior_type == 'vamp' else None,
            'pr_q_means': [mu.data.cpu() for mu in self.pr_q_means] if self.prior_type == 'vamp' and self.pr_q_means else None,
            'pr_q_logvars': [logvar.data.cpu() for logvar in self.pr_q_logvars] if self.prior_type == 'vamp' and self.pr_q_logvars else None,
        }, save_path)
        
        print(f"模型已保存到 {save_path}")

    @classmethod
    def load(cls, load_path: str, device=None):
        """
        加载模型
        
        参数:
            load_path: 加载路径
            device: 设备
            
        返回:
            加载的模型
        """
        # 加载保存的状态
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # 设置设备
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 创建模型实例
        model = cls(
            n_genes=checkpoint['n_genes'],
            n_topics=checkpoint['n_topics'],
            gene_emb_dim=checkpoint['gene_emb_dim'],
            normalize_beta=checkpoint['normalize_beta'],
            input_batch_id=checkpoint['input_batch_id'],
            enable_batch_bias=checkpoint['enable_batch_bias'],
            enable_global_bias=checkpoint['enable_global_bias'],
            n_batches=checkpoint['n_batches'],
            prior_type=checkpoint['prior_type'],
            device=device
        )
        
        # 加载模型参数
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 对于混合先验，恢复先验状态
        if checkpoint['prior_type'] == 'vamp':
            model.prior.num_tasks = checkpoint['prior_num_tasks']
            model.prior.weights = checkpoint['prior_weights'].to(device)
            model.prior.task_weight = checkpoint['prior_task_weight'].to(device)
            model.prior.mu_list = [mu.to(device) for mu in checkpoint['prior_mu_list']]
            
            if checkpoint['pr_q_means'] is not None:
                model.pr_q_means = [nn.Parameter(mu.to(device), requires_grad=False) for mu in checkpoint['pr_q_means']]
                model.pr_q_logvars = [nn.Parameter(logvar.to(device), requires_grad=False) for logvar in checkpoint['pr_q_logvars']]
        
        return model

    def generate_x(self, n_samples: int = 25) -> torch.Tensor:
        """
        生成样本
        参考自vae/model/boost.py中的generate_x方法
        
        参数:
            n_samples: 生成样本数量
            
        返回:
            生成的样本
        """
        self.eval()
        
        if self.prior_type == 'vamp' and len(self.pr_q_means) > 0:
            # 从混合先验中采样
            n_comp = len(self.pr_q_means)
            weights = (self.prior.weights[:n_comp] / self.prior.task_weight[:n_comp]).cpu().numpy()
            
            # 随机选择组件
            idx = np.random.choice(n_comp, size=n_samples, replace=True, p=weights / weights.sum())
            
            # 从选定的组件中采样
            z_samples = []
            for i in idx:
                z_mu = self.pr_q_means[i]
                z_logvar = self.pr_q_logvars[i]
                z_sample = self._reparameterize(z_mu, z_logvar)
                z_samples.append(z_sample)
            
            # 合并样本
            z_samples = torch.cat(z_samples, dim=0)
        else:
            # 使用标准正态分布
            z_samples = torch.randn(n_samples, self.n_topics, device=self.device)
        
        # 计算主题分布
        theta = F.softmax(z_samples, dim=-1)
        
        # 解码
        recon_log = self.decoder(theta)
        
        # 返回生成的样本
        return torch.exp(recon_log)  # 将对数概率转换为概率
# 在 models/cl_scetm.py 中添加以下方法

def train_new_component_boovae_style(self, 
                                   X_opt: torch.Tensor,
                                   max_steps: int = 30000,
                                   lbd: float = 1.0) -> float:
    """
    使用BooVAE方法训练新的伪输入组件
    
    参数:
        X_opt: 当前task的优化数据
        max_steps: 最大训练步数
        lbd: BooVAE的lambda参数
        
    返回:
        训练完成后的组件权重
    """
    
    if self.prior_type != 'vamp':
        raise ValueError("只有vamp先验支持BooVAE风格训练")
    
    # 缓存最优先验（基于当前数据）
    self._cache_optimal_prior(X_opt)
    
    # 初始化新的伪输入参数
    self.h_mu = nn.Parameter(
        torch.randn(1, self.n_genes, device=self.device) * 0.1
    )
    h_optimizer = torch.optim.Adam([self.h_mu], lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        h_optimizer, patience=100, factor=0.5
    )
    
    # BooVAE训练参数
    loss_hist = [1e10]
    min_steps = 2000
    loss_threshold = 1e-2
    
    print("开始BooVAE风格的组件训练...")
    
    # 获取当前先验参数
    current_means, current_logvars, current_weights = self._get_current_prior_params()
    
    # BooVAE训练循环
    for boost_ep in range(1, max_steps + 1):
        h_optimizer.zero_grad()
        
        # 计算BooVAE的boosting损失
        loss, entropy, log_mean_q, log_p_z = self._calculate_boosting_loss_boovae(
            current_means, current_logvars, current_weights, lbd
        )
        
        loss.backward()
        loss_hist.append(loss.item())
        h_optimizer.step()
        scheduler.step(loss)
        
        # BooVAE的停止条件
        if (abs(loss_hist[-1] - loss_hist[-2]) < loss_threshold and 
            boost_ep > min_steps):
            print(f'组件训练完成，共 {boost_ep} 步')
            break
        
        if boost_ep % 1000 == 0:
            print(f"Step {boost_ep}, Loss: {loss.item():.6f}")
    
    # 计算最优权重
    optimal_weight = self._compute_optimal_weight_boovae()
    
    return optimal_weight

def _cache_optimal_prior(self, X_opt: torch.Tensor):
    """缓存最优先验参数"""
    print("缓存最优先验...")
    
    with torch.no_grad():
        means, logvars = self.encoder(X_opt)
    
    self._cached_optimal_means = means
    self._cached_optimal_logvars = logvars
    self._cached_optimal_weights = torch.ones(means.shape[0], device=self.device) / means.shape[0]

def _get_current_prior_params(self):
    """获取当前先验参数"""
    if len(self.pr_q_means) > 0:
        current_means = torch.cat([mu for mu in self.pr_q_means], dim=0)
        current_logvars = torch.cat([logvar for logvar in self.pr_q_logvars], dim=0)
        current_weights = self.prior.weights[:len(self.pr_q_means)].clone()
    else:
        # 如果还没有组件，使用标准正态分布
        current_means = torch.zeros(1, self.n_topics, device=self.device)
        current_logvars = torch.zeros(1, self.n_topics, device=self.device)
        current_weights = torch.ones(1, device=self.device)
    
    return current_means, current_logvars, current_weights

def _calculate_boosting_loss_boovae(self, pr_means, pr_logvars, pr_w, lbd=1):
    """计算BooVAE风格的boosting损失"""
    
    initial_training = self.training
    if initial_training:
        self.eval()
    
    # 编码新的伪输入
    z_q_mean, z_q_logvar = self.encoder(self.h_mu)
    z_sample = self._reparameterize(z_q_mean, z_q_logvar)
    
    # 计算熵正则化项
    entropy = 0.5 * (1 + math.log(2*math.pi) + z_q_logvar).sum()
    
    # 计算最优先验的对数概率
    log_mean_q = self._compute_optimal_prior_logprob(z_sample)
    
    # 计算当前先验的对数概率
    log_p_z = self._log_gaussian_mixture(z_sample, pr_means, pr_logvars, pr_w)
    
    # BooVAE损失
    loss = -entropy - lbd * log_mean_q + lbd * log_p_z
    
    if initial_training:
        self.train()
    
    return loss, entropy, log_mean_q, log_p_z

def _compute_optimal_prior_logprob(self, z_sample):
    """计算最优先验的对数概率"""
    return self._log_gaussian_mixture(
        z_sample,
        self._cached_optimal_means,
        self._cached_optimal_logvars,
        self._cached_optimal_weights
    )

def _log_gaussian_mixture(self, z_sample, means, logvars, weights):
    """计算高斯混合分布的对数概率"""
    z_sample = z_sample.unsqueeze(1)  # [batch_size, 1, latent_dim]
    means = means.unsqueeze(0)        # [1, n_components, latent_dim]
    logvars = logvars.unsqueeze(0)    # [1, n_components, latent_dim]
    
    weights = weights / weights.sum()
    log_w = torch.log(weights).unsqueeze(0).to(z_sample.device)
    
    # 计算高斯对数概率
    log_comps = -0.5 * (math.log(2.0 * math.pi) + logvars + 
                        torch.pow(z_sample - means, 2) / (torch.exp(logvars) + 1e-5))
    log_comps = log_comps.sum(dim=2)  # 对特征维度求和
    
    log_probs = torch.logsumexp(log_comps + log_w, dim=1)
    return log_probs.mean()

def _compute_optimal_weight_boovae(self) -> float:
    """使用BooVAE方法计算最优权重"""
    max_iter = 1000
    tol = 1e-4
    lr = 0.1
    
    w = torch.tensor(0.5, device=self.device)
    
    print("计算最优权重...")
    
    for i in range(max_iter):
        grad = self._compute_alpha_gradient(w)
        w -= lr / (i + 1.) * grad
        w = torch.clamp(w, 1e-4, 0.99)
        
        if i > 20 and abs(grad.item()) < tol:
            break
    
    return w.item()

def _compute_alpha_gradient(self, alpha):
    """计算权重梯度"""
    with torch.no_grad():
        # 从新组件采样
        z_q_mean, z_q_logvar = self.encoder(self.h_mu)
        h_sample = self._reparameterize(z_q_mean, z_q_logvar)
        
        # 从现有先验采样
        if len(self.pr_q_means) > 0:
            idx = np.random.randint(len(self.pr_q_means))
            p_mu = self.pr_q_means[idx]
            p_logvar = self.pr_q_logvars[idx]
            p_sample = self._reparameterize(p_mu, p_logvar)
        else:
            p_sample = torch.randn_like(h_sample)
        
        # 计算梯度项
        grad_h = self._compute_gradient_term(h_sample, alpha)
        grad_p = self._compute_gradient_term(p_sample, alpha)
    
    return grad_h - grad_p

def _compute_gradient_term(self, z_sample, alpha):
    """计算梯度项"""
    with torch.no_grad():
        log_q_z = self._compute_optimal_prior_logprob(z_sample)
        
        # 新组件概率
        z_q_mean, z_q_logvar = self.encoder(self.h_mu)
        log_h_z = -0.5 * (math.log(2.0 * math.pi) + z_q_logvar + 
                         torch.pow(z_sample - z_q_mean, 2) / (torch.exp(z_q_logvar) + 1e-5))
        log_h_z = log_h_z.sum(dim=1).mean()
        
        # 现有先验概率
        if len(self.pr_q_means) > 0:
            current_means, current_logvars, current_weights = self._get_current_prior_params()
            log_p_z = self._log_gaussian_mixture(z_sample, current_means, current_logvars, current_weights)
        else:
            log_p_z = -0.5 * (math.log(2.0 * math.pi) + torch.pow(z_sample, 2)).sum(dim=1).mean()
        
        log_h_z += torch.log(alpha)
        log_p_z += torch.log(1. - alpha)
        
        comb_log_p = torch.logsumexp(torch.stack([log_p_z, log_h_z]), 0)
    
    return comb_log_p - log_q_z

def accept_new_component_with_weight(self, weight: float):
    """接受新组件并设置权重"""
    if not hasattr(self, 'h_mu'):
        raise ValueError("没有待接受的组件")
    
    # 编码新组件
    with torch.no_grad():
        mu, logvar = self.encoder(self.h_mu)
    
    # 添加到先验
    self.prior.add_component(self.h_mu.data.clone(), alpha=weight)
    self.pr_q_means.append(nn.Parameter(mu.data.clone(), requires_grad=False))
    self.pr_q_logvars.append(nn.Parameter(logvar.data.clone(), requires_grad=False))
    
    # 清理
    del self.h_mu
    print(f"新组件已接受，权重: {weight:.6f}")

def update_existing_component_weights(self, X_opt: torch.Tensor, n_steps: int = 100):
    """更新现有组件权重"""
    if len(self.prior.mu_list) <= 1:
        return
    
    print("更新现有组件权重...")
    
    current_weights = self.prior.weights[:len(self.pr_q_means)].clone().requires_grad_(True)
    weight_optimizer = torch.optim.Adam([current_weights], lr=0.001)
    
    for step in range(n_steps):
        weight_optimizer.zero_grad()
        
        # 计算权重优化损失
        loss = self._compute_weight_optimization_loss(X_opt, current_weights)
        
        loss.backward()
        weight_optimizer.step()
        
        # 确保权重有效
        with torch.no_grad():
            current_weights.data = torch.clamp(current_weights.data, 0.0)
            current_weights.data = current_weights.data / current_weights.data.sum()
    
    # 更新权重
    self.prior.weights[:len(self.pr_q_means)] = current_weights.data
    print(f"权重更新完成: {self.prior.weights[:len(self.pr_q_means)]}")

def _compute_weight_optimization_loss(self, X_opt, weights):
    """计算权重优化损失"""
    with torch.no_grad():
        # 编码数据
        z_mu, z_logvar = self.encoder(X_opt)
        z_sample = self._reparameterize(z_mu, z_logvar)
    
    # 计算当前混合先验的对数概率
    current_means = torch.cat([mu for mu in self.pr_q_means], dim=0)
    current_logvars = torch.cat([logvar for logvar in self.pr_q_logvars], dim=0)
    
    log_prior = self._log_gaussian_mixture(z_sample, current_means, current_logvars, weights)
    
    # 最大化对数似然（最小化负对数似然）
    return -log_prior
import json
import os
from typing import Dict, Any
import pickle


def save_model_state(self, 
                    save_dir: str, 
                    task_id: int = None,
                    save_components_separately: bool = True) -> str:
    """
    保存模型状态
    
    参数:
        save_dir: 保存目录
        task_id: 任务ID（用于文件命名）
        save_components_separately: 是否单独保存每个组件
        
    返回:
        保存路径
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 构造文件名
    if task_id is not None:
        save_prefix = f"cl_scetm_task_{task_id}"
    else:
        save_prefix = f"cl_scetm_step_{len(self.pr_q_means)}"
    
    save_path = os.path.join(save_dir, f"{save_prefix}.pt")
    
    print(f"保存模型状态到: {save_path}")
    
    # 准备保存的状态字典
    state_dict = {
        # 基本模型参数
        'model_state_dict': self.state_dict(),
        
        # 模型配置
        'model_config': {
            'n_genes': self.n_genes,
            'n_topics': self.n_topics,
            'hidden_sizes': self.hidden_sizes,
            'prior_type': self.prior_type,
            'device': str(self.device),
        },
        
        # 先验相关参数
        'prior_info': {
            'n_components': len(self.pr_q_means),
            'weights': self.prior.weights.cpu().detach().numpy().tolist() if hasattr(self.prior, 'weights') else [],
        },
        
        # 伪输入参数
        'pseudo_inputs': [],
        'component_means': [],
        'component_logvars': [],
        
        # 训练状态
        'current_task': getattr(self, 'current_task', 0),
        'task_history': getattr(self, 'task_history', []),
        
        # 时间戳
        'timestamp': datetime.now().isoformat(),
    }
    
    # 保存伪输入和编码后的参数
    if hasattr(self.prior, 'mu_list') and len(self.prior.mu_list) > 0:
        for i, pseudo_input in enumerate(self.prior.mu_list):
            state_dict['pseudo_inputs'].append({
                'component_id': i,
                'pseudo_input': pseudo_input.cpu().detach().numpy(),
                'weight': self.prior.weights[i].item() if i < len(self.prior.weights) else 0.0,
            })
    
    # 保存编码后的均值和方差
    for i, (mean, logvar) in enumerate(zip(self.pr_q_means, self.pr_q_logvars)):
        state_dict['component_means'].append(mean.cpu().detach().numpy())
        state_dict['component_logvars'].append(logvar.cpu().detach().numpy())
    
    # 保存主要状态
    torch.save(state_dict, save_path)
    
    # 保存配置信息（JSON格式，便于查看）
    config_path = os.path.join(save_dir, f"{save_prefix}_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'model_config': state_dict['model_config'],
            'prior_info': state_dict['prior_info'],
            'current_task': state_dict['current_task'],
            'timestamp': state_dict['timestamp'],
        }, f, indent=2)
    
    # 单独保存每个组件（可选）
    if save_components_separately and len(self.pr_q_means) > 0:
        components_dir = os.path.join(save_dir, f"{save_prefix}_components")
        os.makedirs(components_dir, exist_ok=True)
        
        self._save_components_separately(components_dir, task_id)
    
    print(f"模型状态已保存: {save_path}")
    print(f"配置信息已保存: {config_path}")
    
    return save_path

def _save_components_separately(self, components_dir: str, task_id: int = None):
    """单独保存每个组件"""
    
    for i in range(len(self.pr_q_means)):
        component_data = {
            'component_id': i,
            'task_id': task_id,
            'pseudo_input': self.prior.mu_list[i].cpu().detach().numpy(),
            'encoded_mean': self.pr_q_means[i].cpu().detach().numpy(),
            'encoded_logvar': self.pr_q_logvars[i].cpu().detach().numpy(),
            'weight': self.prior.weights[i].item() if i < len(self.prior.weights) else 0.0,
            'timestamp': datetime.now().isoformat(),
        }
        
        component_path = os.path.join(components_dir, f"component_{i}.pkl")
        with open(component_path, 'wb') as f:
            pickle.dump(component_data, f)
    
    print(f"组件单独保存到: {components_dir}")

def load_model_state(self, 
                    load_path: str, 
                    strict: bool = True,
                    load_components: bool = True) -> Dict[str, Any]:
    """
    加载模型状态
    
    参数:
        load_path: 模型文件路径
        strict: 是否严格匹配参数
        load_components: 是否加载组件信息
        
    返回:
        加载的状态信息
    """
    
    print(f"从 {load_path} 加载模型状态...")
    
    # 加载状态字典
    checkpoint = torch.load(load_path, map_location=self.device)
    
    # 加载模型参数
    self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # 恢复先验组件
    if load_components and 'pseudo_inputs' in checkpoint:
        self._restore_components_from_checkpoint(checkpoint)
    
    # 恢复训练状态
    if 'current_task' in checkpoint:
        self.current_task = checkpoint['current_task']
    if 'task_history' in checkpoint:
        self.task_history = checkpoint['task_history']
    
    print("模型状态加载完成")
    print(f"当前任务: {getattr(self, 'current_task', 'N/A')}")
    print(f"组件数量: {len(self.pr_q_means)}")
    
    return checkpoint

def _restore_components_from_checkpoint(self, checkpoint: Dict[str, Any]):
    """从checkpoint恢复组件"""
    
    # 清空现有组件
    self.pr_q_means.clear()
    self.pr_q_logvars.clear()
    
    # 恢复伪输入
    pseudo_inputs = []
    weights = []
    
    for comp_data in checkpoint['pseudo_inputs']:
        pseudo_input = torch.FloatTensor(comp_data['pseudo_input']).to(self.device)
        pseudo_inputs.append(pseudo_input)
        weights.append(comp_data['weight'])
    
    # 重建先验
    if len(pseudo_inputs) > 0:
        # 重新初始化先验
        self.prior = VampPrior(
            k=len(pseudo_inputs),
            decoder=self.decoder,
            device=self.device
        )
        
        # 设置伪输入
        for i, pseudo_input in enumerate(pseudo_inputs):
            if i == 0:
                self.prior.u = nn.Parameter(pseudo_input.unsqueeze(0))
            else:
                self.prior.add_component(pseudo_input, alpha=weights[i])
        
        # 设置权重
        if weights:
            self.prior.weights = nn.Parameter(torch.FloatTensor(weights).to(self.device))
    
    # 恢复编码后的参数
    if 'component_means' in checkpoint and 'component_logvars' in checkpoint:
        for mean_data, logvar_data in zip(checkpoint['component_means'], checkpoint['component_logvars']):
            mean = torch.FloatTensor(mean_data).to(self.device)
            logvar = torch.FloatTensor(logvar_data).to(self.device)
            
            self.pr_q_means.append(nn.Parameter(mean, requires_grad=False))
            self.pr_q_logvars.append(nn.Parameter(logvar, requires_grad=False))

def save_checkpoint_during_training(self, 
                                  checkpoint_dir: str, 
                                  task_id: int,
                                  component_id: int = None) -> str:
    """
    训练过程中保存checkpoint
    
    参数:
        checkpoint_dir: checkpoint目录
        task_id: 任务ID
        component_id: 组件ID（如果在组件训练中）
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if component_id is not None:
        checkpoint_name = f"task_{task_id}_component_{component_id}_checkpoint.pt"
    else:
        checkpoint_name = f"task_{task_id}_checkpoint.pt"
    
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    return self.save_model_state(checkpoint_dir, task_id)

def export_components_for_analysis(self, export_dir: str) -> str:
    """
    导出组件用于分析
    
    参数:
        export_dir: 导出目录
        
    返回:
        导出路径
    """
    
    os.makedirs(export_dir, exist_ok=True)
    
    # 导出所有组件的详细信息
    analysis_data = {
        'model_info': {
            'n_genes': self.n_genes,
            'n_topics': self.n_topics,
            'n_components': len(self.pr_q_means),
            'prior_type': self.prior_type,
        },
        'components': []
    }
    
    for i in range(len(self.pr_q_means)):
        # 计算组件的主题分布
        with torch.no_grad():
            pseudo_input = self.prior.mu_list[i]
            theta = F.softmax(self.pr_q_means[i], dim=-1)
            
            # 找到最重要的主题
            top_topics = torch.topk(theta.squeeze(), k=min(10, self.n_topics))
            
            component_info = {
                'component_id': i,
                'weight': self.prior.weights[i].item() if i < len(self.prior.weights) else 0.0,
                'pseudo_input_stats': {
                    'mean': pseudo_input.mean().item(),
                    'std': pseudo_input.std().item(),
                    'min': pseudo_input.min().item(),
                    'max': pseudo_input.max().item(),
                },
                'topic_distribution': {
                    'top_topic_indices': top_topics.indices.cpu().numpy().tolist(),
                    'top_topic_values': top_topics.values.cpu().numpy().tolist(),
                    'entropy': -(theta * torch.log(theta + 1e-10)).sum().item(),
                },
                'latent_stats': {
                    'mean_norm': torch.norm(self.pr_q_means[i]).item(),
                    'logvar_mean': self.pr_q_logvars[i].mean().item(),
                }
            }
            
            analysis_data['components'].append(component_info)
    
    # 保存分析数据
    analysis_path = os.path.join(export_dir, 'components_analysis.json')
    with open(analysis_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    # 保存原始数据
    raw_data_path = os.path.join(export_dir, 'components_raw_data.pkl')
    raw_data = {
        'pseudo_inputs': [mu.cpu().detach().numpy() for mu in self.prior.mu_list],
        'encoded_means': [mean.cpu().detach().numpy() for mean in self.pr_q_means],
        'encoded_logvars': [logvar.cpu().detach().numpy() for logvar in self.pr_q_logvars],
        'weights': self.prior.weights.cpu().detach().numpy() if hasattr(self.prior, 'weights') else [],
    }
    
    with open(raw_data_path, 'wb') as f:
        pickle.dump(raw_data, f)
    
    print(f"组件分析数据已导出到: {export_dir}")
    return export_dir

@classmethod
def load_from_checkpoint(cls, 
                        load_path: str, 
                        device: str = 'cuda') -> 'CL_scETM':
    """
    从checkpoint加载模型
    
    参数:
        load_path: checkpoint路径
        device: 设备
        
    返回:
        加载的模型实例
    """
    
    print(f"从checkpoint加载模型: {load_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(load_path, map_location=device)
    
    # 获取模型配置
    config = checkpoint['model_config']
    
    # 创建模型实例
    model = cls(
        n_genes=config['n_genes'],
        n_topics=config['n_topics'],
        hidden_sizes=config.get('hidden_sizes', [256, 256]),
        prior_type=config.get('prior_type', 'vamp'),
        device=device
    )
    
    # 加载状态
    model.load_model_state(load_path, strict=True, load_components=True)
    
    return model