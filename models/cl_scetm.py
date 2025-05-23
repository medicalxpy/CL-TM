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
from priors.standard_prior import log_Normal_standard, get_kl
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
        prior_type: 先验类型，'standard'或'mixture'
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
        if prior_type == 'mixture':
            # 初始化混合先验相关参数
            self.n_pseudoinputs = n_pseudoinputs
            self.pseudoinputs_mean = pseudoinputs_mean
            self.pseudoinputs_std = pseudoinputs_std
            
            # 获取示例数据，用于初始化伪输入
            self.X_opt = X_opt
            
            # 如果提供了示例数据，使用其平均值初始化第一个伪输入
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
        
        # KL散度
        if self.prior_type == 'standard':
            # 标准先验
            kl = get_kl(fwd_dict['mu_q_delta'], fwd_dict['logsigma_q_delta'])
        else:
            # 混合先验
            kl = self._calculate_mixture_kl(fwd_dict)
        
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

    def _calculate_mixture_kl(self, fwd_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算混合先验的KL散度
        参考自vae/model/boost.py中的log_p_z和calculate_boosting_loss方法
        
        参数:
            fwd_dict: 前向传播结果字典
            
        返回:
            KL散度
        """
        # 获取编码器输出
        z = fwd_dict['delta']
        mu = fwd_dict['mu_q_delta']
        logsigma = fwd_dict['logsigma_q_delta']
        
        # 计算log q(z|x)，这是编码器输出的后验分布的对数概率
        log_qz = -0.5 * (1 + 2 * logsigma - mu.pow(2) - torch.exp(2 * logsigma)).sum(-1)
        
        # 计算log p(z)，这是先验分布的对数概率
        log_pz = self._log_p_z_mixture(z)
        
        # 计算KL散度 = log q(z|x) - log p(z)
        kl = log_qz - log_pz
        
        return kl

    def _log_p_z_mixture(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算混合先验的对数概率
        参考自vae/model/boost.py中的log_p_z方法
        
        参数:
            z: 隐变量
            
        返回:
            对数概率
        """
        # 如果没有组件，使用标准正态分布作为后备
        if len(self.prior.mu_list) == 0:
            return log_Normal_standard(z, dim=1)
        
        # 获取所有伪输入
        pseudo_mus = []
        pseudo_logvars = []
        
        # 如果已有编码好的伪输入分布，直接使用
        if len(self.pr_q_means) > 0:
            for mu, logvar in zip(self.pr_q_means, self.pr_q_logvars):
                pseudo_mus.append(mu)
                pseudo_logvars.append(logvar)
        else:
            # 否则，对每个伪输入重新计算潜变量分布
            for i, pseudo_input in enumerate(self.prior.mu_list):
                pseudo_input = pseudo_input.to(self.device)
                with torch.no_grad():
                    mu, logvar = self.encoder(pseudo_input)
                pseudo_mus.append(mu)
                pseudo_logvars.append(logvar)
        
        # 拼接所有均值和方差
        pseudo_mus = torch.cat(pseudo_mus, dim=0)  # [n_components, n_topics]
        pseudo_logvars = torch.cat(pseudo_logvars, dim=0)  # [n_components, n_topics]
        
        # 准备计算
        z_expand = z.unsqueeze(1)  # [batch_size, 1, n_topics]
        pseudo_mus = pseudo_mus.unsqueeze(0)  # [1, n_components, n_topics]
        pseudo_logvars = pseudo_logvars.unsqueeze(0)  # [1, n_components, n_topics]
        
        # 计算z在每个组件下的对数概率
        log_p_z_given_c = -0.5 * (
            math.log(2 * math.pi) 
            + pseudo_logvars
            + (z_expand - pseudo_mus).pow(2) / torch.exp(pseudo_logvars)
        ).sum(dim=2)  # [batch_size, n_components]
        
        # 获取组件权重
        weights = self.prior.weights.to(self.device) / self.prior.num_tasks
        log_weights = torch.log(weights).unsqueeze(0)  # [1, n_components]
        
        # 计算混合分布的对数概率
        log_p_z = torch.logsumexp(log_p_z_given_c + log_weights, dim=1)  # [batch_size]
        
        return log_p_z

    def add_component(self, new_pseudo_input: Optional[torch.Tensor] = None, alpha: Optional[float] = None) -> None:
        """
        向混合先验添加新组件
        参考自vae/model/boost.py中的add_component方法
        
        参数:
            new_pseudo_input: 新伪输入，如不提供则使用默认值
            alpha: 新组件的权重，如不提供则使用默认值
        """
        if self.prior_type != 'mixture':
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
        if self.prior_type != 'mixture':
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
        if self.prior_type != 'mixture':
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
            'prior_num_tasks': self.prior.num_tasks if self.prior_type == 'mixture' else 1,
            'prior_weights': self.prior.weights.cpu() if self.prior_type == 'mixture' else None,
            'prior_task_weight': self.prior.task_weight.cpu() if self.prior_type == 'mixture' else None,
            'prior_mu_list': [mu.cpu() for mu in self.prior.mu_list] if self.prior_type == 'mixture' else None,
            'pr_q_means': [mu.data.cpu() for mu in self.pr_q_means] if self.prior_type == 'mixture' and self.pr_q_means else None,
            'pr_q_logvars': [logvar.data.cpu() for logvar in self.pr_q_logvars] if self.prior_type == 'mixture' and self.pr_q_logvars else None,
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
        if checkpoint['prior_type'] == 'mixture':
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
        
        if self.prior_type == 'mixture' and len(self.pr_q_means) > 0:
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