#!/usr/bin/env python3
"""
CL-scETM 增量训练示例
结合BooVAE的动态先验方法和scETM的topic model，实现incremental learning
"""

import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import logging
from pathlib import Path
import time

# 导入自定义模块
from models.cl_scetm import CL_scETM
from trainers.model_trainer import ModelTrainer
from data.preprocess import read_data, preprocess_data, setup_anndata
from data.scETMdataset import create_data_loader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_incremental_datasets(adata, n_datasets=3, split_by='cell_type', min_cells_per_dataset=500):
    """
    将单个数据集分割为多个子数据集，模拟增量学习场景
    
    参数:
        adata: 原始数据集
        n_datasets: 要分割的数据集数量
        split_by: 分割依据 ('cell_type', 'batch', 'random')
        min_cells_per_dataset: 每个数据集的最小细胞数
    
    返回:
        List[AnnData]: 分割后的数据集列表
    """
    logger.info(f"准备将数据集分割为 {n_datasets} 个子数据集，分割方式: {split_by}")
    
    datasets = []
    
    if split_by == 'cell_type' and 'cell_type' in adata.obs:
        # 按细胞类型分割
        cell_types = adata.obs['cell_type'].cat.categories
        logger.info(f"发现 {len(cell_types)} 种细胞类型: {list(cell_types)}")
        
        # 选择细胞数量较多的类型
        type_counts = adata.obs['cell_type'].value_counts()
        selected_types = type_counts[type_counts >= min_cells_per_dataset].index[:n_datasets]
        
        for i, cell_type in enumerate(selected_types):
            subset = adata[adata.obs['cell_type'] == cell_type].copy()
            subset.obs['dataset_id'] = i
            datasets.append(subset)
            logger.info(f"数据集 {i}: {cell_type}, {subset.n_obs} 个细胞")
            
    elif split_by == 'batch' and 'batch' in adata.obs:
        # 按批次分割
        batches = adata.obs['batch'].cat.categories
        logger.info(f"发现 {len(batches)} 个批次: {list(batches)}")
        
        for i, batch in enumerate(batches[:n_datasets]):
            subset = adata[adata.obs['batch'] == batch].copy()
            subset.obs['dataset_id'] = i
            datasets.append(subset)
            logger.info(f"数据集 {i}: batch {batch}, {subset.n_obs} 个细胞")
            
    else:
        # 随机分割
        logger.info("使用随机分割")
        indices = np.random.permutation(adata.n_obs)
        cells_per_dataset = len(indices) // n_datasets
        
        for i in range(n_datasets):
            start_idx = i * cells_per_dataset
            if i == n_datasets - 1:  # 最后一个数据集包含剩余所有细胞
                end_idx = len(indices)
            else:
                end_idx = (i + 1) * cells_per_dataset
            
            subset_indices = indices[start_idx:end_idx]
            subset = adata[subset_indices].copy()
            subset.obs['dataset_id'] = i
            datasets.append(subset)
            logger.info(f"数据集 {i}: 随机分割, {subset.n_obs} 个细胞")
    
    return datasets

def create_incremental_model(n_genes, device):
    """
    创建用于增量学习的CL_scETM模型
    """
    logger.info("创建CL_scETM模型，使用vamp先验进行增量学习")
    
    model = CL_scETM(
        n_genes=n_genes,
        n_topics=50,                    # 主题数量
        hidden_sizes=[128],             # 编码器隐藏层
        gene_emb_dim=400,              # 基因嵌入维度
        bn=True,                       # 批标准化
        dropout_prob=0.1,              # dropout概率
        n_batches=1,                   # 简化：不考虑批次效应
        normalize_beta=False,          # 不标准化beta
        input_batch_id=False,          # 不使用批次ID作为输入
        enable_batch_bias=False,       # 不使用批次偏置
        enable_global_bias=False,      # 不使用全局偏置
        prior_type='vamp',             # 使用vamp先验进行增量学习
        n_pseudoinputs=1,              # 初始伪输入数量
        pseudoinputs_mean=0.0,         # 伪输入初始化均值
        pseudoinputs_std=0.1,          # 伪输入初始化标准差
        device=device
    )
    
    return model

def main():
    """
    主训练流程
    """
    logger.info("="*60)
    logger.info("CL-scETM 增量训练示例开始")
    logger.info("="*60)
    
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 数据路径 - 你需要根据实际情况修改
    data_path = '/volume1/home/pxie/data/PBMC.h5ad'  # 从test.ipynb中获取
    save_dir = './results/incremental_training'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 加载和预处理数据
    logger.info("步骤1: 加载和预处理数据")
    try:
        adata = read_data(data_path)
        logger.info(f"原始数据: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")
        
        # 预处理
        adata = preprocess_data(
            adata,
            normalize=True,
            log_transform=True,
            scale=False,
            min_cells=3,
            min_genes=200,
            hvg_selection=True,
            n_top_genes=2000  # 减少基因数量以加快训练
        )
        
        # 设置数据格式
        adata = setup_anndata(adata, batch_col='batch', cell_type_col='cell_type')
        logger.info(f"预处理后数据: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        logger.info("使用模拟数据进行演示")
        
        # 创建模拟数据
        n_cells, n_genes = 5000, 2000
        X = np.random.negative_binomial(5, 0.3, (n_cells, n_genes)).astype(np.float32)
        
        # 创建模拟的细胞类型
        cell_types = ['T_cell', 'B_cell', 'NK_cell', 'Monocyte', 'DC']
        obs = pd.DataFrame({
            'cell_type': np.random.choice(cell_types, n_cells),
            'batch': np.random.choice(['batch1', 'batch2', 'batch3'], n_cells)
        })
        obs['cell_type'] = obs['cell_type'].astype('category')
        obs['batch'] = obs['batch'].astype('category')
        
        var = pd.DataFrame(index=[f'Gene_{i}' for i in range(n_genes)])
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        logger.info(f"模拟数据: {adata.n_obs} 个细胞, {adata.n_vars} 个基因")
    
    # 2. 准备增量数据集
    logger.info("步骤2: 准备增量数据集")
    datasets = prepare_incremental_datasets(
        adata, 
        n_datasets=3,  # 分成3个数据集
        split_by='cell_type',  # 按细胞类型分割
        min_cells_per_dataset=500
    )
    
    if len(datasets) < 2:
        logger.warning("数据集数量不足，使用随机分割")
        datasets = prepare_incremental_datasets(
            adata, 
            n_datasets=3, 
            split_by='random'
        )
    
    logger.info(f"成功准备 {len(datasets)} 个增量数据集")
    
    # 3. 创建模型和训练器
    logger.info("步骤3: 创建模型和训练器")
    model = create_incremental_model(adata.n_vars, device)
    
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=5e-3,             # 学习率
        weight_decay=0.0,               # 权重衰减
        component_threshold=0.1,        # 伪输入权重阈值
        max_components_per_dataset=5,   # 每个数据集最大组件数
        seed=42
    )
    
    logger.info("模型和训练器创建完成")
    
    # 4. 增量训练
    logger.info("步骤4: 开始增量训练")
    training_start_time = time.time()
    
    for i, dataset in enumerate(datasets):
        dataset_name = f"dataset_{i}"
        logger.info(f"\n{'='*50}")
        logger.info(f"开始训练数据集 {i+1}/{len(datasets)}: {dataset_name}")
        logger.info(f"数据集大小: {dataset.n_obs} 个细胞")
        
        if 'cell_type' in dataset.obs:
            cell_type_counts = dataset.obs['cell_type'].value_counts()
            logger.info(f"细胞类型分布: {dict(cell_type_counts)}")
        
        # 进行增量训练
        dataset_history = trainer.train_incremental(
            new_adata=dataset,
            dataset_name=dataset_name,
            epochs_per_cycle=10,           # 每轮编码器训练的epoch数 (减少以加快演示)
            total_epochs=50,               # 总训练轮数 (减少以加快演示)
            batch_size=512,                # 批大小
            batch_col='batch',
            save_dir=os.path.join(save_dir, dataset_name)
        )
        
        logger.info(f"数据集 {dataset_name} 训练完成")
        logger.info(f"添加了 {dataset_history['final_components']} 个伪输入组件")
        logger.info(f"训练耗时: {dataset_history.get('duration', 0):.1f} 秒")
    
    total_training_time = time.time() - training_start_time
    
    # 5. 保存最终模型和生成报告
    logger.info("步骤5: 保存模型和生成训练报告")
    
    # 保存最终模型
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    trainer.save_model(final_model_path)
    
    # 生成训练摘要
    summary = trainer.get_training_summary()
    logger.info(f"\n{'='*50}")
    logger.info("训练摘要:")
    logger.info(f"总数据集数量: {summary['total_datasets']}")
    logger.info(f"总伪输入组件数: {summary['total_components']}")
    logger.info(f"总训练时间: {total_training_time:.1f} 秒")
    
    for i, dataset_summary in enumerate(summary['datasets']):
        logger.info(f"数据集 {i+1} ({dataset_summary['name']}):")
        logger.info(f"  - 添加组件数: {dataset_summary['components_added']}")
        logger.info(f"  - 训练时间: {dataset_summary['duration']:.1f} 秒")
        if dataset_summary['component_weights']:
            logger.info(f"  - 组件权重: {[f'{w:.3f}' for w in dataset_summary['component_weights']]}")
    
    # 保存摘要到文件
    summary_path = os.path.join(save_dir, 'training_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"训练摘要已保存到: {summary_path}")
    
    # 6. 演示模型推理
    logger.info("步骤6: 演示模型推理")
    model.eval()
    
    # 使用最后一个数据集进行推理演示
    test_dataset = datasets[-1]
    test_loader = create_data_loader(
        test_dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=0
    )
    
    with torch.no_grad():
        for data_dict in test_loader:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}
            
            # 获取细胞嵌入
            output = model(data_dict['cells'])
            delta = output['delta']  # 细胞的主题嵌入
            theta = output['theta']  # 细胞的主题分布
            
            logger.info(f"推理结果示例:")
            logger.info(f"  - Delta (topic embeddings) 形状: {delta.shape}")
            logger.info(f"  - Theta (topic distribution) 形状: {theta.shape}")
            logger.info(f"  - Delta 均值: {delta.mean().item():.4f}, 标准差: {delta.std().item():.4f}")
            logger.info(f"  - Theta 均值: {theta.mean().item():.4f}, 最大值: {theta.max().item():.4f}")
            break
    
    logger.info("="*60)
    logger.info("CL-scETM 增量训练示例完成!")
    logger.info(f"所有结果已保存到: {save_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    # 设置随机种子以保证可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()