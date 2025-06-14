"""
基于高斯先验的scETM增量学习示例

该示例展示了如何使用新的IncrementalTrainer进行增量学习：
1. 使用标准scETM训练第一个数据集
2. 保存第一个数据集训练后的编码器输出统计信息
3. 使用这些统计信息作为先验，训练后续数据集
"""

import os
import numpy as np
import torch
import anndata
import scanpy as sc
from sklearn.datasets import make_blobs
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 导入我们的训练器
from trainers.incremental_trainer import IncrementalTrainer


def create_synthetic_dataset(n_cells: int = 1000, 
                            n_genes: int = 2000, 
                            n_topics: int = 10,
                            random_state: int = None) -> anndata.AnnData:
    """
    创建合成单细胞数据集用于测试
    
    参数:
        n_cells: 细胞数量
        n_genes: 基因数量
        n_topics: 主题数量
        random_state: 随机种子
        
    返回:
        AnnData对象
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 生成主题-基因分布矩阵 (beta)
    beta = np.random.gamma(0.3, 1.0, (n_topics, n_genes))
    beta = beta / beta.sum(axis=1, keepdims=True)  # 归一化
    
    # 生成每个细胞的主题分布 (theta)
    # 使用不同的Dirichlet参数创建不同的细胞类型
    n_cell_types = 3
    cells_per_type = n_cells // n_cell_types
    
    theta_list = []
    cell_types = []
    
    for i in range(n_cell_types):
        # 每种细胞类型有不同的主题偏好
        alpha = np.ones(n_topics) * 0.1
        alpha[i * (n_topics // n_cell_types):(i + 1) * (n_topics // n_cell_types)] = 2.0
        
        theta_type = np.random.dirichlet(alpha, cells_per_type)
        theta_list.append(theta_type)
        cell_types.extend([f'CellType_{i}'] * cells_per_type)
    
    # 处理剩余的细胞
    remaining_cells = n_cells - len(cell_types)
    if remaining_cells > 0:
        alpha = np.ones(n_topics) * 0.5
        theta_remaining = np.random.dirichlet(alpha, remaining_cells)
        theta_list.append(theta_remaining)
        cell_types.extend(['CellType_Mixed'] * remaining_cells)
    
    theta = np.vstack(theta_list)
    
    # 生成基因表达数据
    # 使用泊松分布生成计数数据
    lambda_matrix = np.dot(theta, beta) * 1000  # 缩放因子
    X = np.random.poisson(lambda_matrix)
    
    # 创建AnnData对象
    adata = anndata.AnnData(X=X.astype(np.float32))
    adata.obs['cell_type'] = cell_types
    adata.obs['batch_indices'] = 0  # 单批次
    
    # 添加基因名
    adata.var_names = [f'Gene_{i}' for i in range(n_genes)]
    adata.obs_names = [f'Cell_{i}' for i in range(n_cells)]
    
    # 基本预处理
    adata.X = adata.X + 1  # 避免log(0)
    sc.pp.log1p(adata)
    
    return adata


def create_shifted_dataset(base_adata: anndata.AnnData, 
                          shift_factor: float = 0.5,
                          noise_level: float = 0.1,
                          random_state: int = None) -> anndata.AnnData:
    """
    基于现有数据集创建一个具有分布漂移的新数据集
    
    参数:
        base_adata: 基础数据集
        shift_factor: 分布漂移因子
        noise_level: 噪声水平
        random_state: 随机种子
        
    返回:
        新的AnnData对象
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 复制基础数据
    X_shifted = base_adata.X.copy()
    
    # 添加系统性漂移
    n_genes = X_shifted.shape[1]
    shift_pattern = np.sin(np.linspace(0, 4*np.pi, n_genes)) * shift_factor
    X_shifted = X_shifted + shift_pattern
    
    # 添加随机噪声
    noise = np.random.normal(0, noise_level, X_shifted.shape)
    X_shifted = X_shifted + noise
    
    # 确保非负
    X_shifted = np.maximum(X_shifted, 0.01)
    
    # 创建新的AnnData对象
    adata_shifted = anndata.AnnData(X=X_shifted.astype(np.float32))
    adata_shifted.obs['cell_type'] = [f'Shifted_{ct}' for ct in base_adata.obs['cell_type']]
    adata_shifted.obs['batch_indices'] = 1  # 不同批次
    adata_shifted.var_names = base_adata.var_names
    adata_shifted.obs_names = [f'ShiftedCell_{i}' for i in range(adata_shifted.n_obs)]
    
    return adata_shifted


def main():
    """
    主函数：演示增量学习流程
    """
    print("=" * 60)
    print("基于高斯先验的scETM增量学习示例")
    print("=" * 60)
    
    # 设置参数
    n_genes = 1000
    n_topics = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = 'results/incremental_example'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 创建合成数据集
    print("\n1. 创建合成数据集...")
    dataset1 = create_synthetic_dataset(
        n_cells=800, 
        n_genes=n_genes, 
        n_topics=n_topics,
        random_state=42
    )
    print(f"数据集1: {dataset1.n_obs} 细胞, {dataset1.n_vars} 基因")
    
    dataset2 = create_shifted_dataset(
        dataset1, 
        shift_factor=0.3, 
        noise_level=0.05,
        random_state=123
    )
    print(f"数据集2: {dataset2.n_obs} 细胞, {dataset2.n_vars} 基因")
    
    dataset3 = create_shifted_dataset(
        dataset1, 
        shift_factor=0.6, 
        noise_level=0.1,
        random_state=456
    )
    print(f"数据集3: {dataset3.n_obs} 细胞, {dataset3.n_vars} 基因")
    
    # 2. 初始化增量训练器
    print("\n2. 初始化增量训练器...")
    trainer = IncrementalTrainer(
        n_genes=n_genes,
        n_topics=n_topics,
        hidden_sizes=[128, 64],
        gene_emb_dim=200,
        learning_rate=3e-3,
        weight_decay=1e-4,
        prior_strength=1.0,
        adaptive_strength=True,
        device=device,
        seed=42
    )
    
    # 3. 训练第一个数据集
    print("\n3. 训练第一个数据集（使用标准先验）...")
    history1 = trainer.train_first_dataset(
        adata=dataset1,
        dataset_name="Dataset_1",
        n_epochs=50,
        batch_size=128,
        eval_every=10,
        save_path=os.path.join(output_dir, "model_after_dataset1.pt")
    )
    
    # 评估第一个数据集
    print("\n评估第一个数据集...")
    eval1 = trainer.evaluate_model(dataset1)
    print(f"数据集1评估: {eval1}")
    
    # 4. 增量训练第二个数据集
    print("\n4. 增量训练第二个数据集（使用高斯先验）...")
    history2 = trainer.train_incremental_dataset(
        adata=dataset2,
        dataset_name="Dataset_2",
        n_epochs=50,
        batch_size=128,
        eval_every=10,
        save_path=os.path.join(output_dir, "model_after_dataset2.pt")
    )
    
    # 评估第二个数据集
    print("\n评估第二个数据集...")
    eval2 = trainer.evaluate_model(dataset2)
    print(f"数据集2评估: {eval2}")
    
    # 测试在第一个数据集上的性能（灾难性遗忘测试）
    print("\n测试第一个数据集上的性能（灾难性遗忘测试）...")
    eval1_after_2 = trainer.evaluate_model(dataset1)
    print(f"训练数据集2后，数据集1评估: {eval1_after_2}")
    
    # 5. 增量训练第三个数据集
    print("\n5. 增量训练第三个数据集...")
    history3 = trainer.train_incremental_dataset(
        adata=dataset3,
        dataset_name="Dataset_3",
        n_epochs=50,
        batch_size=128,
        eval_every=10,
        save_path=os.path.join(output_dir, "model_after_dataset3.pt")
    )
    
    # 最终评估所有数据集
    print("\n6. 最终评估所有数据集...")
    eval1_final = trainer.evaluate_model(dataset1)
    eval2_final = trainer.evaluate_model(dataset2)
    eval3_final = trainer.evaluate_model(dataset3)
    
    print(f"最终数据集1评估: {eval1_final}")
    print(f"最终数据集2评估: {eval2_final}")
    print(f"最终数据集3评估: {eval3_final}")
    
    # 7. 获取主题表示
    print("\n7. 获取主题表示...")
    representations1 = trainer.get_topic_representations(dataset1)
    representations2 = trainer.get_topic_representations(dataset2)
    representations3 = trainer.get_topic_representations(dataset3)
    
    print(f"数据集1主题表示: theta shape = {representations1['theta'].shape}")
    print(f"数据集2主题表示: theta shape = {representations2['theta'].shape}")
    print(f"数据集3主题表示: theta shape = {representations3['theta'].shape}")
    
    # 8. 打印训练摘要
    print("\n8. 训练摘要...")
    summary = trainer.get_training_summary()
    print(f"总数据集数: {summary['total_datasets']}")
    print(f"总训练时间: {summary['total_training_time']:.2f} 秒")
    print(f"平均每数据集时间: {summary['average_time_per_dataset']:.2f} 秒")
    
    for i, dataset_info in enumerate(summary['datasets']):
        print(f"  数据集 {i+1}: {dataset_info['name']}, "
              f"样本数={dataset_info['n_samples']}, "
              f"最终损失={dataset_info['final_loss']:.4f}")
    
    # 9. 保存结果
    print("\n9. 保存结果...")
    
    # 保存主题表示
    np.savez(os.path.join(output_dir, 'topic_representations.npz'),
             theta1=representations1['theta'],
             delta1=representations1['delta'],
             theta2=representations2['theta'],
             delta2=representations2['delta'],
             theta3=representations3['theta'],
             delta3=representations3['delta'])
    
    # 保存评估结果
    eval_results = {
        'dataset1_initial': eval1,
        'dataset1_after_dataset2': eval1_after_2,
        'dataset1_final': eval1_final,
        'dataset2_final': eval2_final,
        'dataset3_final': eval3_final
    }
    
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    # 保存训练摘要
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print("=" * 60)
    print("示例完成!")
    print("=" * 60)
    
    # 10. 性能分析
    print("\n10. 性能分析:")
    print("灾难性遗忘评估:")
    forgetting_nll = eval1_final['avg_nll'] - eval1['avg_nll']
    forgetting_loss = eval1_final['avg_loss'] - eval1['avg_loss']
    
    print(f"  数据集1 NLL 变化: {forgetting_nll:+.4f}")
    print(f"  数据集1 总损失变化: {forgetting_loss:+.4f}")
    
    if abs(forgetting_nll) < 0.1:
        print("  ✓ 很好！几乎没有灾难性遗忘")
    elif abs(forgetting_nll) < 0.5:
        print("  ± 轻微的灾难性遗忘")
    else:
        print("  ✗ 存在明显的灾难性遗忘")


if __name__ == "__main__":
    main()