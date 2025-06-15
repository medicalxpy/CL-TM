#!/usr/bin/env python3
"""
将单个adata数据集分为三份进行IncrementalGaussianPrior增量训练的示例脚本

请修改以下路径为你的实际adata文件路径：
- ADATA_PATH: 完整数据集的路径
"""

import os
import logging
import anndata
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from trainers.incremental_trainer import IncrementalTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 请修改这个路径为你的实际adata文件路径 =====
ADATA_PATH = "/path/to/your/complete_dataset.h5ad"  # 修改为你的adata文件路径
# ===============================================

# 训练配置
CONFIG = {
    'n_topics': 50,               # 主题数量
    'hidden_sizes': [128],        # 编码器隐藏层大小
    'gene_emb_dim': 400,         # 基因嵌入维度
    'learning_rate': 5e-3,       # 学习率
    'weight_decay': 0.0,         # 权重衰减
    'prior_strength': 1.0,       # 先验强度
    'adaptive_strength': True,   # 自适应先验强度
    'n_epochs': 100,             # 训练轮数
    'batch_size': 1024,          # 批大小
    'eval_every': 10,            # 评估间隔
    'seed': 42                   # 随机种子
}

# 数据分割配置
SPLIT_CONFIG = {
    'split_ratios': [0.4, 0.3, 0.3],  # 三份数据的比例 (40%, 30%, 30%)
    'stratify_by': None,               # 分层采样的列名，None表示随机分割
    'random_state': 42                 # 分割随机种子
}

def load_and_check_adata(path: str):
    """加载并检查adata文件"""
    logger.info(f"加载完整数据集 from {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    adata = anndata.read_h5ad(path)
    logger.info(f"完整数据集信息:")
    logger.info(f"  - 样本数: {adata.n_obs}")
    logger.info(f"  - 基因数: {adata.n_vars}")
    logger.info(f"  - 观测变量: {list(adata.obs.columns)}")
    
    # 检查是否有批次信息
    if 'batch_indices' not in adata.obs.columns:
        logger.info("  - 未找到batch_indices列，将创建默认批次")
        adata.obs['batch_indices'] = 0
    
    return adata

def split_adata_into_three(adata: anndata.AnnData, 
                          ratios: list = [0.4, 0.3, 0.3],
                          stratify_by: str = None,
                          random_state: int = 42):
    """
    将adata数据集分为三份
    
    参数:
        adata: 完整的adata数据集
        ratios: 三份数据的比例，默认[0.4, 0.3, 0.3]
        stratify_by: 分层采样的列名，None表示随机分割
        random_state: 随机种子
        
    返回:
        三个adata数据集的列表
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1.0，当前为: {sum(ratios)}")
    
    logger.info(f"将数据集分为三份，比例: {ratios}")
    
    n_samples = adata.n_obs
    indices = np.arange(n_samples)
    
    # 准备分层采样的标签
    stratify_labels = None
    if stratify_by and stratify_by in adata.obs.columns:
        stratify_labels = adata.obs[stratify_by].values
        logger.info(f"使用 {stratify_by} 列进行分层采样")
    
    # 第一次分割：分出第一份数据
    indices_first, indices_rest = train_test_split(
        indices,
        test_size=(ratios[1] + ratios[2]),
        stratify=stratify_labels,
        random_state=random_state
    )
    
    # 第二次分割：将剩余数据分为第二份和第三份
    rest_ratio = ratios[2] / (ratios[1] + ratios[2])  # 第三份在剩余数据中的比例
    
    # 如果使用分层采样，需要获取剩余数据的标签
    rest_stratify_labels = None
    if stratify_labels is not None:
        rest_stratify_labels = stratify_labels[indices_rest]
    
    indices_second, indices_third = train_test_split(
        indices_rest,
        test_size=rest_ratio,
        stratify=rest_stratify_labels,
        random_state=random_state + 1
    )
    
    # 创建三个子数据集
    adata_first = adata[indices_first].copy()
    adata_second = adata[indices_second].copy()
    adata_third = adata[indices_third].copy()
    
    # 重置索引
    adata_first.obs_names_make_unique()
    adata_second.obs_names_make_unique()
    adata_third.obs_names_make_unique()
    
    # 添加数据集标识
    adata_first.obs['dataset_split'] = 'first'
    adata_second.obs['dataset_split'] = 'second'
    adata_third.obs['dataset_split'] = 'third'
    
    logger.info(f"数据集分割完成:")
    logger.info(f"  - 第一份: {adata_first.n_obs} 样本 ({adata_first.n_obs/n_samples:.1%})")
    logger.info(f"  - 第二份: {adata_second.n_obs} 样本 ({adata_second.n_obs/n_samples:.1%})")
    logger.info(f"  - 第三份: {adata_third.n_obs} 样本 ({adata_third.n_obs/n_samples:.1%})")
    
    return [adata_first, adata_second, adata_third]

def train_incremental_sequence(trainer: IncrementalTrainer, 
                              adata_list: list,
                              config: dict):
    """
    按顺序进行增量训练
    
    参数:
        trainer: 增量训练器
        adata_list: 三个数据集的列表
        config: 训练配置
        
    返回:
        训练历史列表
    """
    training_histories = []
    
    # 训练第一个数据集（使用标准先验）
    logger.info("=" * 60)
    logger.info("阶段 1/3: 训练第一个数据子集")
    logger.info("=" * 60)
    
    first_history = trainer.train_first_dataset(
        adata=adata_list[0],
        dataset_name="split_1_of_3",
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        eval_every=config['eval_every'],
        save_path="./models/incremental_split_after_1.pt"
    )
    training_histories.append(first_history)
    
    logger.info(f"第一个子集训练完成，最终loss: {first_history['final_loss']:.4f}")
    
    # 训练第二个数据集（使用增量高斯先验）
    logger.info("=" * 60)
    logger.info("阶段 2/3: 增量训练第二个数据子集")
    logger.info("=" * 60)
    
    second_history = trainer.train_incremental_dataset(
        adata=adata_list[1],
        dataset_name="split_2_of_3",
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        eval_every=config['eval_every'],
        save_path="./models/incremental_split_after_2.pt"
    )
    training_histories.append(second_history)
    
    logger.info(f"第二个子集训练完成，最终loss: {second_history['final_loss']:.4f}")
    
    # 训练第三个数据集（继续使用增量高斯先验）
    logger.info("=" * 60)
    logger.info("阶段 3/3: 增量训练第三个数据子集")
    logger.info("=" * 60)
    
    third_history = trainer.train_incremental_dataset(
        adata=adata_list[2],
        dataset_name="split_3_of_3",
        n_epochs=config['n_epochs'],
        batch_size=config['batch_size'],
        eval_every=config['eval_every'],
        save_path="./models/incremental_split_final.pt"
    )
    training_histories.append(third_history)
    
    logger.info(f"第三个子集训练完成，最终loss: {third_history['final_loss']:.4f}")
    
    return training_histories

def evaluate_on_all_subsets(trainer: IncrementalTrainer, 
                           adata_list: list,
                           config: dict):
    """
    在所有子集上评估模型性能
    
    参数:
        trainer: 训练器
        adata_list: 数据集列表
        config: 配置
        
    返回:
        评估结果列表
    """
    logger.info("=" * 60)
    logger.info("评估最终模型在所有子集上的性能")
    logger.info("=" * 60)
    
    eval_results = []
    
    for i, adata in enumerate(adata_list, 1):
        logger.info(f"评估子集 {i}/3...")
        eval_result = trainer.evaluate_model(
            adata, 
            batch_size=config['batch_size']
        )
        eval_result['subset_name'] = f"split_{i}_of_3"
        eval_result['subset_size'] = adata.n_obs
        eval_results.append(eval_result)
        
        logger.info(f"子集 {i} 评估结果:")
        logger.info(f"  - Loss: {eval_result['avg_loss']:.4f}")
        logger.info(f"  - NLL: {eval_result['avg_nll']:.4f}")
        logger.info(f"  - KL: {eval_result['avg_kl']:.4f}")
        logger.info(f"  - Perplexity: {eval_result['perplexity']:.4f}")
    
    return eval_results

def main():
    """主训练函数"""
    logger.info("开始IncrementalGaussianPrior分割训练示例")
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 加载完整数据集
        complete_adata = load_and_check_adata(ADATA_PATH)
        n_genes = complete_adata.n_vars
        
        # 2. 将数据集分为三份
        adata_list = split_adata_into_three(
            complete_adata,
            ratios=SPLIT_CONFIG['split_ratios'],
            stratify_by=SPLIT_CONFIG['stratify_by'],
            random_state=SPLIT_CONFIG['random_state']
        )
        
        # 3. 初始化增量训练器
        trainer = IncrementalTrainer(
            n_genes=n_genes,
            n_topics=CONFIG['n_topics'],
            hidden_sizes=CONFIG['hidden_sizes'],
            gene_emb_dim=CONFIG['gene_emb_dim'],
            learning_rate=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
            prior_strength=CONFIG['prior_strength'],
            adaptive_strength=CONFIG['adaptive_strength'],
            device=device,
            seed=CONFIG['seed']
        )
        
        # 4. 按顺序进行增量训练
        training_histories = train_incremental_sequence(trainer, adata_list, CONFIG)
        
        # 5. 在所有子集上评估模型
        eval_results = evaluate_on_all_subsets(trainer, adata_list, CONFIG)
        
        # 6. 获取训练摘要
        summary = trainer.get_training_summary()
        logger.info("=" * 60)
        logger.info("增量训练摘要")
        logger.info("=" * 60)
        logger.info(f"总数据集数: {summary['total_datasets']}")
        logger.info(f"总训练时间: {summary['total_training_time']:.2f}秒")
        logger.info(f"总样本数: {summary['total_samples']}")
        logger.info(f"平均每个子集训练时间: {summary['average_time_per_dataset']:.2f}秒")
        
        # 展示每个阶段的损失变化
        logger.info("\n各阶段训练损失:")
        for i, history in enumerate(training_histories, 1):
            logger.info(f"  阶段 {i}: {history['final_loss']:.4f}")
        
        # 7. 提取并保存主题矩阵
        logger.info("=" * 60)
        logger.info("提取主题矩阵")
        logger.info("=" * 60)
        
        try:
            # 使用最后一个子数据集提取矩阵（代表最终结果）
            final_adata = adata_list[-1]  # 第三个子数据集
            
            # 保存主题矩阵
            trainer.save_topic_matrices(
                adata=final_adata,
                save_dir="./results/topic_matrices_incremental",
                batch_size=CONFIG['batch_size'],
                save_format='npz'
            )
            
            # 分析主题矩阵
            analysis_results = trainer.analyze_topic_matrices(
                adata=final_adata,
                batch_size=CONFIG['batch_size'],
                top_genes_per_topic=20
            )
            
            # 保存分析结果
            import json
            os.makedirs("./results", exist_ok=True)
            with open("./results/incremental_topic_analysis.json", 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            logger.info("主题矩阵和分析结果已保存到 ./results/ 目录")
            
            # 显示关键统计信息
            logger.info(f"矩阵形状:")
            for name, shape in analysis_results['matrix_shapes'].items():
                logger.info(f"  - {name}: {shape}")
            
            logger.info(f"主题统计:")
            logger.info(f"  - 平均主题使用率: {analysis_results['topic_statistics']['topic_usage_mean']:.4f}")
            logger.info(f"  - 平均细胞熵: {analysis_results['cell_statistics']['cell_entropy_mean']:.4f}")
            
            # 也可以在每个子数据集上分别提取矩阵（可选）
            if input("是否在每个子数据集上分别提取矩阵? (y/n): ").strip().lower() == 'y':
                for i, adata_subset in enumerate(adata_list, 1):
                    subset_matrices = trainer.get_all_topic_matrices(adata_subset, CONFIG['batch_size'])
                    
                    # 保存每个子集的矩阵
                    subset_dir = f"./results/matrices_subset_{i}"
                    os.makedirs(subset_dir, exist_ok=True)
                    
                    save_path = os.path.join(subset_dir, 'topic_matrices.npz')
                    np.savez_compressed(
                        save_path,
                        cell_topic_matrix=subset_matrices['cell_topic_matrix'],
                        topic_gene_matrix=subset_matrices['topic_gene_matrix'],
                        cell_embedding=subset_matrices['cell_embedding']
                    )
                    logger.info(f"子集 {i} 矩阵已保存: {save_path}")
            
        except Exception as e:
            logger.warning(f"主题矩阵提取失败: {str(e)}")
        
        # 8. 获取最终先验信息
        prior_info = trainer.model.get_prior_info()
        logger.info("=" * 60)
        logger.info("最终先验分布信息")
        logger.info("=" * 60)
        for key, value in prior_info.items():
            logger.info(f"{key}: {value}")
        
        # 9. 保存分割后的数据集（可选）
        if input("是否保存分割后的子数据集? (y/n): ").strip().lower() == 'y':
            os.makedirs("./data_splits", exist_ok=True)
            for i, adata in enumerate(adata_list, 1):
                save_path = f"./data_splits/split_{i}_of_3.h5ad"
                adata.write_h5ad(save_path)
                logger.info(f"子数据集 {i} 已保存到: {save_path}")
        
        logger.info("=" * 60)
        logger.info("增量训练完成！")
        logger.info("模型已保存到 ./models/ 目录")
        logger.info("主题矩阵已保存到 ./results/ 目录")
        logger.info("=" * 60)
        
        # 返回结果供进一步分析
        return {
            'trainer': trainer,
            'adata_list': adata_list,
            'training_histories': training_histories,
            'eval_results': eval_results,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs("./models", exist_ok=True)
    
    # 运行主函数
    results = main()