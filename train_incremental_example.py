#!/usr/bin/env python3
"""
使用两个现有adata文件进行IncrementalGaussianPrior学习的示例脚本

请修改以下路径为你的实际adata文件路径：
- FIRST_ADATA_PATH: 第一个数据集的路径
- SECOND_ADATA_PATH: 第二个数据集的路径
"""

import os
import logging
import anndata
import torch
from trainers.incremental_trainer import IncrementalTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== 请修改这些路径为你的实际adata文件路径 =====
FIRST_ADATA_PATH = "/path/to/your/first_dataset.h5ad"    # 修改为第一个adata文件路径
SECOND_ADATA_PATH = "/path/to/your/second_dataset.h5ad"  # 修改为第二个adata文件路径
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

def load_and_check_adata(path: str, dataset_name: str):
    """加载并检查adata文件"""
    logger.info(f"加载数据集: {dataset_name} from {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    adata = anndata.read_h5ad(path)
    logger.info(f"{dataset_name} 数据集信息:")
    logger.info(f"  - 样本数: {adata.n_obs}")
    logger.info(f"  - 基因数: {adata.n_vars}")
    logger.info(f"  - 观测变量: {list(adata.obs.columns)}")
    
    # 检查是否有批次信息
    if 'batch_indices' not in adata.obs.columns:
        logger.info("  - 未找到batch_indices列，将创建默认批次")
        adata.obs['batch_indices'] = 0
    
    return adata

def main():
    """主训练函数"""
    logger.info("开始IncrementalGaussianPrior训练示例")
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 加载数据集
        first_adata = load_and_check_adata(FIRST_ADATA_PATH, "第一个数据集")
        second_adata = load_and_check_adata(SECOND_ADATA_PATH, "第二个数据集")
        
        # 确保两个数据集的基因数量一致
        if first_adata.n_vars != second_adata.n_vars:
            raise ValueError(f"两个数据集的基因数量不一致: {first_adata.n_vars} vs {second_adata.n_vars}")
        
        n_genes = first_adata.n_vars
        logger.info(f"基因数量: {n_genes}")
        
        # 2. 初始化增量训练器
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
        
        # 3. 训练第一个数据集（使用标准先验）
        logger.info("=" * 50)
        logger.info("开始训练第一个数据集")
        logger.info("=" * 50)
        
        first_history = trainer.train_first_dataset(
            adata=first_adata,
            dataset_name="dataset_1",
            n_epochs=CONFIG['n_epochs'],
            batch_size=CONFIG['batch_size'],
            eval_every=CONFIG['eval_every'],
            save_path="./models/incremental_after_first.pt"
        )
        
        logger.info(f"第一个数据集训练完成，最终loss: {first_history['final_loss']:.4f}")
        
        # 4. 训练第二个数据集（使用增量高斯先验）
        logger.info("=" * 50)
        logger.info("开始增量训练第二个数据集")
        logger.info("=" * 50)
        
        second_history = trainer.train_incremental_dataset(
            adata=second_adata,
            dataset_name="dataset_2",
            n_epochs=CONFIG['n_epochs'],
            batch_size=CONFIG['batch_size'],
            eval_every=CONFIG['eval_every'],
            save_path="./models/incremental_after_second.pt"
        )
        
        logger.info(f"第二个数据集训练完成，最终loss: {second_history['final_loss']:.4f}")
        
        # 5. 评估模型性能
        logger.info("=" * 50)
        logger.info("评估模型性能")
        logger.info("=" * 50)
        
        # 在第一个数据集上评估
        eval_first = trainer.evaluate_model(first_adata, batch_size=CONFIG['batch_size'])
        logger.info(f"第一个数据集评估结果: {eval_first}")
        
        # 在第二个数据集上评估
        eval_second = trainer.evaluate_model(second_adata, batch_size=CONFIG['batch_size'])
        logger.info(f"第二个数据集评估结果: {eval_second}")
        
        # 6. 获取训练摘要
        summary = trainer.get_training_summary()
        logger.info("=" * 50)
        logger.info("训练摘要")
        logger.info("=" * 50)
        logger.info(f"总数据集数: {summary['total_datasets']}")
        logger.info(f"总训练时间: {summary['total_training_time']:.2f}秒")
        logger.info(f"总样本数: {summary['total_samples']}")
        logger.info(f"平均每个数据集训练时间: {summary['average_time_per_dataset']:.2f}秒")
        
        # 7. 获取先验信息
        prior_info = trainer.model.get_prior_info()
        logger.info("=" * 50)
        logger.info("最终先验分布信息")
        logger.info("=" * 50)
        for key, value in prior_info.items():
            logger.info(f"{key}: {value}")
        
        logger.info("训练完成！模型已保存到 ./models/ 目录")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs("./models", exist_ok=True)
    
    # 运行主函数
    main()