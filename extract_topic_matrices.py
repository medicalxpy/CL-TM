#!/usr/bin/env python3
"""
从训练好的IncrementalGaussianPrior模型中提取cell-topic矩阵和topic-gene矩阵的工具脚本

使用方法:
python extract_topic_matrices.py --model_path ./models/model.pt --data_path ./data.h5ad --output_dir ./results/
"""

import os
import argparse
import logging
import anndata
import numpy as np
import json
from trainers.incremental_trainer import IncrementalTrainer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从scETM模型中提取主题矩阵')
    
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型文件路径 (.pt)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='数据文件路径 (.h5ad)')
    parser.add_argument('--output_dir', type=str, default='./results/extracted_matrices',
                       help='输出目录，默认: ./results/extracted_matrices')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='批大小，默认: 1024')
    parser.add_argument('--batch_col', type=str, default='batch_indices',
                       help='批次列名，默认: batch_indices')
    parser.add_argument('--save_format', type=str, default='npz',
                       choices=['npz', 'npy', 'csv', 'h5'],
                       help='保存格式，默认: npz')
    parser.add_argument('--top_genes', type=int, default=20,
                       help='每个主题显示的顶级基因数量，默认: 20')
    parser.add_argument('--analyze', action='store_true',
                       help='是否进行详细分析')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()

def load_model_and_data(model_path: str, data_path: str):
    """加载模型和数据"""
    logger.info("加载模型和数据...")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载数据
    logger.info(f"加载数据: {data_path}")
    adata = anndata.read_h5ad(data_path)
    logger.info(f"数据形状: {adata.shape}")
    
    # 检查批次列
    if 'batch_indices' not in adata.obs.columns:
        logger.info("未找到batch_indices列，创建默认批次")
        adata.obs['batch_indices'] = 0
    
    # 创建训练器并加载模型
    logger.info(f"加载模型: {model_path}")
    trainer = IncrementalTrainer(
        n_genes=adata.n_vars,
        n_topics=50,  # 这将从模型文件中覆盖
        device='cuda' if False else 'cpu'  # 避免自动使用GPU
    )
    
    try:
        trainer.load_model(model_path)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    
    return trainer, adata

def extract_and_save_matrices(trainer: IncrementalTrainer, 
                             adata: anndata.AnnData,
                             output_dir: str,
                             batch_size: int,
                             batch_col: str,
                             save_format: str,
                             verbose: bool = False):
    """提取并保存矩阵"""
    logger.info("提取主题矩阵...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取矩阵
    try:
        trainer.save_topic_matrices(
            adata=adata,
            save_dir=output_dir,
            batch_size=batch_size,
            batch_col=batch_col,
            save_format=save_format
        )
        logger.info(f"矩阵已保存到: {output_dir}")
        
        # 如果需要详细输出，打印矩阵信息
        if verbose:
            matrices = trainer.get_all_topic_matrices(adata, batch_size, batch_col)
            logger.info("矩阵详细信息:")
            for name, matrix in matrices.items():
                logger.info(f"  {name}: 形状={matrix.shape}, 数据类型={matrix.dtype}")
                logger.info(f"    均值={matrix.mean():.6f}, 标准差={matrix.std():.6f}")
                logger.info(f"    最小值={matrix.min():.6f}, 最大值={matrix.max():.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"矩阵提取失败: {str(e)}")
        return False

def analyze_matrices(trainer: IncrementalTrainer,
                    adata: anndata.AnnData,
                    output_dir: str,
                    batch_size: int,
                    batch_col: str,
                    top_genes: int):
    """分析主题矩阵"""
    logger.info("分析主题矩阵...")
    
    try:
        analysis_results = trainer.analyze_topic_matrices(
            adata=adata,
            batch_size=batch_size,
            batch_col=batch_col,
            top_genes_per_topic=top_genes
        )
        
        # 保存分析结果
        analysis_path = os.path.join(output_dir, 'topic_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析结果已保存: {analysis_path}")
        
        # 打印关键统计信息
        logger.info("主题分析摘要:")
        logger.info(f"  矩阵形状:")
        for name, shape in analysis_results['matrix_shapes'].items():
            logger.info(f"    {name}: {shape}")
        
        topic_stats = analysis_results['topic_statistics']
        cell_stats = analysis_results['cell_statistics']
        
        logger.info(f"  主题统计:")
        logger.info(f"    平均使用率: {topic_stats['topic_usage_mean']:.4f} ± {topic_stats['topic_usage_std']:.4f}")
        logger.info(f"    平均熵: {topic_stats['topic_entropy_mean']:.4f} ± {topic_stats['topic_entropy_std']:.4f}")
        
        logger.info(f"  细胞统计:")
        logger.info(f"    平均熵: {cell_stats['cell_entropy_mean']:.4f} ± {cell_stats['cell_entropy_std']:.4f}")
        
        # 显示每个主题的顶级基因
        logger.info(f"  每个主题的top-{top_genes}基因:")
        for topic_name, topic_info in analysis_results['top_genes_per_topic'].items():
            top_genes_str = ', '.join(topic_info['gene_names'][:5])  # 只显示前5个
            logger.info(f"    {topic_name}: {top_genes_str}...")
        
        return True
        
    except Exception as e:
        logger.error(f"分析失败: {str(e)}")
        return False

def create_loading_script(output_dir: str, save_format: str):
    """创建一个用于加载保存的矩阵的示例脚本"""
    script_content = f'''#!/usr/bin/env python3
"""
加载提取的主题矩阵的示例脚本
"""

import numpy as np
import json

# 加载矩阵
def load_matrices():
    """加载保存的主题矩阵"""
'''
    
    if save_format == 'npz':
        script_content += '''
    # 加载NPZ格式的矩阵
    data = np.load('topic_matrices.npz')
    
    cell_topic_matrix = data['cell_topic_matrix']  # [n_cells, n_topics]
    topic_gene_matrix = data['topic_gene_matrix']  # [n_topics, n_genes]
    cell_embedding = data['cell_embedding']        # [n_cells, n_topics]
    
    return {
        'cell_topic_matrix': cell_topic_matrix,
        'topic_gene_matrix': topic_gene_matrix,
        'cell_embedding': cell_embedding
    }
'''
    elif save_format == 'npy':
        script_content += '''
    # 加载NPY格式的矩阵
    cell_topic_matrix = np.load('cell_topic_matrix.npy')
    topic_gene_matrix = np.load('topic_gene_matrix.npy')
    cell_embedding = np.load('cell_embedding.npy')
    
    return {
        'cell_topic_matrix': cell_topic_matrix,
        'topic_gene_matrix': topic_gene_matrix,
        'cell_embedding': cell_embedding
    }
'''
    elif save_format == 'h5':
        script_content += '''
    # 加载H5格式的矩阵
    import h5py
    
    with h5py.File('topic_matrices.h5', 'r') as f:
        cell_topic_matrix = f['cell_topic_matrix'][:]
        topic_gene_matrix = f['topic_gene_matrix'][:]
        cell_embedding = f['cell_embedding'][:]
    
    return {
        'cell_topic_matrix': cell_topic_matrix,
        'topic_gene_matrix': topic_gene_matrix,
        'cell_embedding': cell_embedding
    }
'''
    
    script_content += '''
# 加载分析结果
def load_analysis():
    """加载主题分析结果"""
    with open('topic_analysis.json', 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    # 加载矩阵
    matrices = load_matrices()
    print("矩阵加载完成:")
    for name, matrix in matrices.items():
        print(f"  {name}: {matrix.shape}")
    
    # 加载分析结果
    try:
        analysis = load_analysis()
        print("\\n分析结果:")
        print(f"  主题数量: {analysis['matrix_shapes']['cell_topic_matrix'][1]}")
        print(f"  细胞数量: {analysis['matrix_shapes']['cell_topic_matrix'][0]}")
        print(f"  基因数量: {analysis['matrix_shapes']['topic_gene_matrix'][1]}")
    except FileNotFoundError:
        print("\\n未找到分析结果文件")
'''
    
    script_path = os.path.join(output_dir, 'load_matrices_example.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    logger.info(f"示例加载脚本已创建: {script_path}")

def main():
    """主函数"""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("开始提取主题矩阵")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"保存格式: {args.save_format}")
    
    try:
        # 加载模型和数据
        trainer, adata = load_model_and_data(args.model_path, args.data_path)
        
        # 提取并保存矩阵
        success = extract_and_save_matrices(
            trainer=trainer,
            adata=adata,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            batch_col=args.batch_col,
            save_format=args.save_format,
            verbose=args.verbose
        )
        
        if not success:
            logger.error("矩阵提取失败")
            return 1
        
        # 如果需要分析
        if args.analyze:
            analyze_success = analyze_matrices(
                trainer=trainer,
                adata=adata,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                batch_col=args.batch_col,
                top_genes=args.top_genes
            )
            
            if not analyze_success:
                logger.warning("分析失败，但矩阵提取成功")
        
        # 创建示例加载脚本
        create_loading_script(args.output_dir, args.save_format)
        
        logger.info("=" * 50)
        logger.info("主题矩阵提取完成！")
        logger.info(f"结果保存在: {args.output_dir}")
        logger.info("=" * 50)
        
        return 0
        
    except Exception as e:
        logger.error(f"提取过程出错: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())