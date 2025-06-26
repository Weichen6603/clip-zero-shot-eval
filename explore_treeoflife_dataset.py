#!/usr/bin/env python3
"""
探索TreeOfLife-10M数据集结构，查找metadata/catalog.csv文件
"""

import os
import sys
from datasets import load_dataset

def explore_dataset_structure():
    """探索TreeOfLife-10M数据集的完整结构"""
    print("🔍 探索TreeOfLife-10M数据集结构...")
    
    dataset_name = "imageomics/TreeOfLife-10M"
    cache_dir = "/mnt/d/data/treeoflife"
    
    try:
        print("📥 尝试加载完整数据集信息...")
        
        # 方法1: 获取数据集信息 (不下载数据)
        from datasets import get_dataset_config_names, get_dataset_split_names
        try:
            configs = get_dataset_config_names(dataset_name)
            print(f"📋 Available configs: {configs}")
            
            for config in configs:
                splits = get_dataset_split_names(dataset_name, config_name=config)
                print(f"   Config '{config}' splits: {splits}")
        except Exception as e:
            print(f"⚠️  无法获取config信息: {e}")
        
        # 方法2: 尝试直接访问仓库文件
        print("\n🔗 尝试访问HuggingFace仓库文件...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # 列出仓库中的所有文件
            repo_files = api.list_repo_files(dataset_name, repo_type="dataset")
            print(f"📁 仓库文件总数: {len(repo_files)}")
            
            # 查找metadata相关文件
            metadata_files = [f for f in repo_files if 'metadata' in f.lower() or 'catalog' in f.lower()]
            print(f"📊 Metadata相关文件: {metadata_files}")
            
            # 查找CSV文件
            csv_files = [f for f in repo_files if f.endswith('.csv')]
            print(f"📄 CSV文件: {csv_files[:10]}...")  # 只显示前10个
            
        except Exception as e:
            print(f"⚠️  无法访问仓库文件: {e}")
        
        # 方法3: 尝试下载catalog.csv
        print("\n📥 尝试直接下载catalog.csv...")
        try:
            from huggingface_hub import hf_hub_download
            
            catalog_path = hf_hub_download(
                repo_id=dataset_name,
                filename="metadata/catalog.csv",
                repo_type="dataset",
                cache_dir=cache_dir
            )
            print(f"✅ 成功下载catalog.csv到: {catalog_path}")
            
            # 读取前几行看看结构
            import pandas as pd
            catalog_df = pd.read_csv(catalog_path, nrows=5)
            print(f"📋 Catalog结构 (前5行):")
            print(catalog_df.head())
            print(f"📋 Catalog列名: {list(catalog_df.columns)}")
            
            return catalog_path
            
        except Exception as e:
            print(f"⚠️  无法下载catalog.csv: {e}")
        
        # 方法4: 尝试streaming然后检查第一个样本
        print("\n🔄 使用streaming模式检查样本结构...")
        try:
            dataset = load_dataset(
                dataset_name,
                split="train",
                streaming=True,
                cache_dir=cache_dir
            )
            
            first_sample = next(iter(dataset))
            print(f"📋 Streaming样本结构: {list(first_sample.keys())}")
            
            for key, value in first_sample.items():
                if isinstance(value, str):
                    print(f"   {key}: {value[:100]}...")
                else:
                    print(f"   {key}: {type(value)}")
                    
        except Exception as e:
            print(f"⚠️  Streaming检查失败: {e}")
        
    except Exception as e:
        print(f"❌ 数据集探索失败: {e}")
        import traceback
        traceback.print_exc()
        
    return None

if __name__ == "__main__":
    catalog_path = explore_dataset_structure()
    if catalog_path:
        print(f"\n✅ 找到catalog文件: {catalog_path}")
    else:
        print(f"\n❌ 未找到catalog文件")
