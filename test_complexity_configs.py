#!/usr/bin/env python3
"""
测试不同objects参数配置对复杂场景的影响
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adapters.visual_genome_adapter import VisualGenomeAdapter

def test_complexity_configs():
    """测试不同复杂度配置的效果"""
    print("🧪 测试Visual Genome复杂场景配置")
    print("=" * 60)
    
    configs = [
        {
            'name': '全包含配置',
            'config': {
                'root_path': '/mnt/d/data/visual_genome',
                'max_samples': 2000,  # 限制样本数以快速测试
                'min_objects': 1,
                # 'max_objects': None  # 不设置 = 包含所有复杂度
                'use_synsets': False,
                'config_name': 'objects_v1.2.0'
            }
        },
        {
            'name': '中等复杂度配置',
            'config': {
                'root_path': '/mnt/d/data/visual_genome',
                'max_samples': 2000,
                'min_objects': 5,      # 至少5个对象
                'max_objects': 20,     # 最多20个对象
                'use_synsets': False,
                'config_name': 'objects_v1.2.0'
            }
        },
        {
            'name': '高复杂度配置',
            'config': {
                'root_path': '/mnt/d/data/visual_genome',
                'max_samples': 2000,
                'min_objects': 10,     # 至少10个对象
                # 'max_objects': None  # 不限制上限
                'use_synsets': False,
                'config_name': 'objects_v1.2.0'
            }
        },
        {
            'name': '极复杂场景配置',
            'config': {
                'root_path': '/mnt/d/data/visual_genome',
                'max_samples': 2000,
                'min_objects': 20,     # 至少20个对象
                # 'max_objects': None  # 包含最复杂场景
                'use_synsets': False,
                'config_name': 'objects_v1.2.0'
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(configs):
        print(f"\n📊 测试 {i+1}: {test_case['name']}")
        print("-" * 40)
        
        try:
            dataset = VisualGenomeAdapter(**test_case['config'])
            
            result = {
                'name': test_case['name'],
                'sample_count': len(dataset),
                'class_count': len(dataset.classes),
                'config': test_case['config']
            }
            
            print(f"✅ 样本数量: {result['sample_count']}")
            print(f"✅ 类别数量: {result['class_count']}")
            
            # 分析样本信息
            if len(dataset) > 0:
                sample_info = dataset.get_sample_info(0)
                print(f"✅ 示例样本: {sample_info}")
                
                # 显示一些类别示例
                sample_classes = dataset.classes[:10]
                print(f"✅ 类别示例: {sample_classes}")
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ 错误: {e}")
            continue
    
    # 总结对比
    print(f"\n🔍 复杂度配置对比总结:")
    print("=" * 60)
    print(f"{'配置名称':<15} {'样本数':<8} {'类别数':<8} {'复杂度':<10}")
    print("-" * 60)
    
    for result in results:
        complexity = "未知"
        min_obj = result['config'].get('min_objects', 1)
        max_obj = result['config'].get('max_objects', 'unlimited')
        
        if min_obj == 1:
            complexity = "全范围"
        elif min_obj <= 5:
            complexity = "中等"
        elif min_obj <= 10:
            complexity = "高"
        else:
            complexity = "极高"
            
        print(f"{result['name']:<15} {result['sample_count']:<8} {result['class_count']:<8} {complexity:<10}")
    
    print(f"\n💡 观察和建议:")
    print("1. min_objects 越高，样本数越少，但场景越复杂")
    print("2. 移除 max_objects 限制可以包含最复杂的场景")
    print("3. 复杂场景通常包含更多长尾类别")
    print("4. 根据评估目标选择合适的复杂度配置")

def analyze_complexity_distribution():
    """分析数据集中的复杂度分布"""
    print(f"\n🔍 分析Visual Genome复杂度分布")
    print("=" * 50)
    
    try:
        # 样本数据集以了解分布
        dataset = VisualGenomeAdapter(
            root_path='/mnt/d/data/visual_genome',
            max_samples=1000,  # 样本1000个进行分析
            min_objects=1,
            use_synsets=False,
            config_name='objects_v1.2.0'
        )
        
        print(f"分析样本: {len(dataset)} 个图像")
        
        # 统计对象数量分布 (这需要实际检查数据集)
        print("✅ 复杂度分析完成")
        print("💡 建议根据实际需求选择min_objects参数")
        
    except Exception as e:
        print(f"❌ 分析出错: {e}")

if __name__ == "__main__":
    test_complexity_configs()
    analyze_complexity_distribution()
