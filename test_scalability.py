#!/usr/bin/env python3
"""
扩展性测试脚本 - 验证懒加载适配器在更大规模下的表现
"""

import time
import psutil
import os
from pathlib import Path

def monitor_system():
    """监控系统资源使用情况"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    return memory_mb, cpu_percent

def test_scale_performance():
    """测试不同规模下的性能表现"""
    # 现在只测试主配置文件，通过参数调整规模
    configs = [
        ({"max_samples": 500, "max_objects": 10}, "小规模测试 (500样本, 最多10对象)"),
        ({"max_samples": 5000, "max_objects": 20}, "中等规模测试 (5000样本, 最多20对象)"), 
        ({"max_samples": None, "max_objects": None}, "大规模测试 (无限制)")
    ]
    
    results = []
    
    for config_params, description in configs:
        print(f"\n{'='*60}")
        print(f"🧪 {description}")
        print(f"📁 参数: {config_params}")
        print('='*60)
        
        start_time = time.time()
        start_memory, _ = monitor_system()
        
        try:
            # 直接创建适配器进行测试
            from adapters.visual_genome_adapter import VisualGenomeAdapter
            
            print("📊 创建数据集适配器...")
            adapter_config = {
                'root_path': '/mnt/d/data/visual_genome',
                'config_name': 'objects_v1.2.0',
                'min_objects': config_params.get('min_objects', 1),
                **config_params
            }
            adapter = VisualGenomeAdapter(**adapter_config)
            
            print("🔍 初始化并扫描数据集...")
            init_start = time.time()
            num_samples = len(adapter)
            num_classes = len(adapter.classes)
            init_time = time.time() - init_start
            
            end_memory, cpu = monitor_system()
            total_time = time.time() - start_time
            memory_used = end_memory - start_memory
            
            result = {
                'config': str(config_params),
                'description': description,
                'samples': num_samples,
                'classes': num_classes,
                'init_time': init_time,
                'total_time': total_time,
                'memory_used_mb': memory_used,
                'final_memory_mb': end_memory
            }
            results.append(result)
            
            print(f"✅ 完成!")
            print(f"   📈 样本数量: {num_samples:,}")
            print(f"   🏷️  类别数量: {num_classes:,}")
            print(f"   ⏱️  初始化时间: {init_time:.2f}秒")
            print(f"   💾 内存增长: {memory_used:.1f} MB")
            print(f"   📊 当前总内存: {end_memory:.1f} MB")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append({
                'config': str(config_params),
                'description': description,
                'error': str(e)
            })
    
    # 输出汇总报告
    print(f"\n{'='*80}")
    print("📋 扩展性测试汇总报告")
    print('='*80)
    
    for result in results:
        if 'error' in result:
            print(f"❌ {result['description']}: {result['error']}")
            continue
            
        print(f"\n🎯 {result['description']}")
        print(f"   📊 数据规模: {result['samples']:,} 样本, {result['classes']:,} 类别")
        print(f"   ⚡ 性能指标: {result['init_time']:.2f}s 初始化, {result['memory_used_mb']:.1f}MB 内存")
        
        # 计算效率指标
        if result['samples'] > 0:
            samples_per_sec = result['samples'] / result['init_time']
            memory_per_sample = result['memory_used_mb'] / result['samples'] * 1000  # KB per sample
            print(f"   📈 效率指标: {samples_per_sec:.0f} 样本/秒, {memory_per_sample:.1f}KB/样本")
    
    print(f"\n{'='*80}")
    print("🏆 结论:")
    print("  - 如果所有测试都成功且内存增长合理，说明懒加载实现优秀")
    print("  - 内存增长应该与样本数量无关，主要是索引存储")
    print("  - 初始化时间应该随数据集大小线性增长")
    print('='*80)

if __name__ == "__main__":
    print("🚀 开始扩展性测试...")
    test_scale_performance()
