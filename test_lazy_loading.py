#!/usr/bin/env python3
"""
快速测试TreeOfLife adapter的pandas懒加载功能
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lazy_loading():
    """测试pandas分块懒加载"""
    print("🧪 测试TreeOfLife adapter的pandas懒加载...")
    
    from adapters.treeoflife_adapter import TreeOfLifeAdapter
    
    start_time = time.time()
    
    # 创建adapter - 应该很快，不需要构建数据库
    adapter = TreeOfLifeAdapter(
        root_path='/mnt/d/data/treeoflife',
        max_shards=1,  # 只处理1个shard
        max_samples=10  # 只处理10个样本
    )
    
    init_time = time.time() - start_time
    print(f"✅ Adapter初始化完成: {init_time:.2f}秒")
    print(f"📊 加载了 {len(adapter)} 个样本")
    
    # 测试几个样本
    for i in range(min(3, len(adapter))):
        image, label = adapter[i]
        class_name = adapter.classes[label]
        print(f"样本 {i}: 类别={class_name}, 图像大小={image.shape}")
    
    print("✅ 懒加载测试完成!")
    return True

if __name__ == "__main__":
    try:
        test_lazy_loading()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
