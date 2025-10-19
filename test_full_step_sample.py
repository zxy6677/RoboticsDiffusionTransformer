#!/usr/bin/env python3
"""测试全步骤枚举采样"""

import os
import sys
sys.path.insert(0, '.')

from data.hdf5_libero_dataset import HDF5LIBERODataset

# 设置环境变量
os.environ['LIBERO_DATASET_DIR'] = 'datasets/libero_single_task'

print("=" * 80)
print("测试全步骤枚举采样策略")
print("=" * 80)

# 初始化dataset
dataset = HDF5LIBERODataset(
    dataset_name='libero_single_task', 
    use_full_step_enumeration=True
)

print(f"\n📊 Dataset信息:")
print(f"   - 总样本数: {len(dataset)}")
print(f"   - 数据集名称: {dataset.get_dataset_name()}")

# 测试获取样本
print(f"\n🧪 测试采样:")
try:
    for i in [0, 100, 1000]:
        print(f"\n   测试索引 {i}:")
        sample = dataset.get_item(index=i)
        print(f"      ✅ 成功获取样本")
        print(f"      - State shape: {sample['state'].shape}")
        print(f"      - Actions shape: {sample['actions'].shape}")
        print(f"      - Task: {sample['meta']['instruction'][:50]}...")
        if 'step_id' in sample['meta']:
            print(f"      - Step ID: {sample['meta']['step_id']}")
except Exception as e:
    print(f"      ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print(f"\n" + "=" * 80)
print("✅ 测试完成！")
print("=" * 80)

