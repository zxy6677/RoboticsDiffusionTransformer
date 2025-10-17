#!/usr/bin/env python3
"""
计算LIBERO数据集的统计信息（修改翻转后需要重新计算）
"""

import sys
sys.path.append('.')

import json
import numpy as np
from data.hdf5_libero_dataset import HDF5LIBERODataset

print("🔄 正在计算LIBERO数据集统计信息...")
print("=" * 80)

# 创建数据集
dataset = HDF5LIBERODataset()

print(f"数据集名称: {dataset.get_dataset_name()}")
print(f"文件数量: {len(dataset)}")

# 收集所有状态和动作
all_states = []
all_actions = []

print("\n正在收集数据...")
for i in range(len(dataset)):
    if i % 10 == 0:
        print(f"  处理文件 {i+1}/{len(dataset)}")
    
    sample = dataset.get_item(index=i, state_only=False)
    
    # 提取state和action
    state = sample['state']  # (1, 128)
    actions = sample['actions']  # (64, 128)
    
    all_states.append(state)
    all_actions.append(actions)

# 合并所有数据
all_states = np.concatenate(all_states, axis=0)  # (N, 128)
all_actions = np.concatenate(all_actions, axis=0)  # (M, 128)

print(f"\n收集完成:")
print(f"  States shape: {all_states.shape}")
print(f"  Actions shape: {all_actions.shape}")

# 计算统计信息
state_mean = np.mean(all_states, axis=0).tolist()
state_std = np.std(all_states, axis=0).tolist()
state_min = np.min(all_states, axis=0).tolist()
state_max = np.max(all_states, axis=0).tolist()

action_mean = np.mean(all_actions, axis=0).tolist()
action_std = np.std(all_actions, axis=0).tolist()
action_min = np.min(all_actions, axis=0).tolist()
action_max = np.max(all_actions, axis=0).tolist()

# 构建统计字典
stats = {
    'libero_90': {
        'state_mean': state_mean,
        'state_std': state_std,
        'state_min': state_min,
        'state_max': state_max,
        'action_mean': action_mean,
        'action_std': action_std,
        'action_min': action_min,
        'action_max': action_max
    }
}

# 保存到文件
output_file = 'configs/dataset_stat.json'
with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ 统计信息已保存到: {output_file}")

# 显示一些关键统计
from configs.state_vec import STATE_VEC_IDX_MAPPING
pos_indices = [
    STATE_VEC_IDX_MAPPING['right_eef_pos_x'],
    STATE_VEC_IDX_MAPPING['right_eef_pos_y'],
    STATE_VEC_IDX_MAPPING['right_eef_pos_z']
]

print(f"\n📊 关键统计信息（EEF位置）:")
for i, idx in enumerate(pos_indices):
    axis = ['X', 'Y', 'Z'][i]
    print(f"  {axis} (索引{idx}):")
    print(f"    mean={state_mean[idx]:.6f}, std={state_std[idx]:.6f}")
    print(f"    min={state_min[idx]:.6f}, max={state_max[idx]:.6f}")

print("\n=" * 80)
print("✅ 完成！")

