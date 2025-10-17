#!/usr/bin/env python3
"""
简单的调试脚本 - 从训练数据检查action的实际值和符号
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING

# 读取一个demo文件
demo_file = 'data/datasets/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5'

with h5py.File(demo_file, 'r') as f:
    actions = f['data/demo_0/actions'][:]  # (T, 7)
    ee_pos = f['data/demo_0/obs/ee_pos'][:]  # (T, 3)

print("=" * 80)
print("Demo数据检查 - 机械臂运动方向分析")
print("=" * 80)

# 选择前几个时间步
for t in [0, 1, 2, 3, 4, 5]:
    action = actions[t]
    pos_action = action[0:3]  # 归一化值 [-1, 1]
    
    # 计算物理增量
    if t < len(ee_pos) - 1:
        actual_delta = ee_pos[t+1] - ee_pos[t]
        predicted_delta = pos_action * 0.012
        
        print(f"\n时间步 t={t}:")
        print(f"  当前位置: [{ee_pos[t,0]:.4f}, {ee_pos[t,1]:.4f}, {ee_pos[t,2]:.4f}]")
        print(f"  下一位置: [{ee_pos[t+1,0]:.4f}, {ee_pos[t+1,1]:.4f}, {ee_pos[t+1,2]:.4f}]")
        print(f"  实际增量: [{actual_delta[0]:.6f}, {actual_delta[1]:.6f}, {actual_delta[2]:.6f}]")
        print(f"  Action值: [{pos_action[0]:.4f}, {pos_action[1]:.4f}, {pos_action[2]:.4f}] (归一化)")
        print(f"  预测增量: [{predicted_delta[0]:.6f}, {predicted_delta[1]:.6f}, {predicted_delta[2]:.6f}] (action*0.012)")
        
        # 检查符号是否一致
        for i, axis in enumerate(['X', 'Y', 'Z']):
            actual_sign = np.sign(actual_delta[i])
            predicted_sign = np.sign(predicted_delta[i])
            match = '✅' if actual_sign == predicted_sign else '❌'
            print(f"    {axis}轴: 实际={actual_sign:+.0f}, 预测={predicted_sign:+.0f} {match}")

print("\n" + "=" * 80)
print("关键观察")
print("=" * 80)

# 统计方向一致性
consistent_count = 0
total_count = 0

for t in range(min(20, len(actions)-1)):
    action = actions[t]
    pos_action = action[0:3]
    actual_delta = ee_pos[t+1] - ee_pos[t]
    predicted_delta = pos_action * 0.012
    
    for i in range(3):
        if np.abs(actual_delta[i]) > 0.0001:  # 只检查有明显运动的轴
            actual_sign = np.sign(actual_delta[i])
            predicted_sign = np.sign(predicted_delta[i])
            total_count += 1
            if actual_sign == predicted_sign:
                consistent_count += 1

if total_count > 0:
    consistency_rate = consistent_count / total_count * 100
    print(f"方向一致性: {consistent_count}/{total_count} = {consistency_rate:.1f}%")
    
    if consistency_rate > 90:
        print("✅ 训练数据的Action符号与实际运动方向高度一致")
        print("   问题不在数据处理层面")
    else:
        print("❌ 训练数据的Action符号与实际运动方向不一致")
        print("   数据处理有问题")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print("如果方向一致性高（>90%），说明：")
print("1. 训练数据是正确的")
print("2. 模型应该学到正确的映射")
print("3. 评估时的转换逻辑也是正确的")
print("")
print("那么问题可能在：")
print("A. 模型训练不足/没学好")
print("B. 评估时的checkpoint加载有问题")
print("C. LIBERO环境执行action时有额外的处理")

