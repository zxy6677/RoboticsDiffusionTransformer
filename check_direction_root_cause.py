#!/usr/bin/env python3
"""
系统检查方向反转的根本原因
"""

import sys
sys.path.append('.')

import h5py
import numpy as np
from data.hdf5_libero_dataset import HDF5LIBERODataset
from configs.state_vec import STATE_VEC_IDX_MAPPING

print('=' * 80)
print('系统检查：为什么评估时机械臂方向反了？')
print('=' * 80)

# 1. 检查训练数据的action符号
print('\n1. 检查训练数据处理后的action符号')
print('-' * 80)

dataset = HDF5LIBERODataset()
dataset.file_paths = ['data/datasets/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5']

# 读取原始和处理后的数据
with h5py.File(dataset.file_paths[0], 'r') as f:
    raw_actions = f['data/demo_0/actions'][:]
    ee_pos = f['data/demo_0/obs/ee_pos'][:]

# 手动模拟训练数据处理
pos_normalized = raw_actions[:, 0:3]
pos_meters_old = pos_normalized * 0.05  # 旧的错误缩放
pos_meters_new = pos_normalized * 0.012  # 新的正确缩放

print('\n对比缩放因子:')
print(f'原始LIBERO action[10]: [{pos_normalized[10,0]:.4f}, {pos_normalized[10,1]:.4f}, {pos_normalized[10,2]:.4f}]')
print(f'旧scale(0.05)处理后:  [{pos_meters_old[10,0]:.6f}, {pos_meters_old[10,1]:.6f}, {pos_meters_old[10,2]:.6f}] m')
print(f'新scale(0.012)处理后: [{pos_meters_new[10,0]:.6f}, {pos_meters_new[10,1]:.6f}, {pos_meters_new[10,2]:.6f}] m')

# 实际的物理增量
actual_delta = ee_pos[11] - ee_pos[10]
print(f'实际物理增量:         [{actual_delta[0]:.6f}, {actual_delta[1]:.6f}, {actual_delta[2]:.6f}] m')

print('\n符号对比（是否匹配）:')
for i, axis in enumerate(['X', 'Y', 'Z']):
    sign_orig = np.sign(pos_normalized[10, i])
    sign_old = np.sign(pos_meters_old[10, i])
    sign_new = np.sign(pos_meters_new[10, i])
    sign_actual = np.sign(actual_delta[i])
    
    print(f'{axis}: 原始={sign_orig:+.0f}, 旧scale={sign_old:+.0f}, 新scale={sign_new:+.0f}, 实际={sign_actual:+.0f}')
    
    if sign_orig != sign_actual:
        print(f'  ❌ 原始LIBERO action的符号与实际物理增量不符！')

# 2. 检查当前checkpoint训练时用的scale
print('\n2. 检查已训练checkpoint学到的映射')
print('-' * 80)

print('\n当前checkpoint是用 scale=0.05 训练的:')
print('  - 模型学到的映射: State → Action (scale=0.05的物理单位)')
print('  - 评估时如果用 scale=0.012 转换:')
print('     RDT输出 (仍是0.05scale的) ÷ 0.012 → LIBERO')
print('     相当于放大了 0.05/0.012 ≈ 4.17倍')
print('\n这会导致动作幅度错误，但**不应该**导致方向反转！')

# 3. 检查可能导致方向反转的原因
print('\n3. 可能导致方向反转的原因分析')
print('-' * 80)

print('\n可能性A: State的符号问题')
print('  如果State中的ee_pos符号与action不一致...')

# 检查State
with h5py.File(dataset.file_paths[0], 'r') as f:
    ee_pos = f['data/demo_0/obs/ee_pos'][:]
    
print(f'\nState (ee_pos) 在时刻10: [{ee_pos[10,0]:.4f}, {ee_pos[10,1]:.4f}, {ee_pos[10,2]:.4f}]')
print(f'State (ee_pos) 在时刻11: [{ee_pos[11,0]:.4f}, {ee_pos[11,1]:.4f}, {ee_pos[11,2]:.4f}]')
delta_state = ee_pos[11] - ee_pos[10]
print(f'State变化: [{delta_state[0]:.6f}, {delta_state[1]:.6f}, {delta_state[2]:.6f}]')

print(f'\nAction[10]: [{pos_normalized[10,0]:.4f}, {pos_normalized[10,1]:.4f}, {pos_normalized[10,2]:.4f}]')

print('\n符号一致性检查:')
for i, axis in enumerate(['X', 'Y', 'Z']):
    sign_delta_state = np.sign(delta_state[i])
    sign_action = np.sign(pos_normalized[10, i])
    match = '✅' if sign_delta_state == sign_action else '❌'
    print(f'{axis}: State变化={sign_delta_state:+.0f}, Action={sign_action:+.0f} {match}')

print('\n可能性B: 评估时State转换有误')
print('  如果评估时State的ee_pos符号处理不对...')
print('  需要检查 eval_rdt_libero.py 中的 convert_libero_state_to_rdt 函数')

print('\n可能性C: RDT预训练数据的坐标系不同')
print('  如果RDT预训练用的坐标系与LIBERO相反...')
print('  例如: RDT用 X向左, LIBERO用 X向右')
print('  则需要在转换时翻转对应的轴')

print('\n' + '=' * 80)
print('建议的调试步骤：')
print('=' * 80)
print('1. 用修复缩放因子的代码重新计算数据集统计')
print('2. 从头重新训练一个checkpoint（几百步即可测试）')
print('3. 评估新checkpoint，看方向是否正确')
print('4. 如果还是反的，说明问题不在缩放因子，而在坐标系定义')
print('5. 那时再考虑轴翻转')

