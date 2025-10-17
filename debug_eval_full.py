#!/usr/bin/env python3
"""
完整的评估调试脚本 - 打印所有中间值
用于找出机械臂运动方向反转的真正原因
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import torch

# 导入转换函数
from configs.state_vec import STATE_VEC_IDX_MAPPING
from utils.rotation_utils import convert_quaternion_to_6d_rotation, convert_6d_rotation_to_euler, convert_euler_to_6d_rotation
from typing import Dict

def convert_libero_state_to_rdt(obs: Dict, state_dim: int = 128) -> torch.Tensor:
    """将LIBERO observation转换为RDT state格式 (128维)"""
    # 提取LIBERO的状态
    joint_pos = obs['robot0_joint_pos']  # (7,)
    gripper_qpos = obs['robot0_gripper_qpos']  # (2,)
    eef_pos = obs['robot0_eef_pos']  # (3,)
    eef_quat = obs['robot0_eef_quat']  # (4,)
    
    # 初始化RDT state向量 (128维，全0)
    rdt_state = np.zeros(state_dim, dtype=np.float32)
    
    # 按照RDT的统一状态向量填充
    # 1. Joint positions (7维)
    right_arm_indices = [
        STATE_VEC_IDX_MAPPING["right_arm_joint_0"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_1"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_2"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_3"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_4"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_5"],
        STATE_VEC_IDX_MAPPING["right_arm_joint_6"],
    ]
    
    # 填充关节位置
    libero_state = joint_pos
    min_len = min(len(libero_state), len(right_arm_indices))
    rdt_state[right_arm_indices[:min_len]] = libero_state[:min_len]
    
    # 2. Gripper state (1维: 平均值作为gripper开度)
    gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]
    rdt_state[gripper_idx] = np.mean(gripper_qpos)
    
    # 3. End-effector position (3维)
    eef_pos_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_z"],
    ]
    rdt_state[eef_pos_indices[0]] = eef_pos[0]
    rdt_state[eef_pos_indices[1]] = eef_pos[1]
    rdt_state[eef_pos_indices[2]] = eef_pos[2]
    
    # 4. End-effector orientation (6D rotation)
    eef_ori_6d = convert_quaternion_to_6d_rotation(eef_quat.reshape(1, -1))[0]
    
    eef_ori_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"],
    ]
    for i, idx in enumerate(eef_ori_indices):
        rdt_state[idx] = eef_ori_6d[i]
    
    # 重要：训练时State没有归一化，所以评估时也不能归一化！
    return torch.from_numpy(rdt_state).float()

def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """将RDT action (128维) 转换为LIBERO action (7维)"""
    action_128d = rdt_action[0, 0, :].cpu().numpy()
    
    # 步骤1: 提取位置 (物理单位：米)
    pos_x_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_x"]
    pos_y_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_y"]
    pos_z_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
    
    pos_x_meters = action_128d[pos_x_idx]
    pos_y_meters = action_128d[pos_y_idx]
    pos_z_meters = action_128d[pos_z_idx]
    
    # 转换为LIBERO的归一化范围: 米 → [-1, 1]
    pos_x_norm = pos_x_meters / 0.012
    pos_y_norm = pos_y_meters / 0.012
    pos_z_norm = pos_z_meters / 0.012
    
    # 步骤2: 提取6D旋转并转换为欧拉角
    ori_indices = [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
    ]
    
    ori_6d = np.array([action_128d[idx] for idx in ori_indices])
    ori_euler_rad = convert_6d_rotation_to_euler(ori_6d)
    
    # 转换为LIBERO的归一化范围: 弧度 → [-1, 1]
    ori_x_norm = ori_euler_rad[0] / 0.5
    ori_y_norm = ori_euler_rad[1] / 0.5
    ori_z_norm = ori_euler_rad[2] / 0.5
    
    # 步骤3: 提取gripper
    gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]
    gripper_01 = action_128d[gripper_idx]
    
    # 转换为LIBERO的gripper范围: [0,1] → [-1,1]
    gripper_norm = gripper_01 * 2.0 - 1.0
    
    # 组合为LIBERO action (7维)
    libero_action = np.array([
        pos_x_norm, pos_y_norm, pos_z_norm,
        ori_x_norm, ori_y_norm, ori_z_norm,
        gripper_norm
    ])
    
    # Clip到[-1, 1]范围
    libero_action = np.clip(libero_action, -1.0, 1.0)
    
    return libero_action

# 模拟评估的一个步骤
def debug_evaluation_step():
    print("=" * 80)
    print("评估流程完整调试")
    print("=" * 80)
    
    # 1. 模拟LIBERO的观察（从demo中取一个典型值）
    print("\n步骤1: LIBERO观察（输入）")
    print("-" * 80)
    
    obs = {
        'robot0_joint_pos': np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]),
        'robot0_gripper_qpos': np.array([0.04, 0.04]),
        'robot0_eef_pos': np.array([-0.13, 0.11, 1.14]),  # 物理坐标（米）
        'robot0_eef_quat': np.array([0.0, 1.0, 0.0, 0.0]),
    }
    
    print(f"ee_pos: {obs['robot0_eef_pos']}")
    print(f"gripper: {obs['robot0_gripper_qpos']}")
    
    # 2. 转换为RDT State
    print("\n步骤2: 转换为RDT State")
    print("-" * 80)
    
    rdt_state = convert_libero_state_to_rdt(obs)
    print(f"RDT State shape: {rdt_state.shape}")
    
    # 提取关键维度
    eef_pos_x = rdt_state[STATE_VEC_IDX_MAPPING['right_eef_pos_x']].item()
    eef_pos_y = rdt_state[STATE_VEC_IDX_MAPPING['right_eef_pos_y']].item()
    eef_pos_z = rdt_state[STATE_VEC_IDX_MAPPING['right_eef_pos_z']].item()
    
    print(f"RDT State中的ee_pos: [{eef_pos_x:.4f}, {eef_pos_y:.4f}, {eef_pos_z:.4f}]")
    print(f"原始ee_pos:           [{obs['robot0_eef_pos'][0]:.4f}, {obs['robot0_eef_pos'][1]:.4f}, {obs['robot0_eef_pos'][2]:.4f}]")
    
    if np.allclose([eef_pos_x, eef_pos_y, eef_pos_z], obs['robot0_eef_pos'], atol=0.001):
        print("✅ State转换正确，没有改变值")
    else:
        print("❌ State转换有问题")
    
    # 3. 模拟RDT输出
    print("\n步骤3: 模拟RDT输出（假设模型预测正确）")
    print("-" * 80)
    
    # 假设模型应该输出"向右移动0.005米"
    expected_delta = np.array([0.005, 0.0, 0.0])  # 向右（X+）
    print(f"期望的物理增量: {expected_delta} 米")
    
    # 创建一个模拟的RDT输出
    rdt_output = torch.zeros(1, 1, 128)
    rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_x']] = expected_delta[0]
    rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_y']] = expected_delta[1]
    rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_z']] = expected_delta[2]
    rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_gripper_open']] = 0.5  # 半开
    
    print(f"RDT输出的action (128维):")
    print(f"  pos_x: {rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_x']].item():.6f}")
    print(f"  pos_y: {rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_y']].item():.6f}")
    print(f"  pos_z: {rdt_output[0, 0, STATE_VEC_IDX_MAPPING['right_eef_pos_z']].item():.6f}")
    
    # 4. 转换为LIBERO action
    print("\n步骤4: 转换为LIBERO action")
    print("-" * 80)
    
    libero_action = convert_rdt_action_to_libero(rdt_output)
    print(f"LIBERO action (7维): {libero_action}")
    print(f"  pos: [{libero_action[0]:.4f}, {libero_action[1]:.4f}, {libero_action[2]:.4f}]")
    print(f"  ori: [{libero_action[3]:.4f}, {libero_action[4]:.4f}, {libero_action[5]:.4f}]")
    print(f"  gripper: {libero_action[6]:.4f}")
    
    # 5. 验证转换
    print("\n步骤5: 验证转换")
    print("-" * 80)
    
    # 反向计算：LIBERO action应该对应的物理增量
    expected_libero_pos = expected_delta / 0.012  # 物理单位除以scale
    print(f"期望的LIBERO pos: {expected_libero_pos}")
    print(f"实际的LIBERO pos: {libero_action[0:3]}")
    
    if np.allclose(libero_action[0:3], expected_libero_pos, atol=0.01):
        print("✅ 转换正确")
    else:
        print("❌ 转换有问题")
        print(f"差异: {libero_action[0:3] - expected_libero_pos}")
    
    # 6. 检查符号
    print("\n步骤6: 符号检查")
    print("-" * 80)
    
    print(f"期望: 向右移动（X+）")
    print(f"RDT输出: X={expected_delta[0]:.6f} (正数✅)")
    print(f"LIBERO action: X={libero_action[0]:.4f} ({'正数✅' if libero_action[0] > 0 else '负数❌'})")
    
    if libero_action[0] > 0:
        print("\n✅ 如果LIBERO action的X是正数，机械臂应该向右")
        print("   如果实际向左了，问题在LIBERO的action执行或坐标系定义")
    else:
        print("\n❌ LIBERO action的X变成负数了，转换有问题")
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("1. 检查State转换是否正确")
    print("2. 检查RDT输出的符号")
    print("3. 检查转换为LIBERO action的符号")
    print("4. 检查LIBERO执行action时的坐标系")
    print("\n如果所有转换都正确但机械臂还是反向，")
    print("问题可能在：")
    print("- 训练数据本身就有问题（不太可能，已验证）")
    print("- 模型训练时学到了错误的映射")
    print("- LIBERO执行action时的坐标系与我们理解的不同")

if __name__ == "__main__":
    debug_evaluation_step()

