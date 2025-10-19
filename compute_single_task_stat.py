#!/usr/bin/env python3
"""
为单任务数据集计算正确的action统计信息
用于替代dataset_stat.json中不匹配的libero_90统计
"""

import os
import sys
import h5py
import numpy as np
import json

sys.path.append('.')
from utils.rotation_utils import convert_euler_to_6d_rotation
from configs.state_vec import STATE_VEC_IDX_MAPPING

def compute_single_task_stats(hdf5_path, dataset_name="libero_single_task"):
    """计算单任务HDF5文件的action统计信息"""
    
    print(f"="*80)
    print(f"计算 {dataset_name} 的action统计")
    print(f"="*80)
    print(f"数据文件: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # 获取所有episodes
        episodes = list(f['data'].keys())
        print(f"\n总episodes数: {len(episodes)}")
        
        # 收集所有actions和states
        all_actions_raw = []
        all_states_raw = []
        
        for ep_key in episodes:
            actions = f['data'][ep_key]['actions'][:]
            joint_states = f['data'][ep_key]['obs']['joint_states'][:]
            gripper_states = f['data'][ep_key]['obs']['gripper_states'][:]
            ee_pos = f['data'][ep_key]['obs']['ee_pos'][:]
            ee_ori = f['data'][ep_key]['obs']['ee_ori'][:]
            
            all_actions_raw.append(actions)
            
            # 创建state (17维)
            ee_ori_6d = convert_euler_to_6d_rotation(ee_ori)
            states = np.concatenate([
                joint_states,
                gripper_states[:, 0:1],
                ee_pos,
                ee_ori_6d
            ], axis=1)
            all_states_raw.append(states)
        
        all_actions_raw = np.concatenate(all_actions_raw, axis=0)
        all_states_raw = np.concatenate(all_states_raw, axis=0)
        
        print(f"总actions数: {all_actions_raw.shape[0]}")
        print(f"Action维度: {all_actions_raw.shape[1]}")
        
    # ========== 转换为RDT格式（物理单位）==========
    
    # 1. Position: [-1, 1] -> [-1.2, 1.2] cm
    pos_cm = all_actions_raw[:, 0:3] * 1.2
    
    # 2. Orientation: [-1, 1] -> [-0.5, 0.5] rad, 然后转6D
    ori_euler_rad = all_actions_raw[:, 3:6] * 0.5
    ori_6d = convert_euler_to_6d_rotation(ori_euler_rad)
    
    # 3. Gripper: [-1, 1] -> [0, 1]
    gripper = (all_actions_raw[:, 6:7] + 1.0) / 2.0
    
    # 组合为10D action (RDT格式)
    actions_10d = np.concatenate([pos_cm, ori_6d, gripper], axis=1)
    
    # ========== 映射到128维统一action空间 ==========
    
    ACTION_INDICES = [
        STATE_VEC_IDX_MAPPING["right_eef_pos_x"],    # position x
        STATE_VEC_IDX_MAPPING["right_eef_pos_y"],    # position y
        STATE_VEC_IDX_MAPPING["right_eef_pos_z"],    # position z
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],  # orientation (6D)
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"],
        STATE_VEC_IDX_MAPPING["right_gripper_open"],  # gripper
    ]
    
    STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [
        STATE_VEC_IDX_MAPPING["right_gripper_open"]
    ] + [
        STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
    ] + [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
    ]
    
    # 创建128维action和state数组
    num_samples = actions_10d.shape[0]
    actions_128d = np.zeros((num_samples, 128))
    states_128d = np.zeros((num_samples, 128))
    
    # 填充数据
    actions_128d[:, ACTION_INDICES] = actions_10d
    states_128d[:, STATE_INDICES] = all_states_raw
    
    # ========== 计算统计信息 ==========
    
    action_mean = np.mean(actions_128d, axis=0)
    action_std = np.std(actions_128d, axis=0)
    action_min = np.min(actions_128d, axis=0)
    action_max = np.max(actions_128d, axis=0)
    
    state_mean = np.mean(states_128d, axis=0)
    state_std = np.std(states_128d, axis=0)
    state_min = np.min(states_128d, axis=0)
    state_max = np.max(states_128d, axis=0)
    
    # ========== 打印关键统计 ==========
    
    print(f"\n" + "="*80)
    print("关键统计信息（物理单位）")
    print("="*80)
    
    print(f"\nPosition (cm) - 索引{ACTION_INDICES[0:3]}:")
    print(f"  mean: {action_mean[ACTION_INDICES[0:3]]}")
    print(f"  std:  {action_std[ACTION_INDICES[0:3]]}")
    print(f"  min:  {action_min[ACTION_INDICES[0:3]]}")
    print(f"  max:  {action_max[ACTION_INDICES[0:3]]}")
    
    print(f"\nOrientation 6D - 索引{ACTION_INDICES[3:9]}:")
    print(f"  mean: {action_mean[ACTION_INDICES[3:9]]}")
    print(f"  std:  {action_std[ACTION_INDICES[3:9]]}")
    
    print(f"\nGripper [0,1] - 索引{ACTION_INDICES[9]}:")
    print(f"  mean: {action_mean[ACTION_INDICES[9]]:.4f}")
    print(f"  std:  {action_std[ACTION_INDICES[9]]:.4f}")
    print(f"  min:  {action_min[ACTION_INDICES[9]]:.4f}")
    print(f"  max:  {action_max[ACTION_INDICES[9]]:.4f}")
    
    # ========== 构建结果字典 ==========
    
    result = {
        dataset_name: {
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "state_min": state_min.tolist(),
            "state_max": state_max.tolist(),
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "action_min": action_min.tolist(),
            "action_max": action_max.tolist(),
        }
    }
    
    return result

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="计算单任务HDF5数据的统计信息")
    parser.add_argument("--hdf5_path", type=str, 
                       default="dataset_remote/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5",
                       help="HDF5文件路径")
    parser.add_argument("--dataset_name", type=str,
                       default="libero_single_task",
                       help="数据集名称")
    parser.add_argument("--output", type=str,
                       default="configs/dataset_stat_single_task.json",
                       help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    # 计算统计
    result = compute_single_task_stats(args.hdf5_path, args.dataset_name)
    
    # 保存到文件
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"✅ 统计信息已保存到: {args.output}")
    print("="*80)
    
    print(f"\n📝 使用方法:")
    print(f"1. 更新 data/hdf5_libero_dataset.py 中的数据集名称:")
    print(f"   self.DATASET_NAME = '{args.dataset_name}'")
    print(f"2. 或者将结果添加到 configs/dataset_stat.json")

if __name__ == "__main__":
    main()


