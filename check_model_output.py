#!/usr/bin/env python3
"""
检查模型实际输出的action值
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from transformers import AutoModel
from configs.state_vec import STATE_VEC_IDX_MAPPING

# 指定checkpoint路径（用户应该修改这个）
checkpoint_path = "checkpoints/checkpoint-25000/checkpoint-25000"  # 示例

print("=" * 80)
print("模型输出检查")
print("=" * 80)

# 加载模型
print(f"\n加载checkpoint: {checkpoint_path}")
try:
    model = AutoModel.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    print("请手动指定正确的checkpoint路径")
    sys.exit(1)

# 创建一个简单的测试输入
print("\n创建测试输入...")
device = next(model.parameters()).device
batch_size = 1

# 模拟输入（随机值）
lang_tokens = torch.randn(batch_size, 77, 4096).to(device)
lang_attn_mask = torch.ones(batch_size, 77).bool().to(device)
img_tokens = torch.randn(batch_size, 4, 1152).to(device)
state_tokens = torch.randn(batch_size, 1, 128).to(device)  # 随机state
action_mask = torch.ones(batch_size, 1, 128).to(device)
ctrl_freqs = torch.tensor([10.0]).to(device)

print("输入shape:")
print(f"  lang_tokens: {lang_tokens.shape}")
print(f"  img_tokens: {img_tokens.shape}")
print(f"  state_tokens: {state_tokens.shape}")

# 预测action
print("\n预测action...")
with torch.no_grad():
    try:
        pred_actions = model.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state_tokens,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs
        )
        print(f"✅ 预测成功，输出shape: {pred_actions.shape}")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        sys.exit(1)

# 分析输出
print("\n" + "=" * 80)
print("输出分析")
print("=" * 80)

action_128d = pred_actions[0, 0, :].cpu().numpy()

# 提取位置
pos_x_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_x"]
pos_y_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_y"]
pos_z_idx = STATE_VEC_IDX_MAPPING["right_eef_pos_z"]

pos_x = action_128d[pos_x_idx]
pos_y = action_128d[pos_y_idx]
pos_z = action_128d[pos_z_idx]

print(f"\n预测的位置action (物理单位 - 米):")
print(f"  X: {pos_x:.6f}")
print(f"  Y: {pos_y:.6f}")
print(f"  Z: {pos_z:.6f}")

# 转换为LIBERO格式
pos_x_norm = pos_x / 0.012
pos_y_norm = pos_y / 0.012
pos_z_norm = pos_z / 0.012

print(f"\n转换为LIBERO action (归一化):")
print(f"  X: {pos_x_norm:.4f}")
print(f"  Y: {pos_y_norm:.4f}")
print(f"  Z: {pos_z_norm:.4f}")

# Gripper
gripper_idx = STATE_VEC_IDX_MAPPING["right_gripper_open"]
gripper = action_128d[gripper_idx]
gripper_libero = gripper * 2.0 - 1.0

print(f"\nGripper:")
print(f"  RDT输出: {gripper:.4f} (应该在[0,1])")
print(f"  LIBERO: {gripper_libero:.4f} (应该在[-1,1])")

# 统计所有action值
print(f"\n" + "=" * 80)
print("整个128维action向量统计:")
print("=" * 80)
non_zero = np.count_nonzero(np.abs(action_128d) > 0.0001)
print(f"非零元素数量: {non_zero}/{len(action_128d)}")
print(f"值范围: [{action_128d.min():.6f}, {action_128d.max():.6f}]")
print(f"均值: {action_128d.mean():.6f}")
print(f"标准差: {action_128d.std():.6f}")

# 检查是否所有值都是负的或正的
if np.all(action_128d[np.abs(action_128d) > 0.0001] < 0):
    print("\n❌ 警告: 所有非零action都是负值！这不正常")
elif np.all(action_128d[np.abs(action_128d) > 0.0001] > 0):
    print("\n❌ 警告: 所有非零action都是正值！这不正常")
else:
    print("\n✅ Action值有正有负，看起来正常")

print("\n" + "=" * 80)
print("说明:")
print("这只是随机输入的测试，实际评估时需要真实的观察数据。")
print("如果模型输出异常（全负/全正/全零），可能说明模型训练有问题。")

