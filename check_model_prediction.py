#!/usr/bin/env python3
"""
检查模型在训练数据上的预测 - 看模型学到了什么
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import h5py
from data.hdf5_libero_dataset import HDF5LIBERODataset
from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
import argparse

def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    print(f"加载checkpoint: {checkpoint_path}")
    
    # 加载配置
    import yaml
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建模型
    rdt = RDTRunner(
        action_dim=128,
        pred_horizon=64,
        config=config,
        lang_token_dim=4096,
        img_token_dim=1152,
        state_token_dim=128,
        max_lang_cond_len=77,
        img_cond_len=1,
        dtype=torch.float32
    )
    
    # 加载权重
    state_dict = torch.load(os.path.join(checkpoint_path, 'unwrapped_model/diffusion_model.bin'), map_location='cpu')
    rdt.load_state_dict(state_dict)
    rdt.eval()
    
    return rdt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint路径')
    args = parser.parse_args()
    
    print("=" * 80)
    print("模型预测检查 - 在训练数据上测试")
    print("=" * 80)
    
    # 加载模型
    rdt = load_checkpoint(args.checkpoint)
    
    # 加载一个训练样本
    dataset = HDF5LIBERODataset()
    sample = dataset.get_item()
    
    # 提取数据
    state = torch.from_numpy(sample['state']).unsqueeze(0).float()  # (1, 1, 128)
    action_gt = torch.from_numpy(sample['actions'][0:1]).unsqueeze(0).float()  # (1, 1, 128) 第一步
    
    # 模拟编码器输出（用零向量代替）
    lang_tokens = torch.zeros(1, 77, 4096)
    lang_attn_mask = torch.ones(1, 77).bool()
    img_tokens = torch.zeros(1, 1, 1152)
    action_mask = torch.ones(1, 1, 128)
    ctrl_freqs = torch.tensor([10.0])
    
    print("\n输入State:")
    pos_indices = [
        STATE_VEC_IDX_MAPPING['right_eef_pos_x'],
        STATE_VEC_IDX_MAPPING['right_eef_pos_y'],
        STATE_VEC_IDX_MAPPING['right_eef_pos_z']
    ]
    
    for i, name in enumerate(['X', 'Y', 'Z']):
        idx = pos_indices[i]
        val = state[0, 0, idx].item()
        print(f"  EEF_{name}: {val:.6f}")
    
    print("\n真实Action (Ground Truth):")
    for i, name in enumerate(['X', 'Y', 'Z']):
        idx = pos_indices[i]
        val = action_gt[0, 0, idx].item()
        print(f"  Action_{name}: {val:.6f} 米")
    
    # 模型预测
    with torch.no_grad():
        pred_action = rdt.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_attn_mask,
            img_tokens=img_tokens,
            state_tokens=state,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs
        )
    
    print("\n模型预测Action:")
    for i, name in enumerate(['X', 'Y', 'Z']):
        idx = pos_indices[i]
        val = pred_action[0, 0, idx].item()
        gt_val = action_gt[0, 0, idx].item()
        print(f"  Pred_{name}: {val:.6f} 米 (GT: {gt_val:.6f})")
    
    print("\n符号对比:")
    for i, name in enumerate(['X', 'Y', 'Z']):
        idx = pos_indices[i]
        pred_val = pred_action[0, 0, idx].item()
        gt_val = action_gt[0, 0, idx].item()
        
        pred_sign = '+' if pred_val >= 0 else '-'
        gt_sign = '+' if gt_val >= 0 else '-'
        match = '✅' if pred_sign == gt_sign else '❌'
        
        print(f"  {name}轴: 预测={pred_sign}, 真实={gt_sign} {match}")
    
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("如果符号全部匹配 -> 模型学对了，问题在评估转换")
    print("如果符号相反 -> 模型学反了，问题在训练数据或训练过程")
    print("如果预测值接近0 -> 模型没学好/训练不足")

if __name__ == "__main__":
    main()

