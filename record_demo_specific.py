#!/usr/bin/env python3
"""
录制指定HDF5文件的指定demo的演示视频
"""

import os
import sys
import cv2
import numpy as np
import h5py
from datetime import datetime

def record_demo_from_hdf5(hdf5_path, demo_id=1):
    """从HDF5文件中提取指定demo的演示视频
    
    Args:
        hdf5_path: HDF5文件路径
        demo_id: demo ID (默认为1，即demo_1)
    """
    
    print(f"📁 演示数据文件: {hdf5_path}")
    print(f"🎯 录制 demo_{demo_id}")
    
    # 检查文件是否存在
    if not os.path.exists(hdf5_path):
        print(f"❌ 文件不存在: {hdf5_path}")
        return None, 0
    
    # 创建输出目录
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # 从文件名提取任务名称
    filename = os.path.basename(hdf5_path)
    task_name = filename.replace('_demo.hdf5', '')
    
    # 创建视频录制器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"demo_{task_name}_demo{demo_id}_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (128, 128))
    
    print(f"🎬 开始录制演示: {video_filename}")
    print(f"📁 输出路径: {video_path}")
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查有哪些demo
            print(f"\n📋 文件中的demo列表:")
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            for key in sorted(demo_keys):
                print(f"  - {key}")
            
            # 选择指定的demo
            demo_key = f'demo_{demo_id}'
            full_demo_path = f'data/{demo_key}'
            
            if full_demo_path not in f:
                print(f"❌ 未找到演示数据: {demo_key}")
                print(f"   可用的demo: {demo_keys}")
                return None, 0
            
            print(f"\n✅ 找到 {demo_key}")
            
            # 获取图像序列
            images = f[f'{full_demo_path}/obs/agentview_rgb'][:]
            actions = f[f'{full_demo_path}/actions'][:]
            
            # 尝试获取其他信息
            try:
                rewards = f[f'{full_demo_path}/rewards'][:]
            except:
                rewards = None
            
            try:
                dones = f[f'{full_demo_path}/dones'][:]
            except:
                dones = None
            
            print(f"\n📊 演示数据信息:")
            print(f"  - 图像序列长度: {len(images)}")
            print(f"  - 动作序列长度: {len(actions)}")
            if rewards is not None:
                print(f"  - 奖励序列长度: {len(rewards)}")
            if dones is not None:
                print(f"  - 完成状态长度: {len(dones)}")
            
            # 显示动作统计
            print(f"\n📏 动作统计:")
            print(f"  - 动作维度: {actions.shape}")
            print(f"  - 位置范围: [{actions[:, 0:3].min():.3f}, {actions[:, 0:3].max():.3f}]")
            print(f"  - 旋转范围: [{actions[:, 3:6].min():.3f}, {actions[:, 3:6].max():.3f}]")
            print(f"  - Gripper范围: [{actions[:, 6].min():.3f}, {actions[:, 6].max():.3f}]")
            
            # 录制视频
            print(f"\n🎥 录制视频...")
            frames = []
            for i, img in enumerate(images):
                # 确保图像是正确的格式
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                
                # 转换图像格式 (RGB -> BGR for OpenCV)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 写入视频帧
                out.write(img_bgr)
                frames.append(img_bgr)
                
                if i % 20 == 0 or i == len(images) - 1:
                    action_str = f"pos=[{actions[i,0]:.2f},{actions[i,1]:.2f},{actions[i,2]:.2f}]"
                    info_str = f"  📍 帧 {i+1}/{len(images)}: {action_str}"
                    if rewards is not None and i < len(rewards):
                        info_str += f", 奖励={rewards[i]:.2f}"
                    if dones is not None and i < len(dones):
                        info_str += f", 完成={dones[i]}"
                    print(info_str)
                
                # 检查是否任务完成
                if dones is not None and i < len(dones) and dones[i]:
                    print(f"  🏁 任务在第{i+1}帧完成")
            
            print(f"\n✅ 视频录制完成!")
            print(f"🎥 视频已保存: {video_path}")
            print(f"📊 总帧数: {len(frames)}")
            
            return video_path, len(frames)
            
    except Exception as e:
        print(f"❌ 录制过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, 0
    
    finally:
        out.release()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='录制LIBERO HDF5演示视频')
    parser.add_argument('--hdf5', type=str, 
                       default='data/datasets/libero_90/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it_demo.hdf5',
                       help='HDF5文件路径')
    parser.add_argument('--demo_id', type=int, default=1,
                       help='Demo ID (默认为1)')
    
    args = parser.parse_args()
    
    print("🎬 LIBERO HDF5演示视频录制器")
    print("=" * 80)
    
    try:
        video_path, frame_count = record_demo_from_hdf5(args.hdf5, args.demo_id)
        
        if video_path:
            print("\n✅ 演示录制完成!")
            print(f"📁 视频文件: {video_path}")
            print(f"📊 帧数: {frame_count}")
            print(f"\n💡 查看视频:")
            print(f"   ls -lh {video_path}")
        else:
            print("\n❌ 录制失败")
            return 1
        
    except Exception as e:
        print(f"❌ 录制失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

