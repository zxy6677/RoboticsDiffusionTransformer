#!/usr/bin/env python3
"""
使用真实的LIBERO演示数据录制演示视频
直接从HDF5文件中提取图像序列
"""

import os
import sys
import cv2
import numpy as np
import h5py
from datetime import datetime

def record_demo_from_hdf5():
    """从HDF5文件中直接提取演示视频 - 录制Task2"""
    
    # Task2的演示数据文件路径 (第三个任务)
    demo_file = '/home/ubuntu/LIBERO/datasets/libero_90/KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5'
    
    print(f"📁 演示数据文件: {demo_file}")
    print(f"🎯 录制任务: Task2 - put_the_black_bowl_in_the_top_drawer_of_the_cabinet")
    
    # 创建输出目录
    output_dir = "videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建视频录制器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"libero_real_demo_task_02_KITCHEN_SCENE10_put_the_black_bowl__{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (128, 128))
    
    print(f"🎬 开始录制真实演示: {video_filename}")
    print(f"📁 输出路径: {video_path}")
    
    try:
        with h5py.File(demo_file, 'r') as f:
            # 选择第一个演示序列
            demo_key = 'data/demo_0'
            if demo_key not in f:
                print(f"❌ 未找到演示数据: {demo_key}")
                return None, 0
            
            # 获取图像序列
            images = f[f'{demo_key}/obs/agentview_rgb'][:]
            actions = f[f'{demo_key}/actions'][:]
            rewards = f[f'{demo_key}/rewards'][:]
            dones = f[f'{demo_key}/dones'][:]
            
            print(f"📊 演示数据信息:")
            print(f"  - 图像序列长度: {len(images)}")
            print(f"  - 动作序列长度: {len(actions)}")
            print(f"  - 奖励序列长度: {len(rewards)}")
            print(f"  - 完成状态长度: {len(dones)}")
            
            # 录制视频
            frames = []
            for i, img in enumerate(images):
                # 转换图像格式 (RGB -> BGR for OpenCV)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # 写入视频帧
                out.write(img_bgr)
                frames.append(img_bgr)
                
                if i % 20 == 0 or i == len(images) - 1:  # 每20帧显示一次，或者最后一步
                    reward = rewards[i] if i < len(rewards) else 0
                    done = dones[i] if i < len(dones) else False
                    print(f"  📍 帧 {i+1}: 奖励={reward}, 完成={done}")
                
                # 如果任务完成，显示完成信息但不提前结束
                if i < len(dones) and dones[i]:
                    print(f"  🏁 任务在第{i+1}帧完成，奖励={rewards[i]}")
                    # 不提前结束，继续录制到最后
            
            print(f"🎥 真实演示视频已保存: {video_path}")
            print(f"📊 总帧数: {len(frames)}")
            
            return video_path, len(frames)
            
    except Exception as e:
        print(f"❌ 录制过程中出错: {e}")
        return None, 0
    
    finally:
        out.release()

def main():
    print("🎬 LIBERO_90真实演示视频录制器 (从HDF5)")
    print("🎯 录制任务: Task2 - put_the_black_bowl_in_the_top_drawer_of_the_cabinet")
    print("=" * 80)
    
    try:
        video_path, frame_count = record_demo_from_hdf5()
        
        if video_path:
            print("\n✅ 真实演示录制完成!")
            print(f"📁 视频文件: {video_path}")
            print(f"📊 帧数: {frame_count}")
        else:
            print("\n❌ 录制失败")
            return 1
        
    except Exception as e:
        print(f"❌ 录制失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
