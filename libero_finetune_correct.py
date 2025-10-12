#!/usr/bin/env python3
"""
基于README指导的RDT在LIBERO数据集上的正确微调脚本
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="RDT在LIBERO上的微调训练")
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO任务ID")
    parser.add_argument("--max_steps", type=int, default=10000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--output_dir", type=str, default="checkpoints/libero_finetune", help="输出目录")
    
    args = parser.parse_args()
    
    # 设置输出目录
    task_name = f"task_{args.task_id:02d}_KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it"
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 开始RDT在LIBERO任务{args.task_id}上的微调训练")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 最大训练步数: {args.max_steps}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 学习率: {args.learning_rate}")
    
    # 构建训练命令，基于README中的指导
    cmd = [
        "python", "main.py",
        "--pretrained_model_name_or_path=checkpoints/rdt-1b",
        "--pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl", 
        "--pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384",
        f"--output_dir={output_dir}",
        f"--train_batch_size={args.batch_size}",
        "--sample_batch_size=64",
        f"--max_train_steps={args.max_steps}",
        "--checkpointing_period=1000",
        "--sample_period=500",
        "--checkpoints_total_limit=40",
        "--lr_scheduler=constant",
        f"--learning_rate={args.learning_rate}",
        "--mixed_precision=bf16",
        "--dataloader_num_workers=8",
        "--image_aug",
        "--dataset_type=finetune",
        "--state_noise_snr=40",
        "--load_from_hdf5",
        "--report_to=wandb"
    ]
    
    print(f"🔧 执行命令: {' '.join(cmd)}")
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
