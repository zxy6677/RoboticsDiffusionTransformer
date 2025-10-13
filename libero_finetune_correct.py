#!/usr/bin/env python3
"""
基于README指导的RDT在LIBERO数据集上的正确微调脚本
支持任务选择和CUDA设备配置
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="RDT在LIBERO上的微调训练")
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO任务ID (0-89)")
    parser.add_argument("--max_steps", type=int, default=15000, help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=32, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--output_dir", type=str, default="checkpoints/libero_finetune", help="输出目录")
    parser.add_argument("--cuda_device", type=str, default="0", help="CUDA设备ID (0, 1, 2, ...)")
    parser.add_argument("--use_deepspeed", action="store_true", help="是否使用DeepSpeed")
    parser.add_argument("--checkpointing_period", type=int, default=500, help="检查点保存周期")
    parser.add_argument("--sample_period", type=int, default=250, help="验证采样周期")
    parser.add_argument("--checkpoints_total_limit", type=int, default=60, help="最大检查点数量")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    args = parser.parse_args()
    
    # 验证任务ID范围
    if args.task_id < 0 or args.task_id > 89:
        print(f"❌ 错误: 任务ID必须在0-89之间，当前值: {args.task_id}")
        return 1
    
    # 设置CUDA设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print(f"🔧 使用CUDA设备: {args.cuda_device}")
    
    # 设置输出目录
    task_name = f"task_{args.task_id:02d}_KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it"
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 开始RDT在LIBERO任务{args.task_id}上的微调训练")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 最大训练步数: {args.max_steps}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 学习率: {args.learning_rate}")
    print(f"💾 检查点保存周期: {args.checkpointing_period}")
    print(f"🔍 验证采样周期: {args.sample_period}")
    print(f"⚡ 使用DeepSpeed: {args.use_deepspeed}")
    
    # 构建训练命令，基于README中的指导
    if args.use_deepspeed:
        cmd = [
            "deepspeed", "main.py",
            "--deepspeed=./configs/zero2.json"
        ]
    else:
        cmd = ["python", "main.py"]
    
    # 添加训练参数
    cmd.extend([
        "--pretrained_model_name_or_path=checkpoints/rdt-1b",
        "--pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl", 
        "--pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384",
        f"--output_dir={output_dir}",
        f"--train_batch_size={args.batch_size}",
        "--sample_batch_size=64",
        f"--max_train_steps={args.max_steps}",
        f"--checkpointing_period={args.checkpointing_period}",
        f"--sample_period={args.sample_period}",
        f"--checkpoints_total_limit={args.checkpoints_total_limit}",
        "--lr_scheduler=constant",
        f"--learning_rate={args.learning_rate}",
        "--mixed_precision=bf16",
        "--dataloader_num_workers=8",
        "--image_aug",
        "--dataset_type=finetune",
        "--state_noise_snr=40",
        "--load_from_hdf5",
        "--report_to=wandb",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--lr_warmup_steps={args.warmup_steps}",
        f"--adam_weight_decay={args.weight_decay}",
        "--adam_beta1=0.9",
        "--adam_beta2=0.999",
        "--adam_epsilon=1e-8"
    ])
    
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


