#!/bin/bash

# 从checkpoint-2000恢复训练（修复OOM问题）
# 用于验证是否是sample_batch_size导致的崩溃

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/single_task_scene10_improved"

# 数据集路径自动检测
if [ -d "./data/datasets/libero_single_task" ]; then
    export LIBERO_DATASET_DIR="./data/datasets/libero_single_task"
    echo "使用数据集路径: data/datasets/libero_single_task"
elif [ -d "./datasets/libero_single_task" ]; then
    export LIBERO_DATASET_DIR="./datasets/libero_single_task"
    echo "使用数据集路径: datasets/libero_single_task"
else
    echo "❌ 错误：未找到单任务数据集！"
    exit 1
fi

# 使用8张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "============================================"
echo "从checkpoint-2000恢复训练"
echo "============================================"
echo "修复内容："
echo "  - sample_batch_size: 64 → 8 ✅"
echo "  - sample_period: 500 → 1000 ✅"
echo "  - num_sample_batches: 2 → 1 ✅"
echo "  - dataloader_workers: 8 → 4 ✅"
echo ""
echo "验证目标："
echo "  - 如果能顺利通过第3000步采样 → 证实是OOM问题"
echo "============================================"

# 使用Accelerate启动多GPU训练
accelerate launch \
    --num_processes 8 \
    --multi_gpu \
    --mixed_precision bf16 \
    main.py \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=/share_data/zhukefei/checkpoints/libero_finetune_single_task_full \
    --resume_from_checkpoint="/share_data/zhukefei/checkpoints/libero_finetune_single_task_full/checkpoint-2000" \
    --train_batch_size=16 \
    --sample_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --num_sample_batches=1 \
    --max_train_steps=100000 \
    --checkpointing_period=2000 \
    --checkpoints_total_limit=30 \
    --sample_period=1000 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --lr_warmup_steps=5000 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to="wandb"

echo ""
echo "============================================"
echo "训练完成！"
echo "============================================"

