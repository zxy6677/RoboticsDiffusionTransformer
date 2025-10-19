#!/bin/bash

# 单任务过拟合测试脚本
# 严格按照README Fine-Tuning指南配置
# 目标：验证训练和评估代码的正确性

export CUDA_VISIBLE_DEVICES=1

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/single_task_scene10"

# 重要：使用单任务数据集路径（自动检测）
# 优先使用 data/datasets/libero_single_task（远程服务器）
# 备用 datasets/libero_single_task（本地）
if [ -d "./data/datasets/libero_single_task" ]; then
    export LIBERO_DATASET_DIR="./data/datasets/libero_single_task"
    echo "使用数据集路径: data/datasets/libero_single_task"
elif [ -d "./datasets/libero_single_task" ]; then
    export LIBERO_DATASET_DIR="./datasets/libero_single_task"
    echo "使用数据集路径: datasets/libero_single_task"
else
    echo "❌ 错误：未找到单任务数据集！"
    echo "请确保数据集在以下路径之一："
    echo "  - ./data/datasets/libero_single_task"
    echo "  - ./datasets/libero_single_task"
    exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

# 使用单GPU训练（不用DeepSpeed）
python main.py \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=8 \
    --sample_batch_size=8 \
    --max_train_steps=20000 \
    --checkpointing_period=500 \
    --sample_period=500 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --gradient_accumulation_steps=1 \
    --lr_warmup_steps=100

echo ""
echo "训练完成！现在可以评估模型是否学会了这个任务。"
echo "评估命令："
echo "python eval_sim/eval_rdt_libero.py --config configs/base.yaml --pretrained checkpoints/single_task_overfit/checkpoint-XXXX --text_encoder google/t5-v1_1-xxl --vision_encoder google/siglip-so400m-patch14-384 --benchmark libero_90 --num_tasks 1 --max_steps 100 --record_video --video_output_dir videos/single_task_test"
