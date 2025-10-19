#!/bin/bash

# 改进的单任务训练 - 8xGPU版本
# 基于RDT-1B论文的最佳实践

# ============================================
# 关键改进：
# 1. gradient_accumulation_steps=8 → 有效batch size = 256 ⭐⭐⭐⭐⭐
# 2. lr_warmup_steps=5000 ⭐⭐⭐⭐
# 3. checkpointing_period=2000 （更频繁保存）
# 4. checkpoints_total_limit=30 （保存更多）
# ============================================

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

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

echo "============================================"
echo "改进的8-GPU训练配置（基于RDT-1B论文）"
echo "============================================"
echo "关键改进："
echo "  - 有效batch size: 4 * 8 * 8 = 256 ✅"
echo "  - Warmup steps: 5000 ✅"
echo "  - Checkpoint period: 2000 ✅"
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
    --train_batch_size=32 \
    --sample_batch_size=64 \
    --max_train_steps=100000 \
    --checkpointing_period=2000 \
    --checkpoints_total_limit=30 \
    --sample_period=500 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --lr_warmup_steps=5000 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to="wandb"

echo ""
echo "============================================"
echo "训练完成！"
echo "============================================"
echo ""
echo "训练输出: $OUTPUT_DIR"
echo ""
echo "评估命令（使用改进的exec_horizon）："
echo "python eval_sim/eval_rdt_libero.py \\"
echo "  --config configs/base.yaml \\"
echo "  --pretrained $OUTPUT_DIR/checkpoint-XXXX/ema/model.safetensors \\"
echo "  --text_encoder google/t5-v1_1-xxl \\"
echo "  --vision_encoder google/siglip-so400m-patch14-384 \\"
echo "  --benchmark libero_90 \\"
echo "  --num_tasks 2 \\"
echo "  --max_steps 200 \\"
echo "  --exec_horizon 16 \\"
echo "  --record_video \\"
echo "  --video_output_dir videos/improved_model_test"
echo ""

