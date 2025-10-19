#!/bin/bash

# 单任务训练 - 2xGPU版本
# 使用Accelerate进行分布式训练
# 两张4090可以大幅加速训练

export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/single_task_scene10_2gpu"

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

# 使用两张GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

echo "============================================"
echo "使用8张GPU进行分布式训练"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "预期加速: 约1.8-1.9x (2张GPU, 考虑通信开销)"
echo "============================================"

# 使用Accelerate启动多GPU训练
# --num_processes 2: 使用2个进程（2张GPU）
# --mixed_precision bf16: 使用bfloat16混合精度
accelerate launch \
    --num_processes 8 \
    --multi_gpu \
    --mixed_precision bf16 \
    main.py \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=/share_data/zhukefei/checkpoints/libero_finetune_single_task \
    --train_batch_size=4 \
    --sample_batch_size=8 \
    --max_train_steps=30000 \
    --checkpointing_period=10000 \
    --sample_period=500 \
    --checkpoints_total_limit=20 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --gradient_accumulation_steps=1 \
    --lr_warmup_steps=100 \
    --report_to="wandb"

echo ""
echo "============================================"
echo "2-GPU训练完成！"
echo "============================================"
echo ""
echo "训练输出: $OUTPUT_DIR"
echo ""
echo "评估命令："
echo "python eval_sim/eval_rdt_libero.py --config configs/base.yaml --pretrained $OUTPUT_DIR/checkpoint-XXXX/ema/model.safetensors --text_encoder google/t5-v1_1-xxl --vision_encoder google/siglip-so400m-patch14-384 --benchmark libero_90 --num_tasks 2 --max_steps 200 --exec_horizon 16 --record_video --video_output_dir videos/single_task_2gpu_test"

