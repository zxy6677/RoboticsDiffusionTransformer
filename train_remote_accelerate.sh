#!/bin/bash

# 使用accelerate进行单机多GPU训练
# 更稳定可靠的单机多GPU训练方案

echo "🚀 开始远程服务器A800多GPU训练..."
echo "📊 使用GPU: 0,2,3,4 (避开正在使用的GPU 1)"
echo "🎯 输出目录: /share_data/zhukefei/checkpoints/libero_finetune_v1"
echo "🔧 使用accelerate进行单机多GPU训练"
echo "=" * 60

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

# 创建输出目录
mkdir -p /share_data/zhukefei/checkpoints/libero_finetune_v1

# 使用accelerate进行单机多GPU训练
# 4个GPU，每个GPU的batch_size=8，总batch_size=32
accelerate launch --num_processes=8 --main_process_port=29500 main.py \
    --pretrained_model_name_or_path=checkpoints/rdt-1b \
    --pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl \
    --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 \
    --output_dir=/share_data/zhukefei/checkpoints/libero_finetune_v3 \
    --train_batch_size=4 \
    --sample_batch_size=8 \
    --max_train_steps=200000 \
    --sample_period=500 \
    --lr_scheduler=constant \
    --learning_rate=1e-4 \
    --mixed_precision=bf16 \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type=finetune \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb \
    --gradient_accumulation_steps=2 \
    --lr_warmup_steps=500 \
    --adam_weight_decay=0.01 \
    --adam_beta1=0.9 \
    --adam_beta2=0.999 \
    --adam_epsilon=1e-8 \
    --checkpointing_period=1000 \
    --checkpoints_total_limit=40

echo "✅ 多GPU训练完成!"

