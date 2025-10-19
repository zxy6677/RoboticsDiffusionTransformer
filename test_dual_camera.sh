#!/bin/bash
# 测试双摄像头配置的评估脚本

echo "========================================"
echo "测试双摄像头评估配置"
echo "========================================"
echo ""
echo "改进点："
echo "  ✅ 自动检测并使用eye_in_hand摄像头"
echo "  ✅ 与训练时的输入保持一致（2个摄像头）"
echo "  ✅ 可能提升精细操作（抓碗）的成功率"
echo ""
echo "配置："
echo "  - exec_horizon: 16 (推荐配置)"
echo "  - 摄像头: agentview + eye_in_hand (如果可用)"
echo ""
echo "========================================"
echo ""

export CUDA_VISIBLE_DEVICES=1

python eval_sim/eval_rdt_libero.py \
  --config configs/base.yaml \
  --pretrained checkpoints/libero_finetune_single_task_2/checkpoint-20000/ema/model.safetensors \
  --text_encoder google/t5-v1_1-xxl \
  --vision_encoder google/siglip-so400m-patch14-384 \
  --benchmark libero_90 \
  --num_tasks 2 \
  --max_steps 250 \
  --exec_horizon 16 \
  --record_video \
  --video_output_dir videos/dual_camera_horizon16

echo ""
echo "========================================"
echo "测试完成！"
echo "查看视频: videos/dual_camera_horizon16/"
echo "========================================"

