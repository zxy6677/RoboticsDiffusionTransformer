#!/bin/bash
# 远程服务器训练前完整检查脚本
# 用法: bash remote_training_check.sh

echo "=========================================="
echo "🔍 远程训练环境检查"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# 检查函数
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# ============================================
# 1. 基础环境检查
# ============================================
echo "1️⃣  基础环境检查"
echo "----------------------------------------"

# 检查Python版本
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    check_pass "Python: $PYTHON_VERSION"
else
    check_fail "Python未安装"
fi

# 检查conda环境
if [ ! -z "$CONDA_DEFAULT_ENV" ]; then
    check_pass "Conda环境: $CONDA_DEFAULT_ENV"
else
    check_warn "未激活conda环境，请运行: conda activate rdt"
fi

# 检查PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU数量: {torch.cuda.device_count()}')" 2>/dev/null
if [ $? -eq 0 ]; then
    check_pass "PyTorch和CUDA检查通过"
else
    check_fail "PyTorch或CUDA有问题"
fi

echo ""

# ============================================
# 2. GPU检查
# ============================================
echo "2️⃣  GPU检查"
echo "----------------------------------------"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    check_pass "检测到 $GPU_COUNT 张GPU"
    
    # 显示GPU状态
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader | while read line; do
        echo "   GPU: $line"
    done
    
    # 检查是否有8张GPU
    if [ "$GPU_COUNT" -ge 8 ]; then
        check_pass "GPU数量充足 (≥8张)"
    else
        check_warn "GPU数量少于8张，脚本配置为8-GPU训练"
    fi
else
    check_fail "nvidia-smi不可用"
fi

echo ""

# ============================================
# 3. 数据集检查
# ============================================
echo "3️⃣  数据集检查"
echo "----------------------------------------"

# 检查数据集目录
if [ -d "./data/datasets/libero_single_task" ]; then
    DATASET_DIR="./data/datasets/libero_single_task"
    check_pass "找到数据集目录: $DATASET_DIR"
elif [ -d "./datasets/libero_single_task" ]; then
    DATASET_DIR="./datasets/libero_single_task"
    check_pass "找到数据集目录: $DATASET_DIR"
else
    check_fail "未找到数据集目录"
    DATASET_DIR=""
fi

# 检查数据集文件
if [ ! -z "$DATASET_DIR" ]; then
    HDF5_COUNT=$(find "$DATASET_DIR" -name "*.hdf5" | wc -l)
    if [ "$HDF5_COUNT" -gt 0 ]; then
        check_pass "找到 $HDF5_COUNT 个HDF5文件"
        
        # 列出数据集文件及大小
        echo "   数据集文件:"
        find "$DATASET_DIR" -name "*.hdf5" -exec ls -lh {} \; | awk '{print "   - " $9 " (" $5 ")"}'
        
        # 计算总大小
        TOTAL_SIZE=$(find "$DATASET_DIR" -name "*.hdf5" -exec du -ch {} + | grep total | awk '{print $1}')
        echo "   总大小: $TOTAL_SIZE"
    else
        check_fail "数据集目录中没有HDF5文件"
    fi
fi

echo ""

# ============================================
# 4. 数据集统计信息检查 ⭐ 重要
# ============================================
echo "4️⃣  数据集统计信息检查 ⭐"
echo "----------------------------------------"

if [ -f "configs/dataset_stat.json" ]; then
    check_pass "找到 configs/dataset_stat.json"
    
    # 检查是否包含 libero_single_task
    if grep -q '"libero_single_task"' configs/dataset_stat.json; then
        check_pass "包含 libero_single_task 统计信息"
        
        # 显示统计信息摘要
        echo "   统计信息摘要:"
        python3 << 'EOF'
import json
try:
    with open('configs/dataset_stat.json', 'r') as f:
        stats = json.load(f)
    if 'libero_single_task' in stats:
        s = stats['libero_single_task']
        print(f"   - 动作均值范围: [{min(s['action_mean']):.4f}, {max(s['action_mean']):.4f}]")
        print(f"   - 动作标准差范围: [{min(s['action_std']):.4f}, {max(s['action_std']):.4f}]")
        print(f"   - 状态标准差范围: [{min(s['state_std']):.4f}, {max(s['state_std']):.4f}]")
        
        # 检查是否有异常值
        if max(s['action_std']) > 100:
            print("   ⚠️  警告: 动作标准差过大，可能需要重新计算")
        if max(s['state_std']) > 100:
            print("   ⚠️  警告: 状态标准差过大，可能需要重新计算")
except Exception as e:
    print(f"   ✗ 解析统计信息失败: {e}")
EOF
        
    else
        check_fail "缺少 libero_single_task 统计信息！需要重新计算"
        echo ""
        echo "   📝 如何重新计算统计信息:"
        echo "   python compute_single_task_stat.py"
        echo ""
    fi
else
    check_fail "缺少 configs/dataset_stat.json"
fi

# 检查是否需要重新计算统计信息
echo ""
echo "❓ 是否需要重新计算统计信息？"
echo "   只在以下情况需要重新计算:"
echo "   1. 远程数据集与本地数据集内容不同"
echo "   2. configs/dataset_stat.json 缺少 libero_single_task"
echo "   3. 统计值看起来异常（如标准差>100）"
echo ""
echo "   验证数据集一致性:"
if [ ! -z "$DATASET_DIR" ]; then
    find "$DATASET_DIR" -name "*.hdf5" -exec md5sum {} \; > /tmp/remote_dataset_checksums.txt 2>/dev/null
    if [ -f /tmp/remote_dataset_checksums.txt ]; then
        echo "   数据集文件校验和:"
        cat /tmp/remote_dataset_checksums.txt | awk '{print "   " $2 ": " $1}'
        echo ""
        echo "   💡 提示: 将此校验和与本地比较，如果不同则需重新计算统计"
    fi
fi

echo ""

# ============================================
# 5. 配置文件检查
# ============================================
echo "5️⃣  配置文件检查"
echo "----------------------------------------"

# 检查base.yaml
if [ -f "configs/base.yaml" ]; then
    check_pass "找到 configs/base.yaml"
else
    check_fail "缺少 configs/base.yaml"
fi

# 检查其他配置
CONFIG_FILES=(
    "configs/dataset_control_freq.json"
    "configs/dataset_img_keys.json"
    "configs/finetune_datasets_single_task.json"
    "configs/finetune_sample_weights_single_task.json"
)

for config in "${CONFIG_FILES[@]}"; do
    if [ -f "$config" ]; then
        check_pass "找到 $config"
    else
        check_warn "缺少 $config (可能不影响训练)"
    fi
done

echo ""

# ============================================
# 6. 预训练模型检查
# ============================================
echo "6️⃣  预训练模型检查"
echo "----------------------------------------"

PRETRAINED_MODEL="checkpoints/rdt-1b/model.safetensors"
if [ -f "$PRETRAINED_MODEL" ]; then
    MODEL_SIZE=$(ls -lh "$PRETRAINED_MODEL" | awk '{print $5}')
    check_pass "找到预训练模型: $PRETRAINED_MODEL ($MODEL_SIZE)"
    
    # 检查模型大小是否合理（应该是几GB）
    MODEL_SIZE_BYTES=$(stat -f%z "$PRETRAINED_MODEL" 2>/dev/null || stat -c%s "$PRETRAINED_MODEL" 2>/dev/null)
    if [ ! -z "$MODEL_SIZE_BYTES" ]; then
        if [ "$MODEL_SIZE_BYTES" -gt 1000000000 ]; then  # > 1GB
            check_pass "模型大小合理"
        else
            check_warn "模型文件较小，可能不完整"
        fi
    fi
else
    check_fail "缺少预训练模型: $PRETRAINED_MODEL"
fi

# 检查文本和视觉编码器
TEXT_ENCODER="google/t5-v1_1-xxl"
VISION_ENCODER="google/siglip-so400m-patch14-384"

if [ -d "$TEXT_ENCODER" ]; then
    check_pass "找到文本编码器: $TEXT_ENCODER"
else
    check_warn "本地未找到文本编码器，将从HuggingFace下载"
fi

if [ -d "$VISION_ENCODER" ]; then
    check_pass "找到视觉编码器: $VISION_ENCODER"
else
    check_warn "本地未找到视觉编码器，将从HuggingFace下载"
fi

echo ""

# ============================================
# 7. 输出目录检查
# ============================================
echo "7️⃣  输出目录检查"
echo "----------------------------------------"

OUTPUT_DIR="./checkpoints/single_task_scene10_2gpu"
if [ -d "$OUTPUT_DIR" ]; then
    check_warn "输出目录已存在: $OUTPUT_DIR"
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | wc -l)
    if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
        echo "   已有 $CHECKPOINT_COUNT 个checkpoint"
        echo "   最新checkpoint:"
        find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -3 | xargs -I {} basename {}
    fi
else
    check_pass "输出目录不存在，将自动创建"
fi

# 检查磁盘空间
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
check_pass "可用磁盘空间: $DISK_AVAIL"

echo ""

# ============================================
# 8. 训练脚本检查
# ============================================
echo "8️⃣  训练脚本检查"
echo "----------------------------------------"

TRAIN_SCRIPT="train_single_task_2gpu.sh"
if [ -f "$TRAIN_SCRIPT" ]; then
    check_pass "找到训练脚本: $TRAIN_SCRIPT"
    
    # 检查执行权限
    if [ -x "$TRAIN_SCRIPT" ]; then
        check_pass "脚本有执行权限"
    else
        check_warn "脚本无执行权限，运行: chmod +x $TRAIN_SCRIPT"
    fi
    
    # 检查脚本配置
    echo "   训练配置:"
    grep "num_processes" "$TRAIN_SCRIPT" | head -1 | sed 's/^/   /'
    grep "train_batch_size" "$TRAIN_SCRIPT" | head -1 | sed 's/^/   /'
    grep "max_train_steps" "$TRAIN_SCRIPT" | head -1 | sed 's/^/   /'
else
    check_fail "缺少训练脚本: $TRAIN_SCRIPT"
fi

echo ""

# ============================================
# 9. Python依赖检查
# ============================================
echo "9️⃣  Python依赖检查"
echo "----------------------------------------"

REQUIRED_PACKAGES=(
    "torch"
    "accelerate"
    "transformers"
    "safetensors"
    "h5py"
    "PIL"
    "numpy"
    "einops"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        check_pass "$package ($VERSION)"
    else
        check_fail "$package 未安装"
    fi
done

echo ""

# ============================================
# 10. 网络和HuggingFace检查
# ============================================
echo "🔟 网络和HuggingFace检查"
echo "----------------------------------------"

# 检查HuggingFace缓存目录
if [ ! -z "$HF_HOME" ]; then
    check_pass "HF_HOME: $HF_HOME"
elif [ ! -z "$HUGGINGFACE_HUB_CACHE" ]; then
    check_pass "HUGGINGFACE_HUB_CACHE: $HUGGINGFACE_HUB_CACHE"
else
    check_warn "未设置HuggingFace缓存目录环境变量"
    echo "   默认使用: ~/.cache/huggingface"
fi

# 检查网络连接（可选）
if command -v curl &> /dev/null; then
    if curl -s --connect-timeout 3 https://huggingface.co > /dev/null; then
        check_pass "可以连接到HuggingFace"
    else
        check_warn "无法连接到HuggingFace（如果使用本地模型则无影响）"
    fi
fi

echo ""

# ============================================
# 总结
# ============================================
echo "=========================================="
echo "📊 检查总结"
echo "=========================================="
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！可以开始训练${NC}"
    echo ""
    echo "🚀 启动训练:"
    echo "   bash train_single_task_2gpu.sh"
    echo ""
    echo "或后台运行:"
    echo "   nohup bash train_single_task_2gpu.sh > train.log 2>&1 &"
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ 有 $WARNINGS 个警告，但可以尝试训练${NC}"
    echo ""
    echo "建议先解决警告项，或者直接尝试运行:"
    echo "   bash train_single_task_2gpu.sh"
else
    echo -e "${RED}✗ 发现 $ERRORS 个错误，$WARNINGS 个警告${NC}"
    echo ""
    echo "请先解决上述错误再开始训练"
fi

echo ""
echo "=========================================="
echo ""

# 返回错误码
if [ $ERRORS -gt 0 ]; then
    exit 1
else
    exit 0
fi

