# 2-GPU训练使用说明

## 🎯 方案：Accelerate多GPU（已选择）

使用Accelerate的原生多GPU支持，简单高效。

---

## 🚀 快速开始

### 1. 给脚本添加执行权限
```bash
chmod +x train_single_task_2gpu.sh
```

### 2. 启动训练
```bash
bash train_single_task_2gpu.sh
```

或者直接运行：
```bash
./train_single_task_2gpu.sh
```

---

## 📊 关键配置变化

### 与单GPU训练的对比

| 配置项 | 单GPU | 2-GPU | 说明 |
|--------|-------|-------|------|
| **GPU设备** | `CUDA_VISIBLE_DEVICES=1` | `CUDA_VISIBLE_DEVICES=0,1` | 使用GPU 0和1 |
| **启动方式** | `python main.py` | `accelerate launch --num_processes 2 --multi_gpu` | 多进程启动 |
| **Batch Size** | 8 | 16 | 2倍（每卡8） |
| **Dataloader Workers** | 4 | 8 | 2倍加速数据加载 |
| **训练速度** | 1.0x | ~1.8-1.9x | 实际加速比 |

### 为什么不是2.0x加速？

- GPU间通信开销：~5-10%
- 梯度同步时间：~5%
- 实际加速比：1.8-1.9x（非常好的结果！）

---

## ⚙️ 参数说明

### Accelerate参数

```bash
accelerate launch \
    --num_processes 2 \        # 使用2个进程（对应2张GPU）
    --multi_gpu \               # 启用多GPU模式
    --mixed_precision bf16 \    # bfloat16混合精度
    main.py \
    ...
```

### 训练参数优化

```bash
--train_batch_size=16          # 总batch size (每GPU=8)
--sample_batch_size=16         # 采样batch size
--dataloader_num_workers=8     # 数据加载worker数
```

---

## 💡 性能优化建议

### 1. Batch Size调优

根据显存使用情况调整：

```bash
# 保守（安全）
--train_batch_size=16    # 每GPU: 8

# 中等（推荐）
--train_batch_size=20    # 每GPU: 10

# 激进（如果显存够）
--train_batch_size=24    # 每GPU: 12
```

**监控命令**：
```bash
# 另开一个终端，监控GPU使用
watch -n 1 nvidia-smi
```

### 2. Dataloader Workers调优

```bash
# CPU核心较少
--dataloader_num_workers=4

# CPU核心充足（推荐）
--dataloader_num_workers=8

# CPU核心很多
--dataloader_num_workers=12
```

### 3. Gradient Accumulation

如果想用更大的有效batch size但显存不够：

```bash
--train_batch_size=16
--gradient_accumulation_steps=2
# 有效batch size = 16 * 2 = 32
```

---

## 🔍 训练监控

### 1. 查看GPU使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看一次
nvidia-smi
```

**期望看到**：
- 两张GPU的Utilization都在80-100%
- Memory使用相近
- Power使用都在300W+（4090满载）

### 2. 查看训练日志

```bash
# 查看最新的训练输出
tail -f checkpoints/single_task_scene10_2gpu/training.log

# 或者直接在运行终端观察
```

### 3. 训练速度估算

```python
# 假设单GPU训练速度：5秒/step
# 2-GPU预期速度：2.7秒/step (1.8x加速)
# 20000步总时间：
#   单GPU: 20000 * 5 / 3600 = 27.8小时
#   2-GPU: 20000 * 2.7 / 3600 = 15小时 ✅
```

---

## 🐛 常见问题

### Q1: 两张GPU利用率不均衡？

**原因**：数据不均匀或batch size太小

**解决**：
```bash
# 增加batch size
--train_batch_size=20  # 或更大
```

### Q2: 训练速度没有提升？

**检查清单**：
1. 确认两张GPU都在工作：`nvidia-smi`
2. 数据加载是否是瓶颈：增加`--dataloader_num_workers`
3. Batch size是否足够：至少16（每GPU 8）

### Q3: 显存不足OOM？

**解决方案**：
```bash
# 方案1：减小batch size
--train_batch_size=12  # 每GPU: 6

# 方案2：使用gradient accumulation
--train_batch_size=12
--gradient_accumulation_steps=2
```

### Q4: 如何只用单张GPU训练？

使用原来的脚本：
```bash
bash train_single_task.sh
```

---

## 📈 性能对比预期

| 指标 | 单GPU (4090) | 2-GPU (4090x2) |
|------|--------------|----------------|
| **Batch Size** | 8 | 16 |
| **步骤时间** | ~5秒/step | ~2.7秒/step |
| **20K步总时间** | ~28小时 | ~15小时 ✅ |
| **GPU利用率** | 90-100% | 85-95% (each) |
| **显存使用** | ~35GB | ~35GB (each) |

---

## ✅ 验证训练正常运行

### 启动后应该看到：

```
============================================
使用2张GPU进行分布式训练
GPU设备: 0,1
预期加速: 约1.8-1.9x (2张GPU, 考虑通信开销)
============================================

🔍 检测到单任务训练，使用 libero_single_task 统计信息
✅ 使用正确的单任务统计（Position std修复）

Distributed training: world_size=2, rank=0
Distributed training: world_size=2, rank=1

Steps:   0%|          | 0/20000 [00:00<?, ?it/s]
...
```

### 关键信息确认：

- ✅ `world_size=2`：确认2个进程
- ✅ `rank=0`, `rank=1`：两个GPU都在工作
- ✅ `检测到单任务训练`：使用正确统计
- ✅ GPU利用率：两张都在80%+

---

## 🎯 训练完成后

### 1. 检查checkpoint
```bash
ls -lh checkpoints/single_task_scene10_2gpu/checkpoint-*/
```

### 2. 评估模型
```bash
export CUDA_VISIBLE_DEVICES=0  # 评估只需1张GPU

python eval_sim/eval_rdt_libero.py \
    --config configs/base.yaml \
    --pretrained checkpoints/single_task_scene10_2gpu/checkpoint-19000/ema/model.safetensors \
    --text_encoder google/t5-v1_1-xxl \
    --vision_encoder google/siglip-so400m-patch14-384 \
    --benchmark libero_90 \
    --num_tasks 2 \
    --max_steps 200 \
    --exec_horizon 16 \
    --record_video \
    --video_output_dir videos/single_task_2gpu_eval
```

---

## 💾 节省时间估算

| 训练步数 | 单GPU时间 | 2-GPU时间 | 节省时间 |
|---------|-----------|-----------|----------|
| 5,000步 | ~7小时 | ~4小时 | **3小时** |
| 10,000步 | ~14小时 | ~7.5小时 | **6.5小时** |
| 20,000步 | ~28小时 | ~15小时 | **13小时** ✅ |

**结论：20K步训练可以节省半天时间！**

---

## 🔧 高级配置（可选）

### 1. 自定义GPU选择

```bash
# 使用GPU 1和2（而不是0和1）
export CUDA_VISIBLE_DEVICES=1,2
bash train_single_task_2gpu.sh
```

### 2. 调整混合精度

```bash
# 如果遇到数值稳定性问题，可以改用fp16
accelerate launch \
    --mixed_precision fp16 \  # 改为fp16
    ...
```

### 3. 调整通信后端

```bash
# 如果遇到通信问题，可以尝试
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # 禁用P2P（如果有问题）
```

---

## 📝 总结

### 推荐配置（2x 4090）

```bash
✅ GPU设备: 0,1
✅ Batch Size: 16-20
✅ Workers: 8
✅ 混合精度: bf16
✅ 预期加速: 1.8-1.9x
✅ 训练时间: 15小时（20K步）
```

### 立即开始训练

```bash
bash train_single_task_2gpu.sh
```

**使用修复后的统计信息，预期任务2的成功率会显著提升！** 🚀

