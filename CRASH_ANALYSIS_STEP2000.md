# 训练在第2000步崩溃原因分析

## 📊 已知信息

### 训练配置
```bash
- 硬件: 8×A800 (80G显存/卡)
- train_batch_size: 16 (per device)
- sample_batch_size: 64 (per device) ⚠️
- gradient_accumulation_steps: 2
- sample_period: 500
- checkpointing_period: 2000
- num_sample_batches: 2 (默认值)
- dataloader_num_workers: 8
```

### 已确认事实
1. ✅ checkpoint-2000已成功保存（9.2GB，完整）
2. ✅ 磁盘空间充足（311TB可用）
3. ⚠️ 训练在第2000步崩溃
4. ⚠️ 日志已丢失，无法直接查看错误

## 🔍 崩溃原因分析（按可能性排序）

### ⭐⭐⭐⭐⭐ 原因1：采样时显存溢出（OOM）- 最可能

**问题描述：**
```python
sample_batch_size = 64  # per device
num_gpus = 8
num_sample_batches = 2

# 每次采样处理的总样本数
总样本数 = 64 × 8 × 2 = 1024 samples ⚠️⚠️⚠️
```

**为什么会OOM：**

1. **训练阶段显存使用（正常）：**
   ```
   单卡batch = 16 samples
   梯度累积 = 2
   实际前向传播 = 16 samples/step
   显存占用：~30-40GB（安全）
   ```

2. **采样阶段显存使用（危险）：**
   ```
   单卡batch = 64 samples  ← 4倍于训练！
   实际前向传播 = 64 samples/step
   显存占用：~70-80GB（接近上限！）
   ```

3. **第2000步的特殊情况：**
   ```
   第2000步会同时触发：
   - ✅ 保存checkpoint (2000 % 2000 == 0)
   - ✅ 执行采样 (2000 % 500 == 0)
   
   时间线：
   1. 开始保存checkpoint → 显存使用正常
   2. checkpoint保存完成 → 释放临时显存
   3. 开始采样评估 → 加载64 samples/GPU
   4. 💥 OOM崩溃！
   ```

**证据：**
- sample_batch_size=64 远大于 train_batch_size=16
- 代码注释明确说明是 "per device"
- checkpoint已完整保存，说明崩溃发生在checkpoint之后
- 第2000步正好是sample_period的整数倍

**解决方案：**
```bash
# 方案1：降低sample_batch_size（推荐）
--sample_batch_size=8  # 从64降到8

# 方案2：减少采样批次
--num_sample_batches=1  # 从2降到1

# 方案3：降低采样频率
--sample_period=1000  # 从500增加到1000

# 最佳配置：
--sample_batch_size=8 \
--num_sample_batches=2 \
--sample_period=1000
```

---

### ⭐⭐⭐⭐ 原因2：DataLoader workers内存不足

**问题描述：**
```bash
dataloader_num_workers=8  # 每个进程有8个worker
num_processes=8            # 8个GPU进程
总workers = 8 × 8 = 64个worker进程
```

**为什么会出问题：**

1. **全步骤枚举策略的内存需求：**
   ```python
   # 数据集初始化时预构建索引
   self.all_steps = []  # 存储所有 (file_path, episode_key, step_id)
   
   # 假设50个demo，每个demo 300步
   总样本数 = 50 × 300 = 15,000 steps
   
   # 64个worker，每个都需要加载完整索引
   总内存 = 15,000 × 64 × (索引大小) ≈ 数GB
   ```

2. **HDF5文件句柄泄露：**
   - 每个worker打开HDF5文件
   - 可能未正确关闭
   - 长时间运行导致内存累积

**解决方案：**
```bash
# 减少worker数量
--dataloader_num_workers=4  # 从8降到4
```

---

### ⭐⭐⭐ 原因3：WandB日志缓冲区满

**问题描述：**
```bash
--report_to="wandb"
--sample_period=500
```

每500步采样一次，产生大量指标：
- 各数据集的loss
- 各维度的误差
- 可能还有图像/视频

**为什么会崩溃：**
1. WandB日志缓冲区累积
2. 第2000步时缓冲区满
3. 上传失败/阻塞导致进程挂起

**解决方案：**
```bash
# 增加采样间隔
--sample_period=1000

# 或暂时禁用WandB测试
--report_to="tensorboard"
```

---

### ⭐⭐ 原因4：全步骤枚举的边界问题

**问题描述：**
新实现的全步骤枚举可能在某些边界情况有bug。

**可能的bug：**
```python
# 如果某个demo的步数 < action_chunk_size(64)
for step_id in range(max(0, num_steps - self.CHUNK_SIZE + 1)):
    # num_steps=50, CHUNK_SIZE=64
    # range(max(0, 50-64+1)) = range(max(0, -13)) = range(0)
    # 该demo不会产生任何样本 ← 可能导致某些边界问题
```

**解决方案：**
检查数据集构建是否有警告或错误。

---

## 🎯 推荐解决方案（按优先级）

### 立即修复（必须）

**修改 `train_single_task_improved.sh`：**

```bash
# 第59-60行，修改为：
--train_batch_size=16 \
--sample_batch_size=8 \          # 从64改为8 ⭐⭐⭐⭐⭐
--gradient_accumulation_steps=2 \
```

**原因：**
- sample_batch_size=64 导致采样时显存使用量是训练时的4倍
- 8张卡同时采样：64×8=512 samples，显存压力巨大
- 改为8后：8×8=64 samples，与训练时相当

---

### 可选优化

```bash
# 1. 减少worker数量（降低内存压力）
--dataloader_num_workers=4 \

# 2. 降低采样频率（减少中断）
--sample_period=1000 \

# 3. 减少采样批次（加快采样）
--num_sample_batches=1 \
```

---

## 📝 修改后的配置

```bash
accelerate launch \
    --num_processes 8 \
    --multi_gpu \
    --mixed_precision bf16 \
    main.py \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=/share_data/zhukefei/checkpoints/libero_finetune_single_task_full \
    --train_batch_size=16 \
    --sample_batch_size=8 \              # ⭐ 修改点1
    --gradient_accumulation_steps=2 \
    --max_train_steps=100000 \
    --checkpointing_period=2000 \
    --checkpoints_total_limit=30 \
    --sample_period=1000 \               # ⭐ 修改点2（可选）
    --num_sample_batches=1 \             # ⭐ 修改点3（可选）
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --lr_warmup_steps=5000 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \         # ⭐ 修改点4（可选）
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to="wandb"
```

---

## 🔬 验证方法

### 方案1：从checkpoint-2000继续训练

```bash
# 添加 --resume_from_checkpoint 参数
accelerate launch \
    --num_processes 8 \
    --multi_gpu \
    --mixed_precision bf16 \
    main.py \
    --resume_from_checkpoint="/share_data/zhukefei/checkpoints/libero_finetune_single_task_full/checkpoint-2000" \
    ... (其他参数同上)
```

**观察点：**
- 如果能顺利通过2000步后的采样 → 证实是sample_batch_size问题
- 如果再次在相同位置崩溃 → 可能是其他原因

---

### 方案2：禁用采样测试

```bash
# 临时禁用采样
--sample_period=-1
```

**观察点：**
- 如果不再崩溃 → 100%证实是采样相关问题
- 可以让训练先跑起来，稍后再启用采样

---

## 📊 显存使用对比

| 阶段 | 单卡Batch | 总样本数 | 预估显存 | A800可用 | 状态 |
|------|----------|---------|---------|----------|------|
| 训练 | 16 | 128 | ~35GB | 80GB | ✅ 安全 |
| 采样(旧) | 64 | 512 | ~75GB | 80GB | ⚠️ 危险 |
| 采样(新) | 8 | 64 | ~35GB | 80GB | ✅ 安全 |

---

## 🎯 结论

**最可能的原因：** `sample_batch_size=64` 在采样时导致显存溢出。

**核心证据：**
1. checkpoint-2000保存成功（说明训练正常）
2. 第2000步正好触发采样（2000 % 500 == 0）
3. sample_batch_size=64 是 train_batch_size=16 的4倍
4. 没有其他明显的资源瓶颈

**建议：** 立即将 `sample_batch_size` 从 64 改为 8，重新训练。

