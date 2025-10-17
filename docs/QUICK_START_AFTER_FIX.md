# 修复后快速开始指南

## ✅ 修复已完成

所有训练代码已修复并验证！现在可以正确地使用LIBERO数据集微调RDT模型。

---

## 🚀 立即开始训练

### 1. 激活环境

```bash
conda activate rdt
cd /home/ubuntu/RoboticsDiffusionTransformer
```

### 2. 开始训练（推荐配置）

```bash
# 基础训练命令
python main.py \
    --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b \
    --output_dir=checkpoints/libero_finetune_fixed \
    --dataset_type=finetune \
    --load_from_hdf5 \
    --run_name=libero_finetune_fixed \
    --num_train_epochs=10 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --warmup_steps=500 \
    --save_steps=1000 \
    --logging_steps=50 \
    --dataloader_num_workers=4 \
    --push_to_hub=False
```

### 3. 使用DeepSpeed训练（更快）

```bash
# 如果有多GPU或需要更好的内存优化
bash train_remote_deepspeed_single.sh
```

**修改 `train_remote_deepspeed_single.sh` 中的输出目录**:
```bash
OUTPUT_DIR="checkpoints/libero_finetune_fixed"
```

---

## 📊 监控训练

### WandB监控

训练会自动记录到WandB，关注这些指标：

- **`train/loss`**: 应该稳定下降
- **`train/sample_mse`**: 应该比修复前更低
- **学习率曲线**: 确认warmup正常

### 本地监控

```bash
# 查看训练日志
tail -f checkpoints/libero_finetune_fixed/training.log

# 查看checkpoint
ls -lh checkpoints/libero_finetune_fixed/
```

---

## 🧪 评估训练后的模型

### 基础评估

```bash
python eval_sim/eval_rdt_libero.py \
    --config configs/base.yaml \
    --pretrained checkpoints/libero_finetune_fixed/checkpoint-5000 \
    --text_encoder google/t5-v1_1-xxl \
    --vision_encoder google/siglip-so400m-patch14-384 \
    --benchmark libero_90 \
    --num_tasks 5 \
    --max_steps 100
```

### 带视频录制的评估

```bash
python eval_sim/eval_rdt_libero.py \
    --config configs/base.yaml \
    --pretrained checkpoints/libero_finetune_fixed/checkpoint-5000 \
    --text_encoder google/t5-v1_1-xxl \
    --vision_encoder google/siglip-so400m-patch14-384 \
    --benchmark libero_90 \
    --num_tasks 10 \
    --max_steps 100 \
    --record_video \
    --video_output_dir videos/fixed_model_eval
```

---

## 📈 预期结果

### 修复前 vs 修复后

| 指标 | 修复前 | 修复后（预期） |
|-----|--------|--------------|
| 成功率 | 0% ❌ | 20-50% ✅ |
| Loss收敛 | 不稳定 | 稳定下降 |
| 动作合理性 | 异常 | 正常 |
| 迁移学习 | 差 | 好 |

### 评估任务示例

```
Task 0: KITCHEN_SCENE10_close_the_top_drawer
Task 1: KITCHEN_SCENE10_close_drawer_and_put_bowl
Task 2: LIVING_ROOM_SCENE1_pick_up_book
...
```

每个任务评估10次，计算成功率。

---

## 🔍 验证修复

如果想再次确认修复正确：

```bash
# 验证数据格式
python verify_fixed_data.py

# 查看数据集统计
python -c "
import json
with open('configs/dataset_stat.json', 'r') as f:
    stats = json.load(f)
print('Action mean (pos_x):', stats['libero_90']['action_mean'][30])
print('Action std (pos_x):', stats['libero_90']['action_std'][30])
"
```

预期输出：
```
Action mean (pos_x): ~0.000 (接近0的小数)
Action std (pos_x): ~0.014 (米的数量级)
```

---

## 📝 修复的关键变化

### 训练数据

**位置**：
- 修复前: 归一化值 [-1, 1]
- 修复后: 物理单位 [-0.05, 0.05] 米 ✅

**旋转**：
- 修复前: 从归一化欧拉角转换
- 修复后: 从物理弧度转换 ✅

**Gripper**：
- 修复前后: [0, 1] 归一化 ✅

### 评估代码

修复后的评估代码会：
1. 接收RDT的物理单位输出
2. 转换为LIBERO的归一化格式
3. 发送给模拟器执行

---

## ⚠️ 注意事项

### 1. 必须从头训练

❌ **不要使用旧checkpoint**

旧的checkpoint用错误数据训练，必须：
- 从RDT-1B预训练权重开始
- 或者至少从非常早期的checkpoint继续

### 2. 数据集路径

确认LIBERO数据集软链接正确：

```bash
ls -la data/datasets/libero_90
# 应该指向: /home/ubuntu/LIBERO/libero/datasets/libero_90
```

### 3. 内存需求

- 单GPU训练: 至少24GB显存（建议A100）
- 使用DeepSpeed ZeRO-2可以降低显存需求
- batch_size=8 + grad_accum=2 ≈ 有效batch_size=16

---

## 🛠️ 故障排查

### 问题1: CUDA Out of Memory

**解决方案**:
```bash
# 减小batch size
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=4

# 或使用DeepSpeed
bash train_remote_deepspeed_single.sh
```

### 问题2: 数据加载慢

**解决方案**:
```bash
# 增加dataloader workers
--dataloader_num_workers=8
```

### 问题3: 找不到LIBERO

**解决方案**:
```bash
# 检查软链接
ls -la data/datasets/libero_90

# 重新创建软链接
rm data/datasets/libero_90
ln -s /home/ubuntu/LIBERO/libero/datasets/libero_90 data/datasets/libero_90
```

### 问题4: 评估失败

**解决方案**:
```bash
# 确保使用修复后训练的checkpoint
# 不要评估旧的checkpoint！

# 检查LIBERO路径
python -c "
import libero
libero.set_libero_default_path('/home/ubuntu/LIBERO/libero/libero')
print('LIBERO path OK')
"
```

---

## 📚 相关文档

1. **`LIBERO_TRAINING_FIX_COMPLETE.md`** - 完整修复报告
2. **`TRAINING_CODE_FINAL_ANALYSIS.md`** - 详细问题分析
3. **`verify_fixed_data.py`** - 数据验证脚本
4. **`compute_dataset_statistics.py`** - 统计计算脚本

---

## 🎯 检查清单

开始训练前，确认：

- [ ] ✅ 激活了 `rdt` conda环境
- [ ] ✅ LIBERO数据集软链接正确
- [ ] ✅ 使用修复后的代码（可运行 `verify_fixed_data.py` 确认）
- [ ] ✅ 从RDT-1B预训练权重开始（不用旧checkpoint）
- [ ] ✅ 有足够的GPU显存（推荐24GB+）
- [ ] ✅ WandB已配置（可选）

全部确认后，执行训练命令！

---

## 💡 小贴士

### 训练建议

- **Epoch数**: 10-20 epochs足够
- **Checkpoint选择**: 通常中间的checkpoint（如5000-10000步）效果最好
- **验证频率**: 每1000步评估一次
- **早停**: 如果验证loss不再下降，可以提前停止

### 评估建议

- **任务数**: 先评估5个任务快速测试，成功后评估全部90个
- **最大步数**: 100步通常足够
- **视频录制**: 建议至少录制一次以可视化验证
- **多次运行**: 每个任务运行10次取平均成功率

---

## 🚀 开始吧！

一切准备就绪，现在可以开始训练了：

```bash
# 复制这条命令直接运行
conda activate rdt && \
cd /home/ubuntu/RoboticsDiffusionTransformer && \
python main.py \
    --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b \
    --output_dir=checkpoints/libero_finetune_fixed \
    --dataset_type=finetune \
    --load_from_hdf5 \
    --run_name=libero_finetune_fixed \
    --num_train_epochs=10 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-5 \
    --warmup_steps=500 \
    --save_steps=1000 \
    --logging_steps=50
```

祝训练顺利！🎉

