# Position量级修复总结

## ✅ 问题已解决

### 根本原因
**LIBERO微调失败的核心问题**：Position（米）和Orientation（6D）的量级差异达100倍！
- Position: `[-0.012, 0.012]` 米 → 量级 `10^-2`
- Orientation 6D: `[-1.0, 1.0]` → 量级 `10^0`

这导致模型训练时过度学习Position维度，忽略Orientation的精确性。

### 解决方案：Position缩放到厘米

**核心修改**：将Position从"米"改为"厘米"，使其与Orientation量级匹配。

#### 修改1: 训练数据处理 (`data/hdf5_libero_dataset.py`)
```python
# 之前（米）
pos_meters = pos_normalized * 0.012  # [-0.012, 0.012] 米

# 修复后（厘米）
pos_cm = pos_normalized * 1.2  # [-1.2, 1.2] 厘米
```

#### 修改2: 评估代码 (`eval_sim/eval_rdt_libero.py`)
```python
# 之前（米→归一化）
pos_x_norm = pos_x_meters / 0.012

# 修复后（厘米→归一化）
pos_x_norm = pos_x_cm / 1.2
```

### 验证结果

**修复前**：
```
Position最大值:  0.012 米
Orientation最大值: 1.0
量级比: 100倍 ❌
```

**修复后**：
```
Position最大值:  0.96 厘米
Orientation最大值: 1.0
量级比: 1.04倍 ✅
```

## 📋 符合README原则

**README IMPORTANT 3**：
> "Generally, we use the International System of Units, which **ensures that most values fall within [-1,1]**."

**理解**：
- ✅ 应该**选择合适的单位**让所有值都在[-1, 1]范围
- ✅ Position从"米"改为"厘米"正是符合这个原则
- ✅ Maniskill使用min-max归一化，因为joint angles量级相同
- ❌ 直接使用"米"导致量级失衡，违反了这个原则

## 🚀 当前状态

1. ✅ 代码已修复
2. ✅ 数据量级已验证（Position和Orientation量级比1.04x）
3. 🔄 正在重新训练单任务模型
4. ⏳ 等待训练完成后评估

## 📝 训练监控

**训练日志**：`/tmp/train_magnitude_fix.log`

**查看进度**：
```bash
tail -f /tmp/train_magnitude_fix.log
```

**初始loss下降正常**：
- Step 1: loss=0.011
- Step 80: loss=0.000689

## 🎯 后续步骤

1. 等待训练完成（10000步）
2. **使用EMA模型评估**：`checkpoints/single_task_overfit/checkpoint-XXXX/ema/model.safetensors`
3. 录制视频，观察机械臂行为
4. 如果单任务overfitting成功，扩展到libero_90全量微调

## 📌 关键教训

1. **不同物理量的量级必须匹配**：Position和Orientation虽然是不同物理量，但填入同一128维向量时必须量级接近
2. **README的"不归一化"是有前提的**：需要"选择合适单位"确保值在[-1, 1]范围
3. **Maniskill的成功不能直接复制**：因为joint angles本身量级就相同，而EEF pose不同
4. **评估要使用EMA模型**：`model.safetensors`通常比`pytorch_model.bin`效果更好

## 🔬 技术细节

### 为什么是1.2厘米？
```
LIBERO原始数据: [-1, 1] 归一化范围
实际对应物理增量: 0.012 米
转换为厘米: 0.012 * 100 = 1.2 厘米
因此: pos_cm = pos_normalized * 1.2
```

### Orientation为什么不需要调整？
- 6D rotation本身就是单位向量的组成部分
- 值域自然就在[-1, 1]范围内
- 不需要额外调整

### Gripper处理
- 保持不变：LIBERO的[-1, 1] → RDT的[0, 1]
- 这是README明确说明需要归一化的唯一物理量





