# Dataset Statistics 问题诊断报告

## 🎯 核心问题

**发现：使用了错误的dataset statistics，导致训练时loss weighting不正确**

## 📊 问题分析

### 实际情况
- **训练数据**：只有1个任务（`KITCHEN_SCENE10_close_the_top_drawer_and_put_bowl`）
- **使用的统计**：`libero_90`（90个任务的全局统计）

### 关键差异对比

| 维度 | libero_90（错误） | libero_single_task（正确） | 差异倍数 |
|------|-------------------|---------------------------|----------|
| **Position std (cm)** |  |  |  |
| - x维度 | 0.003 | 0.328 | **109x** |
| - y维度 | 0.004 | 0.467 | **117x** |
| - z维度 | 0.005 | 0.486 | **97x** |
| **Gripper std** | 0.499 | 0.479 | 1.04x |
| **Gripper mean** | 0.472 | 0.358 | 差异大 |

## ❌ 为什么这是严重问题？

### 1. Loss Weighting错误

在训练代码中（`data/hdf5_libero_dataset.py:263`）：
```python
state_norm = self.action_std_global + 1e-8  # (128,)
```

这个`state_norm`用于loss的加权：
```python
# 在训练时，loss会被state_norm归一化
loss = (prediction - target) / state_norm
```

**影响**：
- Position的std被低估100倍（0.003 vs 0.3）
- 导致Position的loss权重被**放大100倍**
- 模型会**过度关注position**，把position误差权重看得过高
- 同时**忽略其他维度**的学习

### 2. 为什么任务1表现更好？

评估结果显示的奇怪现象：
- **任务1**（非训练任务）：轨迹更接近合理
- **任务2**（训练任务）：轨迹不对

**原因**：
1. 模型在错误的loss weighting下训练
2. 过度关注position，忽略了其他维度
3. 导致整体行为不协调
4. 任务1"碰巧"在某些维度上与错误的学习方向一致
5. 任务2（真正的训练任务）反而表现差，因为它需要正确的多维度协调

## 🔧 解决方案

### 步骤1：更新数据集配置 ✅

已完成：
1. 计算单任务的正确统计 → `configs/dataset_stat_single_task.json`
2. 添加到主配置文件 → `configs/dataset_stat.json`

### 步骤2：更新训练代码

需要修改`data/hdf5_libero_dataset.py`以使用正确的数据集名称：

```python
# 第23行，修改前：
def __init__(self, dataset_name: str = "libero_90") -> None:

# 修改后（对于单任务训练）：
def __init__(self, dataset_name: str = "libero_single_task") -> None:
```

或者通过环境变量控制：
```python
default_dataset = "libero_90"
if os.path.exists("dataset_remote/") and not os.path.exists("data/datasets/libero_90"):
    default_dataset = "libero_single_task"

def __init__(self, dataset_name: str = default_dataset) -> None:
```

### 步骤3：重新训练模型

使用正确的统计信息重新训练：
```bash
# 设置环境变量指定数据集
export LIBERO_DATASET_NAME="libero_single_task"

# 重新训练
bash train_single_task.sh
```

## 📈 预期改进

修复后应该看到：

### 训练时
- Loss各维度权重正确平衡
- Position loss不会被过度放大
- 模型能正确学习所有维度的协调

### 评估时
- 训练任务（任务2）的表现应该显著改善
- 位置、旋转、抓取器的动作应该协调一致
- 整体轨迹更加合理

## 🔍 如何验证修复有效？

### 1. 检查训练loss
```bash
# 观察训练log，各维度loss应该平衡
# Position loss不应该异常大或小
```

### 2. 对比action统计
在训练开始时，打印使用的action_std：
```python
print(f"Using action_std: {self.action_std_global[[30,31,32,10]]}")
# 应该输出: [0.328, 0.467, 0.486, 0.479]
# 而不是: [0.003, 0.004, 0.005, 0.499]
```

### 3. 评估性能
重新评估训练任务，应该看到：
- 任务2（训练任务）成功率提升
- 轨迹更加协调

## 📝 技术细节

### Dataset Statistics的作用

在RDT训练中：
1. **Action统计用于loss weighting**（非常重要！）
   ```python
   state_norm = action_std + 1e-8
   weighted_loss = loss / state_norm
   ```

2. **不用于action归一化**（按照README IMPORTANT 3）
   - 物理量不归一化
   - 保持物理单位和意义

### 为什么libero_90统计与单任务差异大？

1. **libero_90包含90个不同任务**
   - 不同任务的action分布差异很大
   - 全局统计是所有任务的平均
   
2. **单任务action分布更集中**
   - Position变化范围更大（0.3-0.5 cm std）
   - 因为是特定任务的重复执行

3. **Action vs Action Change**
   - libero_90的统计可能是action change（逐步变化）
   - 单任务计算的是action本身的分布

## ✅ 总结

### 问题根源
使用了`libero_90`（90任务）的统计信息训练单任务模型，导致：
- Position loss权重被错误放大100倍
- 模型过度关注position，忽略其他维度
- 训练任务反而表现差

### 解决方案
1. ✅ 计算单任务的正确统计
2. ⏳ 更新训练代码使用正确统计
3. ⏳ 重新训练模型

### 预期改进
- 训练任务成功率应该显著提升
- 所有维度（位置、旋转、抓取）协调一致
- 修复"训练任务反而差"的奇怪现象

---

**重要性：这可能是导致训练失败的主要原因之一！**

