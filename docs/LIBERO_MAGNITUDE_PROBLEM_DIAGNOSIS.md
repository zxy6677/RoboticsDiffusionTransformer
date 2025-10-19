# LIBERO微调失败的根本原因诊断

## 🔍 问题现状

单任务微调10000步后，模型仍然**完全无法完成任务**：
- **Training loss**: 非常低 (~1e-5)
- **Evaluation**: 模型输出Position值过大，被clip到边界`[1., 1., 1.]`，导致机械臂失控

## 🎯 根本原因

### Position vs Orientation的量级差异

**LIBERO数据特点**：
```
Position (米):         [-0.012, 0.012]      量级: 10^-2
Orientation (6D):      [-1.0, 1.0]          量级: 10^0
量级差异：100倍！
```

**这导致的问题**：
1. **训练时**：模型很容易学会让Position的MSE降到很低，但Orientation的loss仍然很高
2. **Loss weighting失效**：`state_norm`(即action_std)也有100倍差异，无法有效平衡

### 与Maniskill的对比

| 特性 | Maniskill (成功) | LIBERO (失败) |
|------|-----------------|--------------|
| **控制空间** | Joint angles (关节角度) | EEF pose (末端执行器位置+旋转) |
| **数据维度** | 7个关节角 + 1 gripper | 3D position + 6D rotation + gripper |
| **量级特点** | ✅ 所有维度都是弧度，量级相同 | ❌ Position(米) vs Rotation(单位向量)，量级差100倍 |
| **是否归一化** | 是，min-max到[-1,1] | 否，直接使用物理单位 |

**关键发现**：
- README声称"不归一化物理量"（IMPORTANT 3）
- 但Maniskill代码中**明确使用了min-max归一化**：
  ```python
  # scripts/maniskill_model.py:172
  joints = (joints - self.state_min) / (self.state_max - self.state_min) * 2 - 1
  ```

### 为什么Maniskill可以归一化而LIBERO不行？

**Maniskill的特殊性**：
- 7个关节角都是**同一物理量**（弧度）
- 归一化**不破坏维度间的相对关系**
- 本质上是**数值缩放**，不改变物理意义

**LIBERO的问题**：
- Position（米）和Orientation（6D单位向量）是**不同的物理量**
- 直接归一化会混淆两者的物理意义
- README的"不归一化"原则主要针对**不同物理量的混合**

## 📊 实验证据

### 训练数据分析
```
RDT训练actions的范围（物理单位）：
  Position (30-32):     [-0.01, 0.01] 米
  Orientation (33-38):  [-1.0, 1.0]    (6D表示)
  
state_norm (action_std):
  Position:    ~0.003
  Orientation: ~0.04
```

### 模型输出分析
```
RDT输出（checkpoint-10000）：
  Position: [0.243, 0.287, 0.116] 米  ← 超出训练范围20倍！
  被clip到: [1., 1., 1.]
```

**结论**：模型**过度学习了Position维度**，而忽略了Orientation的精确性。

## 💡 可能的解决方案

### 方案1：Position缩放（推荐）
```python
# 将Position从米缩放到厘米，使其与Orientation量级相同
pos_cm = pos_meters * 100  # 0.01米 = 1厘米
action_128d[30:33] = pos_cm
action_128d[33:39] = ori_6d
```

**优点**：
- 简单直接
- 符合README的"选择合适单位"原则
- Position和Orientation量级接近

**缺点**：
- 需要在评估时反向缩放

### 方案2：整体归一化（参考Maniskill）
```python
# 对整个action vector进行min-max归一化
action_128d = (action_128d - action_min) / (action_max - action_min) * 2 - 1
```

**优点**：
- 与Maniskill一致
- 所有维度在[-1, 1]范围

**缺点**：
- 违反README IMPORTANT 3的字面意思
- 需要重新计算全局统计

### 方案3：改进loss weighting
```python
# 使用不同的权重策略
weight_position = 100  # 提高position的权重
weight_orientation = 1
state_norm_adjusted = state_norm.copy()
state_norm_adjusted[30:33] *= weight_position
```

**优点**：
- 不改变数据格式
- 灵活可调

**缺点**：
- 需要手动调参
- 可能需要多次实验

## 🚀 下一步行动

1. **立即尝试**：方案1（Position缩放到厘米）
2. **验证**：单任务overfitting测试
3. **如果方案1失败**：尝试方案2（整体归一化）
4. **记录**：不同方案的效果对比

## 📝 核心教训

**RDT的"不归一化"原则**应该理解为：
- ✅ 不对**同一物理量的不同机器人**进行统一归一化（保持泛化性）
- ✅ 但对**单个机器人内部的不同物理量**，应该**选择合适的单位或缩放**，使其量级接近

**关键**：`README IMPORTANT 3`中的"Generally, we use the International System of Units, which ensures that most values fall within [-1,1]"`这句话暗示了：**应该选择让所有值都在[-1, 1]范围内的单位**！





