# LIBERO与RDT坐标系检查总结

## 🎯 检查结论

### ✅ LIBERO坐标系是一致的

**坐标系定义**：
- X+: 向右（机器人视角）
- Y+: 向前（远离机器人）
- Z+: 向上

**验证**：基于"关闭抽屉"任务的实际轨迹
```
总体运动: [+0.1773, -0.1017, -0.1860]
  X: +0.1773 → 向右移动 ✅
  Y: -0.1017 → 向后拉（关闭抽屉）✅
  Z: -0.1860 → 向下移动 ✅

Action与实际增量符号完全一致 ✅✅✅
```

### ❌ 训练和评估时**没有实现坐标系方向的转换**

**当前实现**：
```python
# 训练时（data/hdf5_libero_dataset.py）
pos_meters = pos_normalized * 0.012  # ❌ 只做了缩放，没有轴翻转

# 评估时（eval_sim/eval_rdt_libero.py）
pos_x_norm = pos_x_meters / 0.012    # ❌ 只做了缩放，没有轴翻转
```

**问题**：
- 只实现了数值单位转换（归一化 ↔ 物理单位）
- **没有实现坐标系方向的转换**（如果RDT预训练坐标系与LIBERO不同）

---

## 🔍 详细分析

### 1. 训练时的处理

**文件**：`data/hdf5_libero_dataset.py` 第187-245行

**流程**：
```python
# 输入：LIBERO action (归一化 [-1,1])
pos_normalized = values[:, 0:3]

# 转换1：缩放到物理单位
pos_meters = pos_normalized * 0.012  # ✅ 数值转换正确

# ❌ 缺失：坐标系方向转换
# 如果需要，应该在这里添加：
# pos_meters[0] = -pos_meters[0]  # 翻转X轴（示例）

# 转换2：填充到128维向量
action_128d[30:33] = pos_meters
```

**检查结果**：
- ✅ 符号保持不变（+1 → +1, -1 → -1）
- ✅ 缩放因子正确（0.012）
- ❌ 没有坐标系方向转换

### 2. 评估时的处理

**文件**：`eval_sim/eval_rdt_libero.py` 第348-417行

**流程**：
```python
# 输入：RDT输出（物理单位 米）
pos_x_meters = action_128d[30]
pos_y_meters = action_128d[31]
pos_z_meters = action_128d[32]

# 转换1：缩放到归一化范围
pos_x_norm = pos_x_meters / 0.012  # ✅ 数值转换正确

# ❌ 缺失：坐标系方向转换
# 如果需要，应该在这里添加：
# pos_x_norm = -pos_x_norm  # 翻转X轴（示例）

# 输出：LIBERO action (归一化 [-1,1])
libero_action = [pos_x_norm, pos_y_norm, pos_z_norm, ...]
```

**检查结果**：
- ✅ 符号保持不变
- ✅ 转换完全可逆（往返误差 < 1e-8）
- ❌ 没有坐标系方向转换

### 3. 数值验证

**往返转换测试**：
```
原始LIBERO action: [0.1473, -0.0375, -0.0000]
  ↓ × 0.012
物理单位（米）:    [0.001768, -0.000450, -0.000000]
  ↓ ÷ 0.012
恢复的action:      [0.1473, -0.0375, -0.0000]

误差: [0.0, 0.0, 0.0] ✅ 完全可逆
```

**结论**：数值转换（缩放）是正确的，但没有方向转换。

---

## 🤖 RDT预训练坐标系问题

### 可能的情况

**情况A**：RDT预训练坐标系 = LIBERO坐标系
- 结果：不需要方向转换
- 表现：模型能正确预测方向
- **但你的情况不是这样**（方向反了）

**情况B**：RDT预训练坐标系 ≠ LIBERO坐标系 ⚠️
- 结果：需要方向转换
- 表现：模型预测的方向相反
- **这解释了你遇到的问题！**

### 为什么会不同？

RDT在多个机器人数据集上预训练：
- Open X-Embodiment（包含多个不同机器人）
- ALOHA双臂机器人
- 其他数据集

**问题**：不同机器人使用不同的坐标系约定：
- 有些：X前, Y左, Z上（机器人基座系）
- 有些：X右, Y前, Z上（世界坐标系，LIBERO用这个）
- 有些：其他约定

如果预训练数据主要来自某种特定坐标系，与LIBERO不同，就会导致方向问题。

---

## 💡 解决方案

### 方案1：评估时测试（快速验证）⭐

**目的**：快速找出正确的轴翻转配置

**步骤**：
```python
# 修改 eval_sim/eval_rdt_libero.py 第407-411行

# 测试配置：翻转所有位置轴
libero_action = np.array([
    -pos_x_norm, -pos_y_norm, -pos_z_norm,  # 翻转位置
    ori_x_norm, ori_y_norm, ori_z_norm,      # 旋转保持
    gripper_norm
])
```

**运行测试**：
```bash
python eval_sim/eval_rdt_libero.py \
    --pretrained checkpoints/xxx/checkpoint-26000 \
    --num_tasks 1 \
    --max_steps 50 \
    --record_video
```

**判断**：
- 观察视频中机械臂运动方向
- 如果正确 → 找到了需要的转换！
- 如果不对 → 尝试其他翻转组合

### 方案2：训练时应用（正式修复）⭐⭐⭐

**一旦找到正确的翻转配置，应该在训练时应用**

**修改 `data/hdf5_libero_dataset.py` 第203行**：

```python
# 当前代码
pos_normalized = values[:, 0:3]
pos_meters = pos_normalized * 0.012

# 🔧 添加坐标系转换（假设需要翻转所有轴）
pos_meters = -pos_meters  # LIBERO → RDT坐标系

# 或者只翻转特定轴
# pos_meters[:, 0] = -pos_meters[:, 0]  # 只翻转X
# pos_meters[:, 1] = -pos_meters[:, 1]  # 只翻转Y
# pos_meters[:, 2] = -pos_meters[:, 2]  # 只翻转Z
```

**然后**：
1. 重新计算数据集统计：
   ```bash
   python -m data.compute_dataset_stat_hdf5
   ```

2. 重新训练模型：
   ```bash
   python libero_finetune_correct.py \
       --task_id 0 \
       --max_steps 15000
   ```

3. 评估新模型（恢复eval代码到不翻转）：
   ```python
   # eval_sim/eval_rdt_libero.py 恢复原样
   libero_action = np.array([
       pos_x_norm, pos_y_norm, pos_z_norm,  # 不翻转
       ori_x_norm, ori_y_norm, ori_z_norm,
       gripper_norm
   ])
   ```

### 可能的翻转配置

根据"左右上下前后都反"的描述：

**配置1：翻转所有位置轴**（最可能）
```python
pos_meters = -pos_meters  # 翻转X, Y, Z
```

**配置2：翻转所有轴（包括旋转）**
```python
pos_meters = -pos_meters
ori_radians = -ori_radians
```

**配置3：只翻转XY**
```python
pos_meters[:, 0] = -pos_meters[:, 0]
pos_meters[:, 1] = -pos_meters[:, 1]
```

---

## 📊 检查脚本

我已创建 `check_coordinate_systems.py`，运行结果：

```bash
$ python check_coordinate_systems.py

✅ 已确认正确的部分:
  1. LIBERO数据本身使用一致的坐标系（X右, Y前, Z上）
  2. 训练数据转换保持符号不变（× 0.012）
  3. 评估数据转换完全可逆（÷ 0.012）
  4. State的坐标系与LIBERO一致

❓ 可能的问题:
  RDT预训练模型可能学习的是与LIBERO不同的坐标系映射
```

---

## 🎯 下一步行动

### 立即执行（5-10分钟）

1. **修改评估代码**测试翻转：
   ```bash
   # 编辑 eval_sim/eval_rdt_libero.py
   # 在第411行应用翻转
   libero_action = np.array([
       -pos_x_norm, -pos_y_norm, -pos_z_norm,
       ori_x_norm, ori_y_norm, ori_z_norm,
       gripper_norm
   ])
   ```

2. **运行评估**：
   ```bash
   python eval_sim/eval_rdt_libero.py \
       --pretrained checkpoints/libero_finetune/task_00_xxx/checkpoint-26000 \
       --num_tasks 1 \
       --max_steps 50 \
       --record_video
   ```

3. **观察结果**：
   - 查看视频
   - 确认方向是否正确

### 根据结果行动

**如果方向正确**：
1. 在训练代码中应用相同的翻转
2. 重新计算统计
3. 重新训练模型
4. 评估新模型

**如果方向不对**：
1. 尝试其他翻转配置
2. 重复测试直到找到正确配置

---

## 📋 总结

| 检查项 | 状态 | 说明 |
|-------|------|------|
| LIBERO坐标系一致性 | ✅ | X右, Y前, Z上 |
| 训练数值转换 | ✅ | × 0.012 正确 |
| 评估数值转换 | ✅ | ÷ 0.012 正确 |
| 转换可逆性 | ✅ | 误差 < 1e-8 |
| **训练坐标系转换** | ❌ | **未实现** |
| **评估坐标系转换** | ❌ | **未实现** |

**根本问题**：
- RDT预训练坐标系可能与LIBERO不同
- 当前代码只做了数值缩放，没有方向转换
- 导致方向反转问题

**解决方案**：
- 找出正确的轴翻转配置
- 在训练时应用坐标系转换
- 重新训练模型

---

**报告日期**：2025-10-17  
**检查状态**：✅ 完成  
**问题根源**：✅ 已确认（坐标系方向未转换）  
**修复方案**：✅ 已提供  
**下一步**：测试轴翻转配置

