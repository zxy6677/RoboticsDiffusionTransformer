# LIBERO评估坐标系问题分析报告

**问题描述**: 训练5000步后，评估时机械臂运动方向反了（上下反，左右反）

---

## 🔍 关键发现

### 1. ❌ 缩放因子错误（已确认）

通过分析LIBERO数据集，发现：

```python
# 实际测量的缩放因子
action到物理增量的缩放因子统计:
  均值:   0.011844
  中位数: 0.012313
  标准差: 0.004193
```

**当前代码使用的缩放因子**: `0.05`  
**实际应该使用的缩放因子**: `~0.012`  
**错误倍数**: 约4倍

#### 影响

- **训练数据**: action * 0.05 → 物理增量放大了4倍
- **评估转换**: 物理增量 / 0.05 → action缩小了4倍
- **结果**: 训练数据的幅度不正确，但这**不能解释方向反了**的问题

---

### 2. ⚠️ 可能的坐标系不匹配

用户报告"上下反，左右反"，这暗示某些轴的**符号**可能反了。

可能原因：
1. **RDT预训练使用的坐标系** 与 **LIBERO的坐标系** 定义不同
2. **某些轴需要取反**以匹配两个坐标系

#### LIBERO的坐标系（已验证）

从"关闭抽屉"任务分析：
```
任务：close_the_top_drawer
末端位置变化：
  X: +0.1773 (向右)
  Y: -0.1017 (向后，靠近机器人) ← 符合拉抽屉
  Z: -0.1860 (向下)
```

**LIBERO坐标系约定**：
- X+: 向右
- Y+: 向前（远离机器人）
- Y-: 向后（靠近机器人）
- Z+: 向上
- Z-: 向下

---

### 3. ✅ 符号转换验证

我们的代码转换（训练和评估）在**数学上完全可逆**，符号保持：
```
原始: X=0.3589, Y=0.5732, Z=-0.1045
  ↓ × 0.05
训练: X=0.017946, Y=0.028661, Z=-0.005223
  ↓ ÷ 0.05
还原: X=0.3589, Y=0.5732, Z=-0.1045 ✅
```

**结论**: 我们的转换代码本身没有引入符号错误。

---

## 🤔 可能的根本原因

### 假设1: RDT预训练坐标系与LIBERO不同

RDT在多个机器人数据集上预训练，可能使用了不同的坐标系约定。

**可能的情况**：
- RDT使用：X向左，Y向后，Z向上
- LIBERO使用：X向右，Y向前，Z向上

如果是这样，需要在转换时翻转某些轴：
```python
# 假设需要翻转X和Y
pos_meters = pos_normalized * 0.012
pos_meters[0] = -pos_meters[0]  # 翻转X
pos_meters[1] = -pos_meters[1]  # 翻转Y
```

### 假设2: 缩放因子错误导致模型学习错误

虽然缩放因子错误不会直接导致方向反转，但可能：
1. 模型在错误的数据分布上训练
2. 学到了错误的映射关系
3. 间接导致输出方向错误

---

## 🔧 修复方案

### 方案A: 修复缩放因子（优先尝试）

```python
# data/hdf5_libero_dataset.py
# 修改第203行和209行

# === 步骤1: 使用正确的缩放因子 ===
pos_normalized = values[:, 0:3]
pos_meters = pos_normalized * 0.012  # 改为0.012而不是0.05

ori_normalized = values[:, 3:6]
ori_radians = ori_normalized * 0.06  # 相应调整（0.5 * 0.012/0.05 ≈ 0.12）
# 或者保持0.5不变，只改位置
```

```python
# eval_sim/eval_rdt_libero.py
# 修改第372-374行和393-395行

# === 评估转换 ===
pos_x_norm = pos_x_meters / 0.012  # 改为0.012
pos_y_norm = pos_y_meters / 0.012
pos_z_norm = pos_z_meters / 0.012

ori_x_norm = ori_euler_rad[0] / 0.12  # 相应调整
# 或者保持0.5不变
```

### 方案B: 添加坐标轴翻转（如果方案A不work）

如果修复缩放因子后方向还是反的，尝试翻转某些轴：

```python
# eval_sim/eval_rdt_libero.py
# 在convert_rdt_action_to_libero函数中

# 步骤1: 提取并转换
pos_x_norm = pos_x_meters / 0.012
pos_y_norm = pos_y_meters / 0.012
pos_z_norm = pos_z_meters / 0.012

# 步骤2: 根据用户反馈翻转轴
# 如果"左右反"，翻转X
pos_x_norm = -pos_x_norm

# 如果"上下反"，翻转Z  
pos_z_norm = -pos_z_norm

# 如果"前后反"，翻转Y
# pos_y_norm = -pos_y_norm
```

### 方案C: 重新训练（如果需要）

如果缩放因子错误导致模型学习了错误的映射：
1. 修复缩放因子
2. 重新计算数据集统计
3. 从RDT-1B重新开始训练

---

## 📊 验证步骤

### 1. 先测试方案A（修复缩放因子）

```bash
# 修改代码后，使用当前checkpoint评估
python eval_sim/eval_rdt_libero.py \
    --pretrained checkpoints/xxx \
    --num_tasks 1 \
    --max_steps 50 \
    --record_video
```

观察机械臂运动：
- 如果方向正确了 → 问题解决✅
- 如果方向还是反的 → 尝试方案B

### 2. 测试方案B（添加轴翻转）

根据观察到的具体方向错误，逐一尝试翻转X/Y/Z：

```python
# 测试1: 只翻转X
pos_x_norm = -pos_x_norm

# 测试2: 只翻转Z  
pos_z_norm = -pos_z_norm

# 测试3: 同时翻转X和Z
pos_x_norm = -pos_x_norm
pos_z_norm = -pos_z_norm
```

每次修改后评估，直到找到正确的组合。

---

## 🎯 推荐行动

### 立即执行

1. **修复缩放因子**（方案A）
   - 数据加载器：0.05 → 0.012
   - 评估代码：0.05 → 0.012
   - 重新计算数据集统计

2. **使用当前checkpoint测试**
   - 如果方向正确 → 重新训练以获得更好效果
   - 如果方向还反 → 进入方案B

3. **记录测试结果**
   - 视频录制
   - 记录哪些轴反了

### 后续优化

如果方向修复后：
1. 从头重新训练（使用正确的缩放因子）
2. 预期成功率会显著提升

---

## 📝 关键代码位置

| 文件 | 行号 | 需要修改 |
|------|------|----------|
| `data/hdf5_libero_dataset.py` | 203 | `pos_meters = pos_normalized * 0.012` |
| `data/hdf5_libero_dataset.py` | 209 | `ori_radians = ori_normalized * 0.5` (保持或调整) |
| `eval_sim/eval_rdt_libero.py` | 372-374 | `pos_*_norm = pos_*_meters / 0.012` |
| `eval_sim/eval_rdt_libero.py` | 393-395 | `ori_*_norm = ori_euler_rad[*] / 0.5` (保持或调整) |

---

**状态**: 🔴 需要立即修复  
**优先级**: P0  
**预计解决时间**: 1-2小时（测试+重新训练）

