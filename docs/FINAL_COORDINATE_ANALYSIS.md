# LIBERO与RDT坐标系一致性 - 最终分析报告

**分析方法**：通过代码阅读和数据追踪（非观察推测）

---

## 🎯 核心结论

### 1. 训练代码中的坐标转换

**文件**：`data/hdf5_libero_dataset.py` 第203行

**实际代码**：
```python
pos_meters = pos_normalized * 0.012  # 只做缩放
```

**结论**：
- ✅ 实现了数值单位转换（归一化 → 米）
- ❌ **没有实现坐标系方向转换**
- 没有任何 `-pos_meters` 或轴翻转代码
- 符号保持不变（+1 → +1, -1 → -1）

### 2. 评估代码中的坐标转换

**文件**：`eval_sim/eval_rdt_libero.py` 第372-374行

**实际代码**：
```python
pos_x_norm = pos_x_meters / 0.012  # 只做缩放
pos_y_norm = pos_y_meters / 0.012
pos_z_norm = pos_z_meters / 0.012
```

**结论**：
- ✅ 实现了数值单位转换（米 → 归一化）
- ❌ **没有实现坐标系方向转换**
- 没有任何符号翻转代码
- 转换完全可逆

### 3. 数据流验证

**实验**：追踪实际训练样本

```
原始LIBERO action[103]: [0.000, 0.589, -0.244]
× 0.012 (代码转换):      [0.000, 0.007, -0.003]
实际训练样本:            [0.000, 0.007, -0.003]

符号检查:
  X: +0 → +0 ✅
  Y: +1 → +1 ✅
  Z: -1 → -1 ✅
```

**结论**：数据转换过程保持符号不变

---

## 📋 回答你的问题

### Q1: LIBERO和RDT的坐标系是否一致？

**A1**：
- **代码假设**：一致（代码没有做坐标转换）
- **实际情况**：需要通过评估验证
- **如果你的模型方向反了**：说明它们**不一致**

### Q2: 训练和评估时有没有实现坐标转换？

**A2**：
- **数值转换**：✅ 有（缩放因子 0.012）
- **坐标系方向转换**：❌ **没有**
  - 训练时：无轴翻转
  - 评估时：无轴翻转

---

## 🔍 代码证据

### 证据1：训练代码完整流程

```python
# data/hdf5_libero_dataset.py 第187-245行

def fill_in_action(values):
    # 步骤1: 数值缩放
    pos_normalized = values[:, 0:3]      # LIBERO归一化 [-1,1]
    pos_meters = pos_normalized * 0.012  # → 物理单位（米）
    
    # ❌ 没有步骤2: 坐标系转换（如果需要的话）
    # pos_meters[0] = -pos_meters[0]  # 这样的代码不存在
    
    # 步骤3: 填充到128维向量
    action_10d = np.concatenate([pos_meters, ori_6d, gripper])
    uni_vec[..., UNI_ACTION_INDICES] = action_10d
    return uni_vec
```

### 证据2：评估代码完整流程

```python
# eval_sim/eval_rdt_libero.py 第348-417行

def convert_rdt_action_to_libero(rdt_action):
    # 步骤1: 提取RDT输出
    pos_x_meters = action_128d[30]
    pos_y_meters = action_128d[31]
    pos_z_meters = action_128d[32]
    
    # 步骤2: 数值缩放
    pos_x_norm = pos_x_meters / 0.012
    pos_y_norm = pos_y_meters / 0.012
    pos_z_norm = pos_z_meters / 0.012
    
    # ❌ 没有步骤3: 坐标系转换（如果需要的话）
    # pos_x_norm = -pos_x_norm  # 这样的代码不存在
    
    # 步骤4: 构建LIBERO action
    libero_action = np.array([
        pos_x_norm, pos_y_norm, pos_z_norm,  # 无翻转
        ori_x_norm, ori_y_norm, ori_z_norm,
        gripper_norm
    ])
    return libero_action
```

### 证据3：数据流追踪结果

```python
# analyze_coordinate_consistency.py 运行结果

方法1: 追踪训练数据流
  时间步10: 符号匹配 [True, True, True] ✅
  时间步20: 符号匹配 [True, True, True] ✅  
  时间步30: 符号匹配 [True, True, True] ✅

方法2: 检查训练代码
  是否包含位置符号翻转: ❌ 无
  位置缩放因子: 0.012 ✅

方法3: 检查评估代码
  是否包含位置符号翻转: ❌ 无
  位置缩放因子: 0.012 ✅

方法4: 检查实际训练样本
  符号一致性: [✅, ✅, ✅]
```

---

## 💡 为什么会有方向反转问题？

### 代码的隐含假设

```python
# 当前代码隐含假设：
LIBERO坐标系 == RDT预训练坐标系

# 因此只做了单位转换：
训练: LIBERO(归一化) × 0.012 → RDT(米)
评估: RDT(米) ÷ 0.012 → LIBERO(归一化)
```

### 如果假设错误

```python
# 实际情况可能是：
LIBERO坐标系 != RDT预训练坐标系

# 例如（假设）：
LIBERO: X右, Y前, Z上
RDT:    X左, Y后, Z下

# 那么需要：
训练: LIBERO × 0.012 → 翻转某些轴 → RDT
评估: RDT → 翻转某些轴 → ÷ 0.012 → LIBERO
```

### 你的情况

**症状**："左右上下前后都相反"
**诊断**：RDT预训练坐标系与LIBERO完全相反
**需要**：翻转所有位置轴（或部分轴）

---

## 🔧 如何修复

### 方法：先评估测试，再训练修复

1. **在评估代码中测试翻转**（快速验证）
   ```python
   # eval_sim/eval_rdt_libero.py 第411行
   libero_action = np.array([
       -pos_x_norm, -pos_y_norm, -pos_z_norm,  # 测试翻转
       ori_x_norm, ori_y_norm, ori_z_norm,
       gripper_norm
   ])
   ```

2. **运行评估，观察结果**
   - 如果方向正确 → 找到了正确的转换
   - 如果不对 → 尝试其他翻转组合

3. **在训练代码中应用相同的翻转**
   ```python
   # data/hdf5_libero_dataset.py 第203行后
   pos_meters = pos_normalized * 0.012
   pos_meters = -pos_meters  # 添加坐标系转换
   ```

4. **重新训练和评估**

---

## 📊 完整的转换表

| 阶段 | 输入 | 处理 | 输出 | 坐标转换 |
|------|------|------|------|----------|
| **训练** | LIBERO归一化<br>[-1,1] | × 0.012 | RDT物理单位<br>(米) | ❌ 无 |
| **评估** | RDT物理单位<br>(米) | ÷ 0.012 | LIBERO归一化<br>[-1,1] | ❌ 无 |

**问题**：如果LIBERO坐标系 ≠ RDT坐标系，缺少坐标转换会导致方向错误。

**修复**：在"处理"步骤中添加坐标轴翻转。

---

## 📝 总结

### 通过代码分析确认的事实

1. ✅ **训练代码只做数值缩放**（× 0.012）
2. ✅ **评估代码只做数值缩放**（÷ 0.012）
3. ❌ **两者都没有坐标系方向转换**
4. ✅ **数据转换符号保持一致**
5. ❌ **代码假设LIBERO和RDT坐标系相同**

### 如果方向反转

**原因**：代码的假设错误，两个坐标系实际不同
**解决**：添加坐标系转换代码（轴翻转）

### 验证方法

不依赖观察，通过评估：
1. 测试不同的轴翻转配置
2. 录制视频观察结果
3. 找到正确的转换
4. 应用到训练中

---

**报告日期**：2025-10-17  
**分析方法**：代码阅读 + 数据追踪  
**可信度**：✅ 高（基于实际代码，非推测）

