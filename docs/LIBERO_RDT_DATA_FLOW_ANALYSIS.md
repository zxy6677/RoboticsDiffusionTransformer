# LIBERO-RDT数据流分析

## 问题现状
单任务微调后，机械臂完全没学会demo动作轨迹，开始乱转动。

## 数据流分析

### 1. LIBERO原始数据格式
```
Demo文件中的actions:
- Shape: (71, 7)
- 维度: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
- 范围: [-1, 1] (归一化后的值)
- Gripper: -1 (closed), +1 (open)
```

### 2. RDT训练数据处理 (`data/hdf5_libero_dataset.py`)

#### 当前处理流程:
```python
# Step 1: LIBERO归一化值 -> 物理单位
pos_normalized = values[:, 0:3]  # [-1, 1]
pos_meters = pos_normalized * 0.012  # 转换为米

ori_normalized = values[:, 3:6]  # [-1, 1]
ori_radians = ori_normalized * 0.5  # 转换为弧度

# Step 2: 弧度 -> 6D rotation
ori_6d = convert_euler_to_6d_rotation(ori_radians)  # 单位向量

# Step 3: 填入128维向量
action_128d[30:33] = pos_meters  # position (米)
action_128d[33:39] = ori_6d      # 6D rotation (单位向量)
action_128d[10] = gripper_01     # gripper [0, 1]
```

#### 问题所在:
**Position和Orientation的量级差异巨大！**
- Position: ~0.012米 (数量级: 10^-2)
- Orientation 6D: ~1.0 (数量级: 10^0)
- **量级相差100倍！**

### 3. Maniskill vs LIBERO对比

| 特性 | Maniskill | LIBERO |
|------|-----------|--------|
| 控制空间 | Joint position (7个关节角) | EEF pose (3D位置 + 旋转) |
| 数据维度 | 7D + 1 gripper | 3D position + 6D rotation + 1 gripper |
| 量级问题 | ✅ 所有关节角都是弧度，量级相同 | ❌ Position(米) vs Rotation(单位向量)，量级差100倍 |
| 归一化 | Min-max归一化到[-1, 1] | ? |
| 物理意义 | 关节角度保持物理意义 | ? |

### 4. README IMPORTANT 3的矛盾

> "No physical quantities (except the gripper width) are normalized during pre-training."

但Maniskill代码中明确使用了min-max归一化:
```python
# maniskill_model.py Line 172
joints = (joints - self.state_min) / (self.state_max - self.state_min) * 2 - 1
```

**可能的解释**:
1. Maniskill使用joint space，所有维度都是角度，量级相同，归一化不破坏相对关系
2. LIBERO使用EEF pose space，不同维度是不同物理量，直接归一化会破坏物理意义

### 5. 可能的解决方案

#### 方案A: 对Position进行缩放
```python
# 将position放大到与orientation相同的量级
pos_scaled = pos_meters * 100  # 0.012 * 100 = 1.2
action_128d[30:33] = pos_scaled
action_128d[33:39] = ori_6d
```

#### 方案B: 对Orientation进行缩放
```python
# 将orientation缩小到与position相同的量级
ori_6d_scaled = ori_6d * 0.01
action_128d[30:33] = pos_meters
action_128d[33:39] = ori_6d_scaled
```

#### 方案C: 分别归一化（可能违反README原则）
```python
# Position: 归一化到[-1, 1]
pos_norm = pos_meters / pos_max  # pos_max from dataset stats

# Orientation: 已经是单位向量，范围在[-1, 1]
action_128d[30:33] = pos_norm
action_128d[33:39] = ori_6d
```

####方案D: 使用不同的loss weighting
```python
# 在训练时，对不同维度使用不同的权重
loss_position = mse_loss(pred_pos, target_pos) * weight_pos
loss_orientation = mse_loss(pred_ori, target_ori) * weight_ori
```
这正是`state_norm`的作用！

### 6. state_norm的作用

在`train/sample.py`中:
```python
loss = F.mse_loss(noise_pred, target, reduction="none")
loss = loss / (state_norm + 1e-3)  # 根据std进行加权
```

**关键**：如果`state_norm`计算错误，会导致训练时权重不平衡！

之前的bug:
```python
# 错误：使用state的RMS
state_norm = np.sqrt(np.mean(libero_states**2, axis=0))

# 修复：使用action的std
state_norm = self.action_std_global
```

但这里`action_std_global`是基于物理单位计算的，会有巨大的量级差异！

### 7. 下一步调查

1. 检查预训练数据的格式和量级
2. 检查预训练时的`state_norm`计算方式
3. 考虑LIBERO是否需要特殊处理






