# LIBERO微调训练代码分析报告

## 🔍 实际数据分析

### LIBERO原始Actions数据
从 `KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5` 检查结果：

```
Actions shape: (65, 7)
Actions范围: [-1.0, 0.908]
格式: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
```

**每个维度统计**：
| 维度 | 含义 | 最小值 | 最大值 | 均值 | 标准差 |
|-----|------|--------|--------|------|--------|
| 0 | pos_x | -0.329 | 0.908 | 0.212 | 0.431 |
| 1 | pos_y | -0.391 | 0.846 | 0.356 | 0.477 |
| 2 | pos_z | -0.359 | 0.000 | -0.238 | 0.116 |
| 3 | ori_x | -0.009 | 0.000 | -0.002 | 0.003 |
| 4 | ori_y | -0.095 | 0.000 | -0.036 | 0.031 |
| 5 | ori_z | -0.029 | 0.058 | 0.004 | 0.020 |
| 6 | gripper | -1.000 | -1.000 | -1.000 | 0.000 |

**关键发现**：
- ✅ Actions已经是归一化的增量控制信号（[-1, 1]范围）
- ✅ Gripper使用[-1, 1]表示（-1=关闭，1=打开）
- ✅ 这与LIBERO的OSC_POSE控制器一致

---

## 📋 训练代码分析

### 1. 数据加载器实现

文件：`data/hdf5_libero_dataset.py`

#### ✅ 正确的部分

**1.1 Actions读取**（L154）
```python
actions_seq = actions[step_id:step_id+self.CHUNK_SIZE]
```
✅ 直接读取原始actions，保持[-1, 1]范围

**1.2 状态映射**（L163-185）
```python
def fill_in_state(values):
    # 将LIBERO 17维状态映射到RDT 128维空间
    # 使用right_arm部分（single-arm要求）
    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]] + ...
    
    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
    uni_vec[..., UNI_STATE_INDICES] = values
    return uni_vec
```
✅ 符合README要求：单臂机器人填充到right arm部分

**1.3 旋转转换**（L191-192）
```python
ori_3d = values[:, 3:6]  # 3D欧拉角
ori_6d = convert_euler_to_6d_rotation(ori_3d)  # 转换为6D
```
✅ 符合README IMPORTANT 2：使用6D旋转表示

**1.4 Gripper归一化**（L195-197）
```python
gripper_raw = values[:, 6:7]  # [-1, 1]
gripper_normalized = (gripper_raw + 1.0) / 2.0  # 转换为[0, 1]
```
✅ 符合README IMPORTANT 3：Gripper使用min-max归一化到[0, 1]

**1.5 Actions映射**（L199-223）
```python
action_10d = np.concatenate([
    values[:, 0:3],        # 位置 (3D) - 保持原值
    ori_6d,                # 旋转 (6D)
    gripper_normalized     # gripper (1D) [0,1]
], axis=1)  # (T, 10)

UNI_ACTION_INDICES = [
    STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
    STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
] + [
    STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
    ...
    STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
] + [
    STATE_VEC_IDX_MAPPING["right_gripper_open"]
]

uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
uni_vec[..., UNI_ACTION_INDICES] = action_10d
```
✅ 正确映射到128维统一动作空间

---

### ❌ 发现的问题

#### 问题 1: 位置和旋转值的物理单位混淆 ⚠️⚠️⚠️

**README要求**（IMPORTANT 3）：
> No physical quantities (except the gripper width) are normalized during pre-training. This can preserve each physical quantity's meaning, thereby promoting generalization across robots. Therefore, we encourage you **not to normalize any physical quantities** but to **choose appropriate units** for them. Generally, we use the **International System of Units**.

**当前实现问题**：
```python
# 在 fill_in_action() 中
action_10d = np.concatenate([
    values[:, 0:3],  # ❌ 这是归一化的增量信号[-1, 1]，不是物理单位！
    ori_6d,          # ❌ 从归一化的欧拉角转换来的
    gripper_normalized
], axis=1)
```

**问题分析**：

LIBERO的actions是**归一化的增量控制信号**：
- 位置: [-1, 1] 对应 [-0.05m, 0.05m] 的物理增量
- 旋转: [-1, 1] 对应 [-0.5rad, 0.5rad] 的物理增量

但RDT的README要求：
- 位置应该用**米**（物理单位）
- 旋转应该用**弧度**（物理单位）

**当前代码直接使用归一化值**，这违反了README的IMPORTANT 3要求！

---

#### 问题 2: Actions的语义混淆

**LIBERO actions语义**：
- 代表**增量控制信号**（delta control）
- 用于OSC_POSE控制器的输入

**RDT训练期望**：
- 根据README，应该是**物理单位的值**
- 但实际预训练数据可能使用不同的约定

**矛盾点**：
1. 如果RDT在预训练时学习的是**归一化的控制信号**，那么当前实现可能是对的
2. 如果RDT在预训练时学习的是**物理单位的增量**，那么需要转换

---

## 🔧 修复建议

### 方案A：转换为物理单位（严格遵循README）

```python
def fill_in_action(values):
    # LIBERO actions: [-1, 1] 归一化范围
    # 需要转换为物理单位
    
    # 位置：[-1, 1] → [-0.05, 0.05] 米
    pos_normalized = values[:, 0:3]  # (T, 3)
    pos_meters = pos_normalized * 0.05  # 转换为米
    
    # 旋转：[-1, 1] → [-0.5, 0.5] 弧度
    ori_normalized = values[:, 3:6]  # (T, 3)
    ori_radians = ori_normalized * 0.5  # 转换为弧度
    ori_6d = convert_euler_to_6d_rotation(ori_radians)  # 转换为6D
    
    # Gripper: [-1, 1] → [0, 1]
    gripper_raw = values[:, 6:7]
    gripper_normalized = (gripper_raw + 1.0) / 2.0
    
    action_10d = np.concatenate([
        pos_meters,           # 物理单位：米
        ori_6d,              # 物理单位：6D旋转（从弧度转换）
        gripper_normalized   # 归一化：[0, 1]
    ], axis=1)
    
    # ... 其余代码相同
```

### 方案B：保持当前实现（假设预训练使用归一化值）

如果预训练数据集也使用归一化的控制信号，那么当前实现可能是正确的。

**验证方法**：
1. 检查预训练数据集的actions范围
2. 查看其他数据集的加载器实现（`hdf5_vla_dataset.py`）
3. 对比训练效果

---

## 🔍 进一步调查

### 检查预训练数据集的处理

让我们查看 `hdf5_vla_dataset.py` 中其他数据集是如何处理的：

```python
# 需要检查的关键点：
# 1. 其他数据集的actions是否也是归一化值？
# 2. 是否有物理单位转换？
# 3. 评论中是否有说明？
```

### 检查训练配置

```python
# configs/dataset_stat.json 中的统计信息
# 如果 libero_90 的 action_mean/std 显示值在物理单位范围内，
# 说明应该使用物理单位
```

---

## 📊 当前评估结果的解释

### 评估结果：成功率 0%

**可能原因分析**：

1. **数据格式不匹配** ⭐⭐⭐
   - 训练时使用了归一化值[-1, 1]
   - 但预训练模型期望物理单位
   - 导致动作幅度和含义不匹配

2. **评估代码的动作转换错误** ✅ 已修复
   - 之前使用state统计信息反归一化action（错误）
   - 已修复为直接使用RDT输出

3. **训练步数不足**
   - Checkpoint-25000可能还没有充分学习

4. **坐标系映射问题**
   - X/Z轴的反转可能不正确

---

## ✅ 已确认正确的部分

1. ✅ **数据集路径配置**
   - `configs/finetune_datasets.json` 包含 libero_90
   - `configs/dataset_control_freq.json` 设置为20Hz
   - `configs/finetune_sample_weights.json` 有采样权重

2. ✅ **状态向量映射**
   - 正确使用right_arm部分（单臂要求）
   - 索引映射符合`state_vec.py`

3. ✅ **6D旋转转换**
   - 正确实现了欧拉角到6D旋转的转换
   - 符合README要求

4. ✅ **Gripper归一化**
   - 正确将[-1, 1]归一化到[0, 1]
   - 符合README要求

5. ✅ **图像处理**
   - 正确加载agentview和eye_in_hand图像
   - 正确处理图像历史和mask

---

## 🎯 推荐行动

### 立即行动（高优先级）

1. **验证预训练数据集的actions格式**
   ```bash
   # 检查 hdf5_vla_dataset.py 中的处理
   # 对比其他数据集的actions范围
   ```

2. **测试两种方案**
   - 方案A：转换为物理单位（修改训练代码）
   - 方案B：保持当前实现（验证预训练格式）

3. **查看训练日志**
   ```bash
   # 检查训练过程中的loss和sample_mse
   # 如果loss正常下降，说明训练本身没问题
   ```

### 后续行动（中优先级）

1. **增加训练步数**
   - 从checkpoint-25000继续训练到50000+步
   - 观察是否有性能提升

2. **微调评估代码**
   - 验证坐标系转换是否正确
   - 测试不同的轴反转组合

3. **对比分析**
   - 将RDT输出的actions与训练数据对比
   - 检查分布是否匹配

---

## 📝 总结

### 🔴 关键问题

**最重要的问题**：训练代码中的actions处理可能与README要求不一致

- **现状**：直接使用归一化的增量控制信号[-1, 1]
- **README要求**：应该使用物理单位（米、弧度）
- **影响**：可能导致训练的模型与预训练模型的数据分布不匹配

### ✅ 已正确实现

- 数据集配置和路径
- 状态向量映射到right_arm
- 6D旋转表示
- Gripper归一化到[0, 1]
- 图像加载和处理

### 🔍 需要验证

1. 预训练数据集是否使用归一化值还是物理单位
2. 其他数据集的处理方式
3. 训练效果和收敛情况

### 💡 建议

**优先级1**：查看 `hdf5_vla_dataset.py` 和其他数据集的处理，确定正确的数据格式

**优先级2**：根据验证结果，修改 `hdf5_libero_dataset.py` 的actions处理

**优先级3**：重新训练或从checkpoint继续训练，观察效果

---

**报告生成时间**: 2025-10-16  
**分析版本**: 1.0

