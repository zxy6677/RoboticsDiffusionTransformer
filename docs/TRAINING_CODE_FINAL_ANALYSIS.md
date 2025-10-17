# LIBERO微调训练代码最终分析报告

## 🔴 关键发现：训练数据格式问题

### 实际训练数据分析

通过检查实际训练样本，发现了**严重的数据格式问题**：

#### 训练样本的Actions值

```
Actions范围: [-0.616071, 1.000000]

位置 (idx 30-32):
  pos_x: range=[-0.0054, 0.3295], mean=0.1092
  pos_y: range=[-0.3723, 0.0000], mean=-0.2498
  pos_z: range=[-0.6161, -0.0000], mean=-0.2411

6D旋转 (idx 33-38):
  angle_0: range=[0.9544, 1.0000], mean=0.9777
  angle_1: range=[-0.1090, 0.2986], mean=0.1532
  angle_2: range=[-0.1101, 0.0268], mean=-0.0346
  angle_3: range=[-0.2956, 0.1107], mean=-0.1550
  angle_4: range=[0.9446, 1.0000], mean=0.9733
  angle_5: range=[-0.1462, 0.0717], mean=-0.0797

Gripper (idx 10):
  range=[0.0000, 0.0000], mean=0.0000
```

#### 原始LIBERO数据

```
Actions范围: [-1.0, 0.908]

位置:
  pos_x: range=[-0.329, 0.908]
  pos_y: range=[-0.391, 0.846]
  pos_z: range=[-0.359, 0.000]

旋转（欧拉角）:
  ori_x: range=[-0.009, 0.000]
  ori_y: range=[-0.095, 0.000]
  ori_z: range=[-0.029, 0.058]

Gripper:
  range=[-1.0, -1.0]
```

---

## ❌ 问题确认：与README要求不符

### README的明确要求（IMPORTANT 3）

> **IMPORTANT 3:** No physical quantities (except the gripper width) are **normalized** during pre-training. This can preserve each physical quantity's meaning, thereby promoting generalization across robots. Therefore, we encourage you **not to normalize any physical quantities** but to **choose appropriate units** for them. Generally, we use the **International System of Units**, which ensures that most values fall within [-1,1].

**翻译**：
- ❌ 不要归一化物理量（gripper除外）
- ✅ 应该使用物理单位（国际单位制）
- 📏 位置应该用**米**
- 📐 旋转应该用**弧度**

### 当前实现的问题

**位置处理**（`hdf5_libero_dataset.py:200-201`）：
```python
action_10d = np.concatenate([
    values[:, 0:3],  # ❌ 这是归一化值 [-1, 1]，不是米！
    ori_6d,          # ❌ 从归一化的欧拉角转换的
    gripper_normalized
], axis=1)
```

**现状**：
- LIBERO actions: `[-1, 1]` 归一化范围
  - 位置: `[-1, 1]` → 对应 `[-0.05m, 0.05m]` 物理增量
  - 旋转: `[-1, 1]` → 对应 `[-0.5rad, 0.5rad]` 物理增量

**问题**：
- ❌ 训练时直接使用了归一化值
- ❌ 没有转换为物理单位（米、弧度）
- ❌ 违反了README的IMPORTANT 3要求

---

## 📊 影响分析

### 1. 与预训练模型的数据分布不匹配

**预训练模型期望**：
- 位置值在米的数量级：约 `[-0.1, 0.1]` 米
- 旋转值在弧度的数量级：约 `[-0.5, 0.5]` 弧度

**当前训练数据**：
- 位置值: `[-1, 1]` 归一化值（无物理意义）
- 旋转值: 从归一化欧拉角转换的6D表示

**后果**：
- 🔴 数据分布严重偏移
- 🔴 模型难以从预训练权重中迁移学习
- 🔴 可能导致训练不稳定或收敛到次优解

### 2. 评估结果 0% 的根本原因

现在可以解释为什么评估成功率为 0%：

1. **训练时的数据格式错误**
   - 模型学习的是归一化值的分布
   - 预训练权重期望物理单位的分布
   - 两者不匹配

2. **评估时的数据转换也有问题**
   - 评估代码之前也错误地使用了state统计（已修复）
   - 但模型本身训练时就用错了数据格式

3. **动作幅度完全错误**
   - 模型输出的数值范围与LIBERO期望的不匹配
   - 导致机器人执行错误的动作

---

## ✅ 正确的实现方案

### 修复方案：转换为物理单位

修改 `data/hdf5_libero_dataset.py` 中的 `fill_in_action` 函数：

```python
def fill_in_action(values):
    """
    将LIBERO actions转换为RDT统一动作空间
    
    重要：按照README IMPORTANT 3的要求，使用物理单位！
    """
    # LIBERO actions: 7D [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, gripper]
    # 范围: [-1, 1] 归一化范围
    
    # === 步骤1: 转换位置为物理单位（米） ===
    # LIBERO: [-1, 1] 对应 [-0.05m, 0.05m]
    pos_normalized = values[:, 0:3]  # (T, 3)
    pos_meters = pos_normalized * 0.05  # 转换为米
    # 现在范围: 约 [-0.05, 0.05] 米
    
    # === 步骤2: 转换旋转为物理单位（弧度） ===
    # LIBERO: [-1, 1] 对应 [-0.5rad, 0.5rad]
    ori_normalized = values[:, 3:6]  # (T, 3) 欧拉角
    ori_radians = ori_normalized * 0.5  # 转换为弧度
    # 现在范围: 约 [-0.5, 0.5] 弧度
    
    # === 步骤3: 转换为6D旋转表示 ===
    ori_6d = convert_euler_to_6d_rotation(ori_radians)  # (T, 6)
    # 注意：6D旋转从物理单位的弧度转换而来
    
    # === 步骤4: Gripper归一化（按README，这是唯一需要归一化的） ===
    # LIBERO: [-1, 1] → [0, 1]
    gripper_raw = values[:, 6:7]  # (T, 1)
    gripper_normalized = (gripper_raw + 1.0) / 2.0  # Map [-1,1] to [0,1]
    
    # === 步骤5: 组合为10D动作向量 ===
    action_10d = np.concatenate([
        pos_meters,           # 位置：米（物理单位）
        ori_6d,              # 旋转：6D表示（从弧度转换）
        gripper_normalized   # Gripper：[0, 1]（归一化）
    ], axis=1)  # (T, 10)
    
    # === 步骤6: 映射到128维统一动作空间 ===
    UNI_ACTION_INDICES = [
        STATE_VEC_IDX_MAPPING["right_eef_pos_x"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_y"],
        STATE_VEC_IDX_MAPPING["right_eef_pos_z"]
    ] + [
        STATE_VEC_IDX_MAPPING["right_eef_angle_0"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_1"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_2"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_3"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_4"],
        STATE_VEC_IDX_MAPPING["right_eef_angle_5"]
    ] + [
        STATE_VEC_IDX_MAPPING["right_gripper_open"]
    ]
    
    uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
    uni_vec[..., UNI_ACTION_INDICES] = action_10d
    return uni_vec
```

### 预期效果

修复后的训练数据：

```
位置（物理单位：米）:
  pos_x: range ≈ [-0.05, 0.05] 米
  pos_y: range ≈ [-0.05, 0.05] 米
  pos_z: range ≈ [-0.05, 0.05] 米

6D旋转（从弧度转换）:
  从物理单位的弧度 [-0.5, 0.5] 转换
  6D表示的值范围会相应调整

Gripper（归一化）:
  range = [0, 1]
```

---

## 🔄 评估代码也需要相应修复

### 当前评估代码的问题

文件：`eval_sim/eval_rdt_libero.py`

**当前的 `convert_rdt_action_to_libero`**（已经部分修复）：
```python
# 直接使用RDT输出（假设是归一化值）
pos_x = action_128d[pos_x_idx]
```

**问题**：
- 如果训练时使用物理单位，RDT输出也是物理单位
- 需要将物理单位转换回LIBERO的归一化范围

### 正确的评估代码

```python
def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """
    将RDT动作（物理单位）转换为LIBERO动作格式（归一化）
    """
    action_128d = rdt_action[0, 0, :].cpu().numpy()
    
    # === 提取位置（物理单位：米） ===
    pos_x_meters = action_128d[STATE_VEC_IDX_MAPPING["right_eef_pos_x"]]
    pos_y_meters = action_128d[STATE_VEC_IDX_MAPPING["right_eef_pos_y"]]
    pos_z_meters = action_128d[STATE_VEC_IDX_MAPPING["right_eef_pos_z"]]
    
    # 转换为LIBERO归一化范围: 米 → [-1, 1]
    # [-0.05, 0.05]米 对应 [-1, 1]
    pos_x_norm = np.clip(pos_x_meters / 0.05, -1.0, 1.0)
    pos_y_norm = np.clip(pos_y_meters / 0.05, -1.0, 1.0)
    pos_z_norm = np.clip(pos_z_meters / 0.05, -1.0, 1.0)
    
    # === 提取6D旋转并转换为欧拉角（弧度） ===
    ori_indices = [STATE_VEC_IDX_MAPPING[f"right_eef_angle_{i}"] for i in range(6)]
    ori_6d = np.array([action_128d[idx] for idx in ori_indices])
    ori_euler_rad = convert_6d_rotation_to_euler(ori_6d)  # 弧度
    
    # 转换为LIBERO归一化范围: 弧度 → [-1, 1]
    # [-0.5, 0.5]弧度 对应 [-1, 1]
    ori_normalized = np.clip(ori_euler_rad / 0.5, -1.0, 1.0)
    
    # === 提取Gripper ===
    gripper_01 = action_128d[STATE_VEC_IDX_MAPPING["right_gripper_open"]]
    gripper_norm = gripper_01 * 2.0 - 1.0  # [0, 1] → [-1, 1]
    
    # === 构建LIBERO动作 ===
    libero_action = np.array([
        pos_x_norm, pos_y_norm, pos_z_norm,
        ori_normalized[0], ori_normalized[1], ori_normalized[2],
        gripper_norm
    ])
    
    return libero_action
```

---

## 📝 修复步骤

### 步骤1: 修复训练数据加载器 ⭐⭐⭐

```bash
# 编辑 data/hdf5_libero_dataset.py
# 修改 fill_in_action 函数（见上面的正确实现）
```

### 步骤2: 重新计算数据集统计信息

```bash
cd /home/ubuntu/RoboticsDiffusionTransformer
python -m data.compute_dataset_stat_hdf5
```

### 步骤3: 修复评估代码

```bash
# 编辑 eval_sim/eval_rdt_libero.py
# 修改 convert_rdt_action_to_libero 函数（见上面的正确实现）
```

### 步骤4: 重新训练模型

```bash
# 从头开始训练（推荐）
python main.py \
    --pretrained_model_name_or_path=robotics-diffusion-transformer/rdt-1b \
    --output_dir=checkpoints/libero_finetune_fixed \
    --dataset_type=finetune \
    --load_from_hdf5 \
    ... (其他参数)

# 或者从现有checkpoint继续（如果想节省时间）
# 但可能效果不如重新训练
```

### 步骤5: 评估修复后的模型

```bash
python eval_sim/eval_rdt_libero.py \
    --pretrained checkpoints/libero_finetune_fixed/checkpoint-XXXXX \
    --num_tasks 5 \
    --max_steps 100 \
    --record_video
```

---

## 🎯 预期改进

修复后，预期能看到：

1. **训练过程**
   - ✅ 数据分布与预训练模型匹配
   - ✅ 从预训练权重更好地迁移学习
   - ✅ Loss更稳定地下降
   - ✅ sample_mse更低

2. **评估结果**
   - ✅ 成功率显著提升（从0%到可能20-50%）
   - ✅ 机器人运动更合理
   - ✅ 动作幅度正确

3. **长期效果**
   - ✅ 更好的泛化能力
   - ✅ 符合RDT的设计理念
   - ✅ 便于与其他机器人数据融合

---

## 📋 总结

### 🔴 核心问题

**训练数据使用了归一化的增量控制信号，而不是README要求的物理单位。**

### 🎯 根本原因

1. LIBERO的actions是归一化值
2. 代码直接使用了这些归一化值
3. 没有转换为物理单位（米、弧度）
4. 违反了README的IMPORTANT 3要求

### ✅ 解决方案

1. 修改 `fill_in_action` 函数
2. 将归一化值转换为物理单位
3. 重新计算数据集统计
4. 修复评估代码的相应转换
5. 重新训练模型

### 📊 优先级

- **🔴 紧急**: 修复训练数据加载器（最关键）
- **🟡 重要**: 修复评估代码
- **🟢 次要**: 优化其他细节

---

**报告日期**: 2025-10-16  
**版本**: 2.0 Final  
**状态**: ✅ 问题确认，✅ 解决方案明确

