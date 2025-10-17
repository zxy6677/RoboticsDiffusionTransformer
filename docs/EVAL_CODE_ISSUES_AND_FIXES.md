# RDT评估代码问题总结与修复

## 🔍 问题发现过程

通过深入分析LIBERO的evaluation实现和RDT的训练数据处理流程，发现了`eval_rdt_libero.py`中的**严重逻辑错误**。

## 🔴 主要问题

### 问题 1: 对Actions的根本性误解 ⚠️⚠️⚠️

**错误的假设**：
```python
# 错误代码假设：
# 1. RDT输出的是"归一化的状态值"（需要用state统计信息反归一化）
# 2. 反归一化后得到物理单位（米、弧度）
# 3. 然后除以增量范围得到[-1, 1]控制信号

# 实际执行：
pos_x = action_128d[pos_x_idx] * state_std[pos_x_idx] + state_mean[pos_x_idx]
libero_action[0] = pos_x / 0.05  # 假设pos_x是米
```

**实际情况**：

通过分析训练代码（`data/hdf5_libero_dataset.py`），发现：

1. **训练数据中的actions**：
```python
# 从HDF5文件读取LIBERO原始actions
actions = episode_data['actions'][:]  # (T, 7)
# 这些actions是归一化的增量控制信号，范围 [-1, 1]
# 物理含义：
#   - 位置增量: [-1, 1] → [-0.05m, 0.05m]
#   - 旋转增量: [-1, 1] → [-0.5rad, 0.5rad]
#   - gripper: [-1, 1] (直接控制)

# 转换处理
def fill_in_action(values):
    ori_3d = values[:, 3:6]  # 3D欧拉角增量
    ori_6d = convert_euler_to_6d_rotation(ori_3d)  # 转6D
    gripper_normalized = (values[:, 6:7] + 1.0) / 2.0  # [-1,1] → [0,1]
    
    action_10d = np.concatenate([
        values[:, 0:3],        # 位置增量保持不变 (仍是[-1, 1]范围)
        ori_6d,                # 6D旋转
        gripper_normalized     # [0, 1]
    ], axis=1)
    
    # 填充到128维
    uni_vec[..., action_indices] = action_10d
    return uni_vec
```

2. **RDT训练**：
```python
# train/dataset.py
data_dict["actions"] = actions  # 直接使用，没有额外归一化！

# models/rdt_runner.py - compute_loss
action_gt = actions  # ground truth actions
loss = F.mse_loss(pred, target)  # 直接学习预测这些values
```

**结论**：
- RDT学习预测的是**归一化的增量控制信号**，不是物理单位！
- 位置：[-1, 1] 范围（代表相对增量）
- 旋转：6D表示（对应[-0.5, 0.5]弧度的增量范围）
- Gripper：[0, 1] 范围

### 问题 2: 错误的反归一化流程

**错误代码**：
```python
# ❌ 用state的统计信息处理action（完全错误！）
pos_x = action_128d[pos_x_idx] * libero_stats["state_std"][pos_x_idx] + \
        libero_stats["state_mean"][pos_x_idx]

# ❌ 假设得到的是米，然后除以增量范围
libero_action[0] = pos_x / 0.05
```

**为什么这是错的**：
1. `state_mean`和`state_std`是**状态观测**的统计信息（例如关节角度、末端位置）
2. **Actions从未使用state统计信息归一化过**
3. State和Action是不同的空间，使用state统计处理action毫无意义

**举例说明错误**：
```python
# state_mean[30] = -0.079  # 这是平均末端位置X坐标（米）
# state_std[30] = 0.05     # 这是末端位置X的标准差（米）

# 假设RDT输出 action_128d[30] = 0.5 （应该是[-1, 1]范围的增量）
pos_x = 0.5 * 0.05 + (-0.079) = -0.054  # 得到一个"位置"
libero_action[0] = -0.054 / 0.05 = -1.08  # 错误地认为是增量

# 实际上0.5应该直接作为控制信号使用！
```

### 问题 3: 性能问题

每次调用`convert_rdt_action_to_libero`都读取JSON文件：
```python
def convert_rdt_action_to_libero(rdt_action):
    with open('configs/dataset_stat.json', 'r') as f:  # ❌ 每次都读取
        stats = json.load(f)
```

在评估循环中（100步），会读取100次文件。

### 问题 4: 重复导入

函数内部重复导入：
```python
def convert_libero_state_to_rdt(obs):
    import sys
    import os
    sys.path.append(...)
    from utils.rotation_utils import ...
```

## ✅ 修复方案

### 修复 1: 正确的Action转换逻辑

**修复后的代码**：
```python
def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    """将RDT动作转换为LIBERO动作格式
    
    重要：RDT训练时学习的是归一化的增量控制信号，不是绝对的物理值！
    """
    action_128d = rdt_action[0, 0, :].cpu().numpy()
    
    # ✅ 直接提取位置增量（已经是[-1, 1]范围）
    pos_x = action_128d[pos_x_idx]  # 不需要反归一化！
    pos_y = action_128d[pos_y_idx]
    pos_z = action_128d[pos_z_idx]
    
    # ✅ 提取6D旋转并转换为欧拉角
    ori_6d = np.array([action_128d[idx] for idx in ori_indices])
    ori_euler = convert_6d_rotation_to_euler(ori_6d)
    
    # ✅ 欧拉角转换为[-1, 1]控制信号
    # 假设ori_euler在[-0.5, 0.5]弧度范围
    ori_normalized = ori_euler / 0.5
    
    # ✅ Gripper从[0, 1]映射回[-1, 1]
    gripper_01 = action_128d[gripper_idx]
    gripper = gripper_01 * 2.0 - 1.0
    
    # ✅ 构建LIBERO动作（所有值都在[-1, 1]范围）
    libero_action = np.array([
        pos_x, pos_y, pos_z,
        ori_normalized[0], ori_normalized[1], ori_normalized[2],
        gripper
    ])
    
    # ✅ Clip确保安全
    libero_action = np.clip(libero_action, -1.0, 1.0)
    
    return libero_action
```

**关键改变**：
1. ❌ 删除：用state统计反归一化
2. ❌ 删除：除以物理单位范围(0.05, 0.5)
3. ✅ 新增：直接使用RDT输出
4. ✅ 新增：仅对旋转进行范围转换(6D → 欧拉角 → 归一化)

### 修复 2: 缓存JSON读取

```python
# 模块级别缓存
_LIBERO_STATS = None

def _get_libero_stats():
    global _LIBERO_STATS
    if _LIBERO_STATS is None:
        with open('configs/dataset_stat.json', 'r') as f:
            stats = json.load(f)
        _LIBERO_STATS = stats['libero_90']
    return _LIBERO_STATS
```

### 修复 3: 移到模块级别导入

```python
# 文件顶部
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.rotation_utils import (
    convert_quaternion_to_6d_rotation,
    convert_6d_rotation_to_euler
)
```

## 📊 影响分析

### 原错误代码的影响

1. **动作数值完全错误**：
   ```
   原本应该：pos = 0.5 (增量控制信号)
   错误计算：pos = 0.5 * std + mean = 某个位置值
              然后 / 0.05 = 完全错误的数值
   ```

2. **机器人行为异常**：
   - 动作幅度可能过大或过小
   - 方向可能错误
   - 无法完成任务

3. **评估结果不可信**：
   - 成功率可能接近0
   - 不能反映模型真实能力

### 修复后的预期

1. **正确的动作范围**：
   - 所有动作在[-1, 1]范围内
   - 符合LIBERO的控制规范

2. **合理的机器人行为**：
   - 增量控制正常工作
   - 动作幅度合理

3. **可信的评估结果**：
   - 反映RDT模型的真实性能

## 🔬 验证方法

### 1. 打印动作值范围

在评估循环中添加：
```python
print(f"RDT output range: [{rdt_actions.min():.3f}, {rdt_actions.max():.3f}]")
print(f"LIBERO action range: [{libero_action.min():.3f}, {libero_action.max():.3f}]")
```

**预期**：
- LIBERO action应该在[-1, 1]范围内
- 不应该出现大量的clip（所有值都是±1）

### 2. 检查机器人运动

- 机器人应该有小幅度的增量运动
- 不应该出现突然的大幅跳跃
- Gripper应该能正常开合

### 3. 对比训练数据

可以从训练数据中采样actions，对比：
```python
# 训练数据中的action范围应该和评估输出相近
training_action = dataset.get_item()['actions']
print(f"Training action range: [{training_action.min():.3f}, {training_action.max():.3f}]")
```

## 📝 总结

### 根本问题

**对RDT输出空间的误解**：
- 错误理解：RDT输出是"归一化的状态" → 需要反归一化到物理单位
- 正确理解：RDT输出是"归一化的增量控制信号" → 直接使用

### 关键修复

1. **删除错误的反归一化**：不使用state统计处理action
2. **直接使用RDT输出**：仅进行必要的格式转换（6D→欧拉角）
3. **优化性能**：缓存JSON读取

### 重要性

这是一个**严重的逻辑错误**，会导致：
- ❌ 完全错误的动作执行
- ❌ 不可信的评估结果
- ❌ 无法反映模型真实性能

修复后应该能看到：
- ✅ 合理的机器人行为
- ✅ 可信的评估指标
- ✅ RDT模型的真实性能

## 🔄 下一步

1. **测试修复后的代码**：
   ```bash
   python eval_sim/eval_rdt_libero.py --num_tasks 2 --max_steps 50 --record_video
   ```

2. **验证动作范围**：检查输出的动作值是否合理

3. **对比性能**：修复前后的成功率对比

4. **可视化验证**：查看录制的视频，确认机器人运动合理

---

**修复日期**: 2025-10-16  
**修复文件**: `eval_sim/eval_rdt_libero.py`  
**修复内容**: `convert_rdt_action_to_libero` 函数的完全重写

