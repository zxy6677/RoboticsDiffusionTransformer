# 坐标轴翻转修复方案

## 问题确认

经过详细分析，确认：
1. ✅ 数据处理流程正确
2. ✅ 缩放因子正确（0.012）
3. ✅ State归一化已修复
4. ❌ **但机械臂运动方向仍相反**

**结论**：RDT预训练数据的坐标系与LIBERO不同，需要轴翻转。

---

## 用户观察到的现象

> "上下反，左右反"

这暗示需要翻转：
- **X轴**（左右）
- **Z轴**（上下）

或者可能是：
- **Y轴**（前后）+ 某个其他轴

---

## 快速测试方案

### 方法1：修改评估代码（推荐，最快）

编辑 `eval_sim/eval_rdt_libero.py`，在 `convert_rdt_action_to_libero` 函数中：

```python
def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    ...
    # 原始代码（第372-374行）：
    # pos_x_norm = pos_x_meters / 0.012
    # pos_y_norm = pos_y_meters / 0.012
    # pos_z_norm = pos_z_meters / 0.012
    
    # 测试：根据观察到的「上下反，左右反」
    # 尝试翻转X和Z轴
    pos_x_norm = -pos_x_meters / 0.012  # 翻转X（左右）
    pos_y_norm = pos_y_meters / 0.012   # Y不变（前后）
    pos_z_norm = -pos_z_meters / 0.012  # 翻转Z（上下）
    ...
```

### 测试步骤

1. 修改代码（添加负号）
2. 运行快速评估：
   ```bash
   python eval_sim/eval_rdt_libero.py \
       --pretrained checkpoints/xxx/checkpoint-26000 \
       --num_tasks 1 \
       --max_steps 20 \
       --record_video \
       --video_output_dir videos/test_flip_xz
   ```
3. 查看视频，如果方向正确了 → 成功！

### 如果XZ翻转不对，尝试其他组合

| 组合 | X | Y | Z | 描述 |
|------|---|---|---|------|
| 1 | - | + | - | 翻转X和Z（根据用户描述）|
| 2 | - | + | + | 只翻转X |
| 3 | + | - | - | 翻转Y和Z |
| 4 | + | + | - | 只翻转Z |
| 5 | + | - | + | 只翻转Y |
| 6 | - | - | + | 翻转X和Y |
| 7 | - | - | - | 全部翻转 |

修改代码时：
- `+` 表示保持：`pos_x_norm = pos_x_meters / 0.012`
- `-` 表示翻转：`pos_x_norm = -pos_x_meters / 0.012`

---

## 代码修改位置

### eval_sim/eval_rdt_libero.py

找到第372-374行：

```python
# === 步骤1: 提取位置（物理单位：米） ===
...
pos_x_meters = action_128d[pos_x_idx]
pos_y_meters = action_128d[pos_y_idx]
pos_z_meters = action_128d[pos_z_idx]

# 转换为LIBERO的归一化范围: 米 → [-1, 1]
# 修正：使用实际测量的缩放因子 0.012 而不是 0.05

# ⚠️ 添加轴翻转测试（根据RDT预训练坐标系）
pos_x_norm = -pos_x_meters / 0.012  # 翻转X
pos_y_norm = pos_y_meters / 0.012   # 不翻转Y
pos_z_norm = -pos_z_meters / 0.012  # 翻转Z
```

### 同时可能需要翻转旋转

如果位置翻转后正确了，但旋转还是有问题，也需要翻转旋转：

```python
# === 步骤3: 转换为LIBERO的归一化范围: 弧度 → [-1, 1] ===
ori_x_norm = -ori_euler_rad[0] / 0.5  # 翻转
ori_y_norm = ori_euler_rad[1] / 0.5
ori_z_norm = -ori_euler_rad[2] / 0.5  # 翻转
```

---

## 一旦找到正确组合

### 1. 记录正确的翻转配置

例如：
```
✅ 正确配置：翻转X和Z轴
- pos_x: 需要翻转（负号）
- pos_y: 不翻转
- pos_z: 需要翻转（负号）
```

### 2. 更新训练代码（可选）

如果想从头重新训练一个更"正确"的模型，可以在训练数据处理时就翻转：

**data/hdf5_libero_dataset.py**:
```python
# 如果确认需要翻转X和Z
pos_meters = pos_normalized * 0.012
pos_meters[0] = -pos_meters[0]  # 翻转X
pos_meters[2] = -pos_meters[2]  # 翻转Z
```

然后重新训练。但这不是必须的，因为评估时翻转也能工作。

### 3. 提交修复

```bash
git add eval_sim/eval_rdt_libero.py
git commit -m "Fix coordinate system mismatch: flip X and Z axes

RDT pretraining uses different coordinate convention than LIBERO.
Flip X (left-right) and Z (up-down) axes to match LIBERO's coordinate system."
git push
```

---

## 调试日志建议

在测试时，可以添加日志来验证：

```python
def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
    ...
    # 调试：打印翻转前后的值
    print(f"Before flip: X={pos_x_meters:.4f}, Y={pos_y_meters:.4f}, Z={pos_z_meters:.4f}")
    
    pos_x_norm = -pos_x_meters / 0.012
    pos_y_norm = pos_y_meters / 0.012
    pos_z_norm = -pos_z_meters / 0.012
    
    print(f"After flip: X={pos_x_norm:.4f}, Y={pos_y_norm:.4f}, Z={pos_z_norm:.4f}")
    ...
```

---

## 预期结果

找到正确的轴翻转组合后：
- ✅ 机械臂运动方向与demo一致
- ✅ 上下、左右、前后都正确
- ✅ 夹爪正常开合
- ✅ 能成功完成任务

---

## 总结

1. **根本原因**：RDT预训练数据 ≠ LIBERO的坐标系定义
2. **解决方案**：在评估时翻转对应的轴
3. **推荐测试**：先试XZ翻转（根据「上下反，左右反」）
4. **快速验证**：每次测试只需20步，几分钟就能看结果

立即尝试！修改代码 → 测试 → 找到正确组合 → 提交修复

