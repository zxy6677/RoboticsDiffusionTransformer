#!/usr/bin/env python3
"""
测试不同的轴翻转组合，找出正确的坐标系映射

用法：在eval_rdt_libero.py中临时添加轴翻转来测试
"""

print("""
=" * 80)
坐标系问题诊断和测试方案
=" * 80)

问题分析：
1. 数据处理流程验证正确（符号保持）
2. 缩放因子已修复（0.012）
3. State归一化已修复（不归一化）
4. 但机械臂运动方向仍然相反

结论：
RDT预训练数据的坐标系 ≠ LIBERO的坐标系

可能的坐标系差异：
- RDT: X向左, Y向前, Z向上
- LIBERO: X向右, Y向前, Z向上
→ 需要翻转X轴

或：
- RDT: X向右, Y向后, Z向上  
- LIBERO: X向右, Y向前, Z向上
→ 需要翻转Y轴

测试方案：
=" * 80)

方案1：在Action输出层面翻转（评估时）
=" * 80)

在 eval_sim/eval_rdt_libero.py 的 convert_rdt_action_to_libero 函数中：

# 测试1: 翻转X轴
pos_x_norm = -pos_x_meters / 0.012

# 测试2: 翻转Y轴  
pos_y_norm = -pos_y_meters / 0.012

# 测试3: 翻转Z轴
pos_z_norm = -pos_z_meters / 0.012

# 测试4: 翻转XY轴
pos_x_norm = -pos_x_meters / 0.012
pos_y_norm = -pos_y_meters / 0.012

# 测试5: 翻转XZ轴
pos_x_norm = -pos_x_meters / 0.012
pos_z_norm = -pos_z_meters / 0.012

# 测试6: 翻转YZ轴
pos_y_norm = -pos_y_meters / 0.012
pos_z_norm = -pos_z_meters / 0.012

# 测试7: 全部翻转
pos_x_norm = -pos_x_meters / 0.012
pos_y_norm = -pos_y_meters / 0.012
pos_z_norm = -pos_z_meters / 0.012

=" * 80)

方案2：在State输入层面翻转（评估时）
=" * 80)

在 eval_sim/eval_rdt_libero.py 的 convert_libero_state_to_rdt 函数中：

# 翻转State的ee_pos
eef_pos_flipped = eef_pos.copy()
eef_pos_flipped[0] = -eef_pos[0]  # 翻转X
# eef_pos_flipped[1] = -eef_pos[1]  # 翻转Y
# eef_pos_flipped[2] = -eef_pos[2]  # 翻转Z

=" * 80)

快速测试流程：
=" * 80)

1. 选择一个测试（如测试1：翻转X轴）

2. 修改 eval_sim/eval_rdt_libero.py:

   def convert_rdt_action_to_libero(rdt_action: torch.Tensor) -> np.ndarray:
       ...
       # 测试：翻转X轴
       pos_x_norm = -pos_x_meters / 0.012  # 添加负号
       pos_y_norm = pos_y_meters / 0.012
       pos_z_norm = pos_z_meters / 0.012
       ...

3. 评估1个任务（快速）:
   python eval_sim/eval_rdt_libero.py \\
       --pretrained checkpoints/xxx/checkpoint-26000 \\
       --num_tasks 1 \\
       --max_steps 20 \\
       --record_video

4. 观察视频：
   - 如果方向正确了 → 找到了！
   - 如果还是反的 → 尝试下一个测试

5. 找到正确的组合后，记录下来

=" * 80)

推荐测试顺序（从最可能到最不可能）：
=" * 80)

1. 测试2：翻转Y轴（前后）
2. 测试1：翻转X轴（左右）  
3. 测试3：翻转Z轴（上下）
4. 测试4：翻转XY轴
5. 测试7：全部翻转

用户观察到「上下反，左右反」，可能需要测试4（XZ）或测试6（YZ）

=" * 80)

自动化测试脚本（可选）：
=" * 80)

可以创建一个循环，自动测试所有组合：

flip_configs = [
    (False, False, False, '无翻转'),
    (True, False, False, '翻转X'),
    (False, True, False, '翻转Y'),
    (False, False, True, '翻转Z'),
    (True, True, False, '翻转XY'),
    (True, False, True, '翻转XZ'),
    (False, True, True, '翻转YZ'),
    (True, True, True, '翻转XYZ'),
]

for flip_x, flip_y, flip_z, name in flip_configs:
    # 修改转换函数
    # 运行评估
    # 记录结果
    pass

但手动测试可能更快，因为可以立即看到视频效果。

=" * 80)
""")

