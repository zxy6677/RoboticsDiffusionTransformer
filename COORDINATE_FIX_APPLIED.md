# 坐标系翻转修复 - 已应用

**日期**: 2025-10-17  
**修改**: 在训练和评估代码中统一翻转所有位置轴

---

## ✅ 已完成的修改

### 1. 训练代码（数据加载器）

**文件**: `data/hdf5_libero_dataset.py` 第205-209行

**修改内容**:
```python
# 添加了坐标系转换
pos_meters = pos_normalized * 0.012  # 转换为米
pos_meters = -pos_meters  # 翻转所有位置轴（X, Y, Z）
```

**原理**:
- LIBERO坐标系 → 翻转 → RDT坐标系
- 训练时将LIBERO的动作转换为RDT期望的坐标系

### 2. 评估代码

**文件**: `eval_sim/eval_rdt_libero.py` 第411-415行

**修改内容**:
```python
libero_action = np.array([
    -pos_x_norm, -pos_y_norm, -pos_z_norm,  # 从RDT坐标系转换回LIBERO
    ori_x_norm, ori_y_norm, ori_z_norm,
    gripper_norm
])
```

**原理**:
- RDT输出（RDT坐标系） → 翻转 → LIBERO格式
- 评估时将RDT输出转换回LIBERO坐标系

### 3. 数据集统计

**文件**: `configs/dataset_stat.json`

**已重新计算**: ✅
```
EEF位置统计（翻转后）:
  X: mean=-0.079743, std=0.116032, range=[-0.414, 0.180]
  Y: mean=0.026516, std=0.131745, range=[-0.267, 0.290]
  Z: mean=0.944110, std=0.235259, range=[0.474, 1.270]
```

---

## 🔄 完整的数据流

### 训练流程
```
LIBERO原始action (归一化[-1,1])
  ↓ × 0.012 (缩放到米)
物理单位（米）[LIBERO坐标系]
  ↓ × -1 (翻转所有轴)
物理单位（米）[RDT坐标系]
  ↓ 填充到128维向量
训练数据 (RDT格式)
  ↓ 模型训练
RDT模型（学习RDT坐标系）
```

### 评估流程
```
RDT模型输出 (RDT坐标系)
  ↓ 提取位置 [30-32]
物理单位（米）[RDT坐标系]
  ↓ ÷ 0.012 (转换回归一化)
归一化值 [RDT坐标系]
  ↓ × -1 (翻转回LIBERO坐标系)
归一化值 [LIBERO坐标系]
  ↓ 发送给LIBERO环境
机器人执行
```

---

## 🎯 预期效果

修改后，新训练的模型应该：

1. **方向正确**
   - 机器人向正确方向移动
   - 不会出现"左右上下前后都相反"的问题

2. **幅度合理**
   - 动作大小符合演示数据
   - 不会过大或过小

3. **成功率提升**
   - 从0%提升到20-50%
   - 能够完成简单任务

---

## 📋 训练步骤

### 1. 从头开始训练（推荐）

```bash
python libero_finetune_correct.py \
    --task_id 0 \
    --max_steps 15000 \
    --output_dir checkpoints/libero_fixed_coords \
    --cuda_device 0
```

**为什么从头训练**:
- 旧checkpoint是用错误的坐标系训练的
- 继续训练会混淆模型
- 从头训练能学到正确的映射

### 2. 快速验证（几百步即可）

```bash
# 训练500步后就可以测试
python libero_finetune_correct.py \
    --task_id 0 \
    --max_steps 500 \
    --checkpointing_period 500
```

### 3. 评估测试

```bash
python eval_sim/eval_rdt_libero.py \
    --pretrained checkpoints/libero_fixed_coords/task_00_xxx/checkpoint-500 \
    --num_tasks 1 \
    --max_steps 50 \
    --record_video
```

---

## 🔍 如何判断修复是否成功

### 成功的标志

1. **视频中机器人运动**:
   - "关闭抽屉"任务：向后拉（靠近机器人）✅
   - 动作幅度合理（不过大不过小）✅
   - 运动轨迹与演示相似 ✅

2. **评估指标**:
   - Success rate > 0% ✅
   - 能够完成至少部分任务 ✅

### 如果还不对

如果方向还是不对，可能需要：
1. 只翻转部分轴（如只翻转XY）
2. 检查旋转是否也需要翻转
3. 验证LIBERO环境的坐标系定义

---

## 📝 相关文件

### 修改的代码
- `data/hdf5_libero_dataset.py` - 训练数据加载器
- `eval_sim/eval_rdt_libero.py` - 评估代码
- `configs/dataset_stat.json` - 数据集统计（已更新）

### 分析文档
- `FINAL_COORDINATE_ANALYSIS.md` - 坐标系分析
- `COORDINATE_CHECK_SUMMARY.md` - 检查总结
- `check_coordinate_systems.py` - 检查脚本

### 工具脚本
- `compute_dataset_stat.py` - 计算统计信息
- `libero_finetune_correct.py` - 训练脚本

---

## ⚠️ 注意事项

### 1. 旧checkpoint不可用

**旧checkpoint**（未翻转坐标系）:
- ❌ 不能用于新的评估代码
- ❌ 不能继续训练
- 原因：坐标系不匹配

**解决方案**：从预训练模型重新开始训练

### 2. 训练时间

**快速验证**：500-1000步（~30分钟）
- 可以看出方向是否正确
- Loss是否正常下降

**完整训练**：15000-30000步（~2-4小时）
- 获得较好的性能
- Success rate 20-50%

### 3. 数据集统计

每次修改数据转换后，必须重新计算统计：
```bash
python compute_dataset_stat.py
```

---

## 🚀 下一步

1. **立即执行**:
   ```bash
   # 快速验证（500步）
   python libero_finetune_correct.py --task_id 0 --max_steps 500
   ```

2. **检查结果**:
   - 500步后评估
   - 观察视频
   - 确认方向正确

3. **继续训练**:
   - 如果方向正确，继续训练到15000步
   - 如果不对，尝试其他翻转配置

---

**状态**: ✅ 代码已修改，数据集统计已更新  
**下一步**: 开始训练并验证

