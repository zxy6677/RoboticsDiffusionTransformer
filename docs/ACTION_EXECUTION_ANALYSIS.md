# RDT Action执行策略分析

## 🔍 当前问题

**训练时**：模型学习预测64个连续的actions  
**评估时**：只执行其中8个actions（每8步取1个）

**结果**：模型预测的轨迹和实际执行的轨迹不匹配！

## 📊 数据分析

### LIBERO训练数据
```
Demo总步数: 71
CHUNK_SIZE: 64
可生成样本数: 8个

样本结构：
  - 每个样本包含64个连续的actions
  - Position在64步内变化约1厘米
  - 最后的样本被padding（因为demo只有71步）
```

### 训练样本示例
```
样本1的position变化:
  第1步:  [0.627, 0.440, -0.604] 厘米
  第8步:  [0.537, -0.006, -0.736] 厘米  
  第16步: [0.186, -0.768, -0.527] 厘米
  第64步: [0.000, 0.000, 0.000] 厘米 (padding)

总位移: 0.98厘米
```

## ⚠️ 当前评估策略的问题

### 评估代码
```python
# eval_rdt_libero.py L556
actions_np = rdt_actions.squeeze(0).cpu().numpy()  # (64, 128)
actions_to_execute = actions_np[::8]  # 每8个取1个，得到8个action
```

### 问题分析

**训练-评估不匹配**：
- **训练时**：模型学的是"给定当前状态，预测接下来64步的连续actions"
- **评估时**：只执行其中8个actions（第0, 8, 16, 24, 32, 40, 48, 56步）

**后果**：
1. **轨迹不连续**：跳过了中间的actions，导致运动不平滑
2. **时间尺度错误**：64步本应该3.2秒完成，现在只执行8步→0.4秒
3. **状态反馈错误**：模型下一次预测时的state已经偏离了预期轨迹

## 🔬 Maniskill的实现

### Maniskill代码
```python
# eval_rdt_maniskill.py L119
actions = actions[::4, :]  # 每4个取1个，得到16个action
```

**注释说明**：
> "Take 8 steps since RDT is trained to predict **interpolated 64 steps(actual 14 steps)**"

### Maniskill的假设
- 预训练数据可能使用了**temporal interpolation**
- 64步是从实际14步插值得到的
- 因此评估时需要"反插值"：每4步取1个

### LIBERO的情况
- LIBERO数据**没有插值**
- 64步就是原始的64个连续actions
- **不应该跳过任何action！**

## ✅ 正确的评估策略

### 方案1：执行所有64个actions（推荐）
```python
actions_np = rdt_actions.squeeze(0).cpu().numpy()  # (64, 128)
actions_to_execute = actions_np  # 执行全部64个action
```

**理由**：
- ✅ 与训练完全一致
- ✅ 保持轨迹连续性
- ✅ 正确的时间尺度

**潜在问题**：
- 计算量大（每次推理后执行64步）
- 如果环境状态漂移，后续actions可能不准确

### 方案2：Rolling window策略
```python
# 执行前N个actions（例如前8个），然后重新推理
actions_np = rdt_actions.squeeze(0).cpu().numpy()
actions_to_execute = actions_np[:8]  # 只执行前8个

# 执行完后，立即用新的state重新推理
```

**理由**：
- ✅ 减少状态漂移的影响
- ✅ 更频繁的视觉反馈
- ✅ 类似MPC (Model Predictive Control)

### 方案3：根据control frequency调整
```python
# 如果预训练数据的频率不同，可能需要调整
# 例如：预训练50Hz，LIBERO 20Hz
# 那么需要下采样：每2.5步取1个

ratio = pretrain_freq / libero_freq  # 50 / 20 = 2.5
actions_to_execute = actions_np[::int(ratio)]
```

## 🎯 推荐方案

**对于单任务过拟合测试**：使用**方案1（执行全部64个actions）**

**理由**：
1. 当前是单任务overfitting，应该能完美记忆训练数据
2. 如果连这个都失败，说明训练/评估还有其他不一致
3. 执行全部actions能最准确地验证模型学到了什么

**实施步骤**：
```python
# 修改 eval_rdt_libero.py L556
actions_np = rdt_actions.squeeze(0).cpu().numpy()  # (64, 128)
# actions_to_execute = actions_np[::8]  # 旧代码：每8个取1个
actions_to_execute = actions_np  # 新代码：执行全部
```

## 📝 其他可能的问题

### 1. 训练数据的padding
- 最后的样本被padding到64步
- Padding用的是最后一个action的重复
- 模型可能学到了"最后总是不动"的pattern

**验证方法**：
```python
# 检查训练样本的最后几步
sample = dataset.get_item(index=0)
print(sample['actions'][-10:, 30:33])  # 最后10步的position
```

### 2. State和Action的对应关系
- 训练时：`state[t]` → 预测 `actions[t:t+64]`
- 评估时：当前`state` → 预测 `actions`，但执行后state已经变了

**需要验证**：
- State转换是否正确？
- State是否包含足够的信息来预测长期actions？

### 3. 初始状态不匹配
- 训练数据的初始状态分布
- 评估时的初始状态
- 可能需要让评估初始状态更接近训练数据

## 🔧 立即行动

1. **修改评估代码**：执行全部64个actions
2. **重新评估**：看看轨迹是否接近demo
3. **录制视频**：对比demo和模型执行的轨迹
4. **如果仍然失败**：深入检查state转换和数据对齐






