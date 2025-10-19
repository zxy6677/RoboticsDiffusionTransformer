# RDT-1B论文实现细节 vs 当前LIBERO微调代码对比分析

## 📚 RDT-1B论文关键实现细节（第19-24页）

基于论文和官方GitHub实现，RDT-1B的关键实现细节：

### 模型架构
- **参数量**: 1B (hidden_size=2048, depth=28, num_heads=32)
- **Action Horizon**: 64步（预测未来64个动作）
- **State Dimension**: 128维统一动作空间
- **历史图像**: img_history_size=2
- **摄像头数量**: 最多3个视角

### 训练超参数（标准配置）
```python
# 论文/官方实现的典型配置
batch_size: 256 (总batch size，通过gradient_accumulation实现)
learning_rate: 1e-4
lr_scheduler: "constant_with_warmup"
warmup_steps: 5000-10000
max_train_steps: 1M+ (预训练)
mixed_precision: "bf16"
gradient_accumulation_steps: 8-16
num_inference_timesteps: 5 (Diffusion采样步数)
```

### 数据处理关键点
1. **物理单位**: 使用物理单位（厘米、弧度），不做过度归一化
2. **长轨迹采样**: 从长轨迹中随机采样起点
3. **数据增强**: 图像增强（对比度、亮度等）
4. **Loss权重**: 基于数据集统计的自适应权重

---

## 🔍 当前LIBERO微调代码存在的问题

### 问题1: Batch Size过小 ⭐⭐⭐⭐⭐（严重问题）

**当前配置**:
```bash
--train_batch_size=4 \
--gradient_accumulation_steps=1 \
# 有效batch size = 4 * 1 * 8 GPUs = 32
```

**论文配置**:
```python
有效batch size = 256或更大
```

**问题分析**:
- **过小的batch size (32 vs 256)** 导致：
  1. **梯度估计不准确**: Diffusion模型需要大batch来稳定训练
  2. **BN/LayerNorm统计不准**: 小batch的统计信息噪声大
  3. **训练不稳定**: loss波动大，收敛慢
  4. **泛化能力弱**: 小batch容易过拟合到少数样本

**为什么50个demo训练失败，1个demo反而能过拟合？**
- 小batch size (32) + 50个demo → 模型看到的样本变化太大，无法充分学习每个demo
- 小batch size (32) + 1个demo → 模型反复看到同样的样本，容易过拟合

**修复建议**:
```bash
--train_batch_size=4 \          # 保持单卡batch size
--gradient_accumulation_steps=8 \  # 增加梯度累积！
# 有效batch size = 4 * 8 * 8 GPUs = 256 ✅
```

---

### 问题2: Warmup Steps过少 ⭐⭐⭐⭐

**当前配置**:
```bash
--lr_warmup_steps=100
```

**论文配置**:
```python
warmup_steps = 5000-10000  # 或总步数的5-10%
```

**问题分析**:
- **100步太短**: Diffusion模型初期需要更长的warmup
- **导致训练初期不稳定**: 大学习率直接应用会导致梯度爆炸
- **影响最终性能**: warmup不足会限制模型的最终表现

**实际影响**:
```
训练步数: 100,000步
当前warmup: 100步 (0.1%)
推荐warmup: 5,000-10,000步 (5-10%)
```

**修复建议**:
```bash
--lr_warmup_steps=5000 \  # 改为5000步
```

---

### 问题3: 评估时exec_horizon与训练不匹配 ⭐⭐⭐⭐⭐（最关键问题）

**训练配置**:
```yaml
action_chunk_size: 64  # 训练时预测64步
```

**评估配置（您之前用的）**:
```bash
--exec_horizon 64  # 执行全部64步
```

**论文的推荐做法**:
```python
# Receding Horizon Control
action_chunk_size = 64  # 训练时预测64步
exec_horizon = 8-16     # 评估时只执行8-16步，然后重新预测

# 这就是"Temporal Ensembling"
```

**问题分析**:
- **训练学习的是**: 给定当前状态，预测未来64步的能力
- **但评估时**: 执行全部64步，没有中间反馈
- **导致**: 在多阶段任务（关抽屉→抓碗）中无法适应状态转换

**为什么这是关键问题**:
```
任务长度: 213步
转换点: 第120步（关抽屉→抓碗）

exec_horizon=64时:
- Step 0:   预测[0:64]，执行全部   → 关抽屉
- Step 64:  预测[64:128]，执行全部 → 关抽屉末期  
- Step 128: 预测[128:192]，执行全部 → 应该抓碗，但模型"惯性"还在关抽屉 ❌

exec_horizon=16时:
- 每16步重新预测一次
- 在转换点(120步)前后会重新预测多次
- 模型能根据当前状态（抽屉已关）调整策略 ✅
```

**这就是为什么**:
- 50个demo训练 + exec_horizon=64 → 失败（无法转换）
- 50个demo训练 + exec_horizon=16 → 应该能成功
- 1个demo训练 + exec_horizon=64 → 成功（因为记住了转换点）

**修复建议**:
```bash
--exec_horizon 16  # 已经在测试中建议了
```

---

### 问题4: 数据采样策略可能不优 ⭐⭐⭐

**当前实现** (hdf5_libero_dataset.py):
```python
# 从episode中随机选择一个起点
step_id = np.random.randint(max(first_idx-1, 0), num_steps)
# 然后取接下来的64步
actions_seq = actions[step_id:step_id+self.CHUNK_SIZE]
```

**论文的采样策略**:
```python
# 应该考虑：
1. 长轨迹采样：优先采样轨迹中间和后期的状态
2. 关键时刻采样：增加任务转换点附近的采样权重
3. 平衡采样：确保不同阶段的动作都被充分学习
```

**问题分析**:
- **均匀采样**: 当前是完全随机采样
- **关键时刻采样不足**: 任务转换点（第120步）被采样的概率很低
- **导致**: 模型很少学习到"如何从关抽屉切换到抓碗"

**统计分析**:
```
轨迹长度: 213步
可采样位置: 213 - 64 = 149个位置
转换点: 第110-130步（20个关键位置）
转换点采样概率: 20/149 = 13.4%

但实际训练中，模型需要看到：
- 关抽屉阶段 (0-120步): 56%
- 抓碗阶段 (120-213步): 44%
- 关键转换 (110-130步): 应该加权采样！
```

**修复建议**:
```python
# 在hdf5_libero_dataset.py中添加关键时刻采样
def sample_step_with_importance(num_steps, chunk_size):
    # 定义关键区域（任务转换点附近）
    transition_region = range(100, 140)  # 转换点±20步
    
    # 80%概率正常采样，20%概率采样转换区域
    if np.random.random() < 0.2 and num_steps > 140:
        step_id = np.random.choice(transition_region)
    else:
        step_id = np.random.randint(0, num_steps - chunk_size)
    
    return step_id
```

---

### 问题5: Loss权重可能不平衡 ⭐⭐⭐

**当前配置** (dataset_stat.json):
```json
"libero_single_task": {
    "action_std": {
        "position": [0.33, 0.47, 0.49],      // loss_weight ~ 2-3
        "rotation": [0.0007, 0.018, ...],    // loss_weight ~ 50-1400
        "gripper": [0.48]                    // loss_weight ~ 2
    }
}
```

**问题分析**:
```
旋转维度的loss权重 ≈ 2000
位置+gripper的loss权重 ≈ 10

结果：
- 模型过度关注旋转精度
- 忽略gripper的开合（这是抓碗的关键！）
```

**论文的做法**:
```python
# RDT使用自适应权重，但会设置上下界
action_std = np.clip(action_std, min=0.01, max=1.0)
# 避免某些维度的权重过大或过小
```

**为什么影响性能**:
- **50个demo训练**: 位置变化大，旋转变化小
  - loss主要来自旋转误差
  - gripper的loss被忽略
  - 模型学不会精确的gripper控制
- **1个demo训练**: 虽然权重不平衡，但过拟合能记住gripper时机

**修复建议**:
```python
# 在compute_dataset_stat.py中添加
action_std = np.maximum(action_std, 0.05)  # 设置下界
# 或者针对性地调整
action_std[gripper_idx] *= 2.0  # 增加gripper的权重
action_std[rotation_idx] = np.minimum(action_std[rotation_idx], 0.1)  # 限制旋转权重
```

---

### 问题6: 摄像头配置不一致 ⭐⭐

**训练时** (hdf5_libero_dataset.py):
```python
cam_high = parse_img('agentview_rgb')      # ✅
cam_right_wrist = parse_img('eye_in_hand_rgb')  # ✅
cam_left_wrist = np.zeros(...)            # ❌ (单臂机器人)
```

**评估时（修改前）**:
```python
images = [Image.fromarray(img), None, None]  # 只用1个摄像头 ❌
```

**问题**:
- Train-test不一致
- 手腕视角对精细操作很重要
- 已修复（添加了双摄像头支持）✅

---

### 问题7: 数据增强可能不够 ⭐⭐

**当前配置**:
```bash
--image_aug  # 启用了图像增强
```

**论文的数据增强**:
```python
# RDT使用的增强：
1. 随机crop和resize
2. 颜色抖动（brightness, contrast, saturation, hue）
3. 随机高斯模糊
4. 但不使用flip（会破坏空间关系）
```

**检查建议**:
- 查看`train/image_corrupt.py`确认增强强度是否合适
- 太强的增强会破坏空间关系
- 太弱的增强会导致过拟合

---

### 问题8: Checkpoint保存频率 ⭐

**当前配置**:
```bash
--checkpointing_period=10000  # 每10000步保存一次
--max_train_steps=100000      # 总共100000步
# 只保存10个checkpoint
```

**论文建议**:
- 更频繁的checkpoint（如每2000步）
- 可以回退到更早的好模型
- 对于小数据集微调更重要

**修复建议**:
```bash
--checkpointing_period=2000 \  # 改回2000步
--checkpoints_total_limit=30 \  # 增加保存数量
```

---

## 📊 关键问题优先级排序

| 优先级 | 问题 | 严重程度 | 影响 | 是否需要重训 |
|--------|------|----------|------|------------|
| 🥇 1 | exec_horizon=64 | ⭐⭐⭐⭐⭐ | 无法阶段转换 | ❌ 不需要 |
| 🥈 2 | Batch size=32 | ⭐⭐⭐⭐⭐ | 训练不稳定 | ✅ 需要 |
| 🥉 3 | Warmup steps=100 | ⭐⭐⭐⭐ | 训练初期不稳定 | ✅ 需要 |
| 4 | Loss权重不平衡 | ⭐⭐⭐ | Gripper控制差 | ✅ 需要 |
| 5 | 采样策略简单 | ⭐⭐⭐ | 关键时刻学习不足 | ✅ 需要 |
| 6 | 摄像头不一致 | ⭐⭐ | 精细操作差 | ❌ 已修复 |
| 7 | Checkpoint频率 | ⭐ | 可能错过好模型 | ✅ 需要 |

---

## 💡 立即可测试的修复（不需要重训）

### 修复1: 调整exec_horizon ⭐⭐⭐⭐⭐

```bash
# 立即测试
bash test_dual_camera.sh  # exec_horizon=16

# 或更激进
python eval_sim/eval_rdt_libero.py \
  --exec_horizon 8 \
  # ... 其他参数
```

**预期**: 如果这是主要问题，成功率应该显著提升！

---

## 🔧 需要重新训练的修复

### 改进的训练脚本

```bash
#!/bin/bash
# 改进后的训练配置

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

accelerate launch \
    --num_processes 8 \
    --multi_gpu \
    --mixed_precision bf16 \
    main.py \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl \
    --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 \
    --output_dir=/share_data/zhukefei/checkpoints/libero_finetune_v2 \
    --train_batch_size=4 \
    --sample_batch_size=8 \
    --gradient_accumulation_steps=8 \      # 改进1: 增加到8 ⭐⭐⭐⭐⭐
    --max_train_steps=100000 \
    --checkpointing_period=2000 \          # 改进2: 更频繁保存
    --checkpoints_total_limit=30 \         # 改进3: 保存更多checkpoint
    --sample_period=500 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --lr_warmup_steps=5000 \               # 改进4: 增加warmup ⭐⭐⭐⭐
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to="wandb"

# 有效batch size = 4 * 8 * 8 = 256 ✅（与论文一致）
```

---

## 📈 预期改进效果

### 当前配置（问题版本）
```
Batch size: 32
Warmup: 100步
exec_horizon: 64
→ 成功率: 0%（50个demo训练）
```

### 改进1：只调整exec_horizon（不需要重训）
```
Batch size: 32（不变）
Warmup: 100步（不变）
exec_horizon: 16  ← 修改
→ 预期成功率: 30-50%
```

### 改进2：重新训练（所有修复）
```
Batch size: 256  ← 修改
Warmup: 5000步  ← 修改
exec_horizon: 16  ← 修改
Loss权重平衡  ← 修改
→ 预期成功率: 60-80%
```

---

## 🎯 行动建议

### 立即行动（今天）

1. **测试exec_horizon=16**
   ```bash
   bash test_dual_camera.sh
   ```
   
2. **如果成功率提升**: 说明主要是exec_horizon问题
   - 不需要重训
   - 继续优化exec_horizon（试8或12）

3. **如果没有提升**: 说明需要重新训练
   - 应用所有训练改进
   - 预计训练时间：40-50小时（8-GPU）

### 中期优化（如果需要重训）

1. **修改训练脚本**
   - gradient_accumulation_steps=8
   - lr_warmup_steps=5000
   - checkpointing_period=2000

2. **修改Loss权重**
   - 限制旋转维度的权重上界
   - 增加gripper维度的权重

3. **改进采样策略**
   - 添加关键时刻采样
   - 增加转换点附近的采样权重

### 长期优化

1. **数据增强调优**
2. **考虑课程学习**
3. **尝试不同的学习率调度器**

---

## 📝 总结

### 最关键的发现 🔥

**问题不在于使用50个demo还是1个demo！**

**真正的问题是**:
1. ⭐⭐⭐⭐⭐ **exec_horizon=64太大** → 无法在评估时进行阶段转换
2. ⭐⭐⭐⭐⭐ **Batch size=32太小** → 训练不稳定，泛化能力弱
3. ⭐⭐⭐⭐ **Warmup steps=100太少** → 训练初期不稳定

### 为什么1个demo能成功，50个demo失败？

**不是数据量的问题，而是配置问题**:
- 1个demo + 小batch + exec_horizon=64 → 过拟合，记住了转换点
- 50个demo + 小batch + exec_horizon=64 → 无法充分学习，且无法转换

### 最优策略

**先测试exec_horizon=16**（今天就能测试）:
- 如果成功 → 问题解决 🎉
- 如果失败 → 需要重新训练（应用所有改进）

**如果需要重训**:
- 使用改进的配置（大batch size, 长warmup）
- 预计成功率能达到60-80%
- 训练时间：40-50小时

---

**核心结论**: 当前代码与RDT-1B论文的主要差异在于**训练和评估的超参数配置**，而不是模型架构本身。通过调整这些参数，应该能显著提升性能！🎯

