# 快速行动指南：修复LIBERO训练问题

## 🎯 问题诊断结果

基于RDT-1B论文实现细节的对比分析，发现了**3个关键问题**：

### 🥇 问题1: exec_horizon=64（最关键，不需要重训）
- **影响**: 无法在多阶段任务中重新规划
- **现象**: 能关抽屉，但无法切换到抓碗
- **修复**: 评估时使用exec_horizon=16

### 🥈 问题2: Batch Size=32（严重，需要重训）
- **影响**: 训练不稳定，泛化能力弱
- **现象**: 50个demo失败，1个demo能过拟合
- **修复**: gradient_accumulation_steps=8 → 有效batch size=256

### 🥉 问题3: Warmup=100步（重要，需要重训）
- **影响**: 训练初期不稳定，影响最终性能
- **修复**: lr_warmup_steps=5000

---

## ⚡ 立即行动（5分钟，不需要重训）

### 测试方案1: 使用test_dual_camera.sh

```bash
cd /home/ubuntu/RoboticsDiffusionTransformer
export CUDA_VISIBLE_DEVICES=1
bash test_dual_camera.sh
```

这个脚本已经配置了：
- ✅ exec_horizon=16
- ✅ 双摄像头支持
- ✅ 录制视频

### 测试方案2: 手动运行评估

```bash
export CUDA_VISIBLE_DEVICES=1
python eval_sim/eval_rdt_libero.py \
  --config configs/base.yaml \
  --pretrained checkpoints/single_task_scene7/checkpoint-19000/ema/model.safetensors \
  --text_encoder google/t5-v1_1-xxl \
  --vision_encoder google/siglip-so400m-patch14-384 \
  --benchmark libero_90 \
  --num_tasks 2 \
  --max_steps 200 \
  --exec_horizon 16 \
  --record_video \
  --video_output_dir videos/test_exec16_$(date +%Y%m%d_%H%M%S)
```

### 预期结果判断

**如果成功率 > 30%**:
- ✅ exec_horizon是主要问题
- ✅ 可能不需要重新训练
- 🎯 继续优化exec_horizon（试8, 12, 20等）

**如果成功率 < 30%**:
- ⚠️ batch size也是严重问题
- 📦 需要使用改进配置重新训练
- ⏱️ 预计训练时间：40-50小时（8-GPU）

---

## 🔄 重新训练（如果需要）

### 方案A: 本地训练

```bash
cd /home/ubuntu/RoboticsDiffusionTransformer

# 检查GPU可用性
nvidia-smi

# 启动改进的训练
bash train_single_task_improved.sh
```

**训练时间估算**:
```
单GPU: ~400小时
2-GPU: ~200小时
8-GPU: ~40-50小时 ✅ 推荐
```

### 方案B: 远程服务器训练（推荐）

#### 步骤1: 上传改进的训练脚本

```bash
# 在本地执行
scp train_single_task_improved.sh pro-20:/home/zhukefei/RoboticsDiffusionTransformer/
```

#### 步骤2: 登录服务器并修改配置

```bash
# 登录服务器
ssh pro-20

# 进入项目目录
cd /home/zhukefei/RoboticsDiffusionTransformer

# 修改输出路径为服务器路径
sed -i 's|./checkpoints/single_task_scene10_improved|/share_data/zhukefei/checkpoints/libero_improved|g' train_single_task_improved.sh

# 确认修改
grep "OUTPUT_DIR=" train_single_task_improved.sh
```

#### 步骤3: 启动训练

```bash
# 创建tmux会话（防止断线）
tmux new -s train_improved

# 激活conda环境
conda activate rdt

# 启动训练
bash train_single_task_improved.sh

# 分离tmux会话（Ctrl+B 然后按 D）
# 重新连接: tmux attach -t train_improved
```

#### 步骤4: 监控训练

```bash
# 方法1: WandB在线监控
# 访问 https://wandb.ai 查看训练曲线

# 方法2: 查看本地日志
tail -f nohup.out  # 如果使用nohup
# 或直接在tmux中查看

# 方法3: 检查checkpoint
ls -lht /share_data/zhukefei/checkpoints/libero_improved/
```

---

## 📊 关键配置对比

| 配置项 | 当前训练 | 改进训练 | 差异 |
|--------|---------|---------|------|
| 有效Batch Size | 32 | 256 | **8倍** ⭐⭐⭐⭐⭐ |
| Warmup Steps | 100 | 5000 | **50倍** ⭐⭐⭐⭐ |
| Gradient Accumulation | 1 | 8 | **8倍** |
| Exec Horizon (评估) | 64 | 16 | **4倍少** ⭐⭐⭐⭐⭐ |
| Checkpoint Period | 10000 | 2000 | **5倍频繁** |

---

## 🎓 为什么这些改进很重要？

### 1. 大Batch Size (32→256)

**Diffusion模型的特殊需求**:
```python
# Diffusion训练需要：
1. 稳定的梯度估计（需要大batch）
2. 准确的BN/LayerNorm统计（需要大batch）
3. 多样化的噪声采样（需要大batch）

# 小batch的问题：
- 梯度估计噪声大 → 训练不稳定
- 统计信息不准确 → 归一化效果差
- 样本多样性不足 → 泛化能力弱
```

**为什么50个demo失败，1个demo成功？**
```
1个demo (213步):
- 数据量小，batch=32已经足够
- 能够充分记忆整个序列
- 过拟合到转换点

50个demo (10650步):
- 数据量大，batch=32太小
- 梯度估计不稳定
- 无法充分学习所有变化
- 特别是转换点（占10%）学习不足
```

### 2. 更长的Warmup (100→5000步)

**Diffusion模型需要温和启动**:
```python
# Warmup的作用：
1. 避免大学习率导致的训练不稳定
2. 让模型逐步适应数据分布
3. 提高最终收敛质量

# 100步太短的问题：
- 占总训练的0.1%（应该是5-10%）
- 模型还没适应就全速训练
- 影响最终性能上限
```

### 3. 更小的Exec Horizon (64→16)

**Receding Horizon Control原理**:
```python
# 训练时：
action_chunk_size = 64  # 学习长期规划（64步）

# 评估时：
exec_horizon = 16  # 短期执行+重新规划
- 执行16步
- 获取新观察
- 重新预测64步
- 再执行16步
- ...

# 为什么exec_horizon=64失败？
任务转换点在第120步（关抽屉→抓碗）
- Step 0:   预测[0:64]，全执行   → 关抽屉
- Step 64:  预测[64:128]，全执行 → 关抽屉末期
- Step 128: 预测[128:192]，全执行 → 应该抓碗
  但模型在Step 64预测时，抽屉还没完全关
  预测的[128:192]还是关抽屉的动作
  到Step 128执行时，已经晚了！❌

# 为什么exec_horizon=16成功？
- Step 112: 预测[112:176]，执行16步 → 关抽屉完成
- Step 128: 重新预测[128:192] → 此时抽屉已关
  模型根据最新状态，预测抓碗动作 ✅
```

---

## 📈 预期性能提升时间线

### 今天（5分钟）
```bash
# 测试exec_horizon=16
bash test_dual_camera.sh

预期成功率: 30-50%（如果exec_horizon是主要问题）
```

### 明天-后天（如果需要重训）
```bash
# 启动改进配置训练
bash train_single_task_improved.sh

训练进度:
- 10小时:  checkpoint-2000  (可以初步测试)
- 20小时:  checkpoint-10000 (应该看到改进)
- 40小时:  checkpoint-50000 (接近最优)
- 50小时:  checkpoint-100000 (完成)
```

### 3天后（评估新模型）
```bash
# 评估改进的模型
python eval_sim/eval_rdt_libero.py \
  --pretrained /path/to/checkpoint-50000/ema/model.safetensors \
  --exec_horizon 16 \
  # ... 其他参数

预期成功率: 60-80%
```

---

## 🛠️ 故障排除

### 问题1: GPU内存不足

**症状**: `CUDA out of memory`

**解决**:
```bash
# 减小单卡batch size
--train_batch_size=2 \  # 从4改为2
--gradient_accumulation_steps=16 \  # 从8改为16
# 有效batch size仍然是256
```

### 问题2: 训练速度慢

**检查**:
```bash
# 确认使用了混合精度
--mixed_precision="bf16"

# 确认数据加载不是瓶颈
--dataloader_num_workers=8

# 确认GPU利用率
nvidia-smi -l 1  # 应该>90%
```

### 问题3: WandB登录失败

**解决**:
```bash
# 方法1: 离线模式
export WANDB_MODE=offline

# 方法2: 禁用WandB
--report_to="none"  # 在train_single_task_improved.sh中修改

# 方法3: 重新登录
wandb login
```

### 问题4: 训练loss不下降

**检查清单**:
```bash
1. 确认预训练模型加载成功
   grep "loading weights from" log.txt

2. 确认学习率正确
   --learning_rate=1e-4  # 不要太大或太小

3. 确认warmup生效
   --lr_warmup_steps=5000

4. 确认数据加载正确
   # 应该看到 "检测到单任务训练，使用 libero_single_task 统计信息"

5. 查看WandB曲线
   # loss应该在前5000步逐渐下降
```

---

## 📚 参考文档

生成的分析文档：
1. `CODE_COMPARISON_WITH_RDT_PAPER.md` - 详细的代码对比分析
2. `CONFIGURATION_COMPARISON.md` - 配置参数对比表
3. `SPEED_ANALYSIS.md` - 评估速度分析
4. `train_single_task_improved.sh` - 改进的训练脚本

RDT-1B相关资源：
- 论文: https://arxiv.org/pdf/2410.07864
- GitHub: https://github.com/thu-ml/RoboticsDiffusionTransformer
- 模型: https://huggingface.co/robotics-diffusion-transformer/rdt-1b

---

## ✅ 检查清单

### 立即测试（今天）
- [ ] 运行 `bash test_dual_camera.sh`
- [ ] 查看成功率
- [ ] 如果>30%，不需要重训 🎉
- [ ] 如果<30%，准备重训 📦

### 如果需要重训
- [ ] 上传 `train_single_task_improved.sh` 到服务器
- [ ] 修改输出路径为服务器路径
- [ ] 在tmux中启动训练
- [ ] 配置WandB监控
- [ ] 每10小时检查一次checkpoint

### 评估新模型
- [ ] 使用 exec_horizon=16 评估
- [ ] 记录成功率
- [ ] 如果需要，尝试不同的exec_horizon值（8, 12, 20）

---

## 🎯 预期最终结果

**最佳情况**（exec_horizon是主要问题）:
```
- 修改评估参数 → 成功率 30-50%
- 无需重新训练
- 今天就能看到结果
```

**良好情况**（需要重训）:
```
- 使用改进配置训练 2-3天
- 成功率 60-80%
- 达到RDT-1B论文水平
```

**保守情况**（还需进一步优化）:
```
- 改进配置 + loss权重调整 + 采样策略
- 成功率 50-60%
- 需要更多迭代优化
```

---

## 💡 核心结论

**您的代码实现是正确的！** 

**问题在于训练和评估的超参数配置与RDT-1B论文有显著差异。**

**3个关键修复**:
1. ⭐⭐⭐⭐⭐ exec_horizon: 64→16（今天就能测试）
2. ⭐⭐⭐⭐⭐ Batch size: 32→256（需要重训）
3. ⭐⭐⭐⭐ Warmup: 100→5000步（需要重训）

**行动顺序**:
1. 今天测试exec_horizon=16
2. 根据结果决定是否重训
3. 如需重训，使用改进配置

**预期时间**:
- 测试: 5分钟
- 重训: 2-3天（如果需要）
- 达到60-80%成功率

---

**开始行动吧！** 🚀

```bash
cd /home/ubuntu/RoboticsDiffusionTransformer
export CUDA_VISIBLE_DEVICES=1
bash test_dual_camera.sh
```

