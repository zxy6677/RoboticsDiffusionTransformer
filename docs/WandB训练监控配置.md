# Weights & Biases 训练监控配置 📊

## 🎯 为什么使用 WandB？

相比TensorBoard，WandB提供：
- ✅ 云端自动同步，随时随地查看
- ✅ 更美观的可视化界面
- ✅ 实验对比功能
- ✅ 自动保存最佳模型
- ✅ 移动端APP支持

---

## 🔧 方案1: 在线模式（推荐）

### 1. 安装并登录

在远程服务器上：

```bash
# 激活环境
conda activate rdt

# 安装wandb
pip install wandb

# 登录（需要API key）
wandb login
```

获取API Key：
1. 访问 https://wandb.ai/authorize
2. 复制API key
3. 粘贴到终端

### 2. 配置项目名称（可选）

在训练脚本中添加环境变量：

```bash
# 在 train_single_task_2gpu.sh 开头添加
export WANDB_PROJECT="libero-single-task-finetune"
export WANDB_NAME="scene10-8gpu-50demos"
export WANDB_NOTES="使用50个demo训练，修复采样bug后"
```

### 3. 重新开始训练

```bash
bash train_single_task_2gpu.sh
```

### 4. 查看结果

访问：https://wandb.ai/your-username/libero-single-task-finetune

实时查看：
- Loss曲线
- Learning rate
- GPU使用率
- 训练速度
- 样本可视化

---

## 🔧 方案2: 离线模式

如果网络不好或不想上传数据：

```bash
# 设置离线模式
export WANDB_MODE=offline

# 运行训练
bash train_single_task_2gpu.sh

# 训练完成后，可以手动同步
wandb sync wandb/offline-run-xxx
```

---

## 🔧 方案3: 禁用追踪

如果不需要任何监控：

修改训练脚本，将：
```bash
--report_to="wandb"
```

改为：
```bash
--report_to="none"
```

或者设置环境变量：
```bash
export WANDB_DISABLED=true
```

---

## 📊 推荐的WandB配置

### 完整的环境变量设置

在 `train_single_task_2gpu.sh` 开头添加：

```bash
# ============================================
# WandB配置
# ============================================
export WANDB_PROJECT="libero-single-task-finetune"
export WANDB_NAME="scene10-8gpu-batch32-lr1e-4"
export WANDB_NOTES="8张A800, 50个demo, 修复采样bug"
export WANDB_TAGS="libero,single-task,8gpu"

# 可选：离线模式
# export WANDB_MODE=offline

# 可选：禁用wandb
# export WANDB_DISABLED=true
```

---

## 📈 WandB监控的关键指标

训练时会自动记录：

### 损失和优化
- `train/loss` - 训练损失
- `train/lr` - 学习率
- `train/epoch` - 当前epoch

### 性能指标
- `train/steps_per_sec` - 训练速度
- `train/samples_per_sec` - 样本处理速度
- `system/gpu.X.memory` - GPU显存使用

### 采样指标（如果启用sample_period）
- `eval/sample_error` - 采样误差
- 可视化轨迹对比

---

## 🔍 当前训练的WandB配置

### 已更新的配置

```bash
# train_single_task_2gpu.sh 中已添加
--report_to="wandb"  ✅
```

### 下次训练前需要做的

**选项A: 在线模式**
```bash
# 1. SSH到远程服务器
ssh -J zhukefei@134.175.121.223 zhukefei@172.16.0.27

# 2. 激活环境
conda activate rdt

# 3. 安装wandb
pip install wandb

# 4. 登录
wandb login  # 粘贴API key

# 5. pull最新代码
cd ~/RoboticsDiffusionTransformer
git pull

# 6. 停止当前训练（如果在运行）
# 在tmux中按Ctrl+C

# 7. 重新开始训练
bash train_single_task_2gpu.sh
```

**选项B: 离线模式（无需登录）**
```bash
# 1-5步同上

# 6. 设置离线模式
export WANDB_MODE=offline

# 7. 开始训练
bash train_single_task_2gpu.sh
```

**选项C: 继续当前训练（不使用wandb）**

当前训练可以继续，只是没有漂亮的可视化界面。训练完成后再配置wandb用于下次训练。

---

## 💡 小技巧

### 1. 给实验打标签

```bash
export WANDB_TAGS="experiment-v1,bugfix,50demos"
```

### 2. 保存配置到wandb

wandb会自动保存所有命令行参数，方便复现。

### 3. 对比多次训练

在wandb界面可以轻松对比：
- 修复前 vs 修复后
- 不同学习率
- 不同batch size

### 4. 使用wandb API

```python
import wandb

# 查询最佳模型
api = wandb.Api()
runs = api.runs("your-username/libero-single-task-finetune")
best_run = min(runs, key=lambda run: run.summary.get("train/loss", float('inf')))
print(f"Best run: {best_run.name}, loss: {best_run.summary['train/loss']}")
```

---

## 🆚 TensorBoard vs WandB

| 功能 | TensorBoard | WandB |
|------|-------------|-------|
| 云端同步 | ❌ 需要手动上传 | ✅ 自动同步 |
| 移动查看 | ❌ | ✅ |
| 实验对比 | ⚠️ 麻烦 | ✅ 简单 |
| 模型版本管理 | ❌ | ✅ |
| 团队协作 | ⚠️ 需要配置 | ✅ 内置 |
| 安装 | 简单 | 简单 |
| 网络要求 | 无（本地） | 需要（可离线） |

---

## 📝 总结

### 当前状态
- ✅ 训练脚本已更新（添加 `--report_to="wandb"`）
- ⚠️ 远程服务器需要安装wandb
- 🔄 当前训练可以继续（使用默认的tensorboard，虽然没安装）

### 推荐操作
1. 让当前训练继续（已经运行5.21秒/步，速度不错）
2. 下次训练前安装并配置wandb
3. 或者如果急需可视化，可以：
   - 停止当前训练
   - 安装wandb并登录
   - 重新开始训练

### 快速决策

**如果训练刚开始（<100步）**: 建议现在配置wandb
**如果训练已进行较久（>1000步）**: 让它继续，下次训练再用wandb

---

**文档更新**: 2024-10-19

