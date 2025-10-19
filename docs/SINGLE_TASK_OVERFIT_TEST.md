# 单任务过拟合测试指南

## 目的

通过在单个任务上过拟合模型，验证训练和评估代码的正确性。如果模型能够在单任务上达到100%成功率，说明：
1. ✅ 训练代码正确（能够学习）
2. ✅ 评估代码正确（输出方向正确）
3. ✅ 数据处理正确（坐标系、归一化等）

## 已完成的配置

### 1. 数据集准备
- 📁 数据目录: `data/datasets/libero_single_task/`
- 📄 任务文件: `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5`
- 📊 任务描述: 关闭柜子顶部抽屉（最简单的任务）

### 2. 配置文件
- `configs/finetune_datasets_single_task.json` - 数据集列表
- `configs/finetune_sample_weights_single_task.json` - 采样权重
- `configs/dataset_control_freq.json` - 已添加 `libero_single_task: 20`

### 3. 代码修改
- `data/hdf5_libero_dataset.py` - 支持通过`dataset_name`参数指定数据集目录

### 4. 评估任务对应
- ✅ 评估的第一个任务(task_idx=0)是: `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet`
- ✅ 训练数据集中的任务: `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5`
- ✅ **完全匹配！**

## 使用方法

### 步骤1: 开始训练

```bash
conda activate rdt
./train_single_task.sh
```

训练参数说明：
- `batch_size=8` - 适合单GPU
- `num_train_epochs=100` - 足够过拟合
- `learning_rate=1e-4` - 标准学习率
- `save_steps=500` - 每500步保存一次
- `logging_steps=50` - 每50步打印一次

### 步骤2: 监控训练

观察loss曲线：
- 如果loss持续下降 → 模型正在学习 ✅
- 如果loss不下降 → 训练代码可能有问题 ❌

### 步骤3: 评估模型

选择一个checkpoint（例如checkpoint-2000），运行评估：

```bash
python eval_sim/eval_rdt_libero.py \
    --config configs/base.yaml \
    --pretrained checkpoints/single_task_overfit/checkpoint-2000/checkpoint-2000 \
    --text_encoder google/t5-v1_1-xxl \
    --vision_encoder google/siglip-so400m-patch14-384 \
    --benchmark libero_90 \
    --num_tasks 1 \
    --max_steps 100 \
    --record_video \
    --video_output_dir videos/single_task_test
```

### 步骤4: 分析结果

#### 情况A: 成功率 > 80%
- ✅ 训练代码正确
- ✅ 评估代码正确
- ✅ 数据处理正确
- **结论**: 多任务训练效果不好可能是：
  - 训练步数不够
  - 数据量不足
  - 任务太难
  - 需要更多训练时间

#### 情况B: 成功率 < 20%，但动作方向正确
- ✅ 训练代码正确
- ✅ 评估代码正确
- ⚠️ 模型还没学好
- **解决方案**: 继续训练更多步数

#### 情况C: 动作方向完全相反
- ❌ 评估代码有问题
- **需要检查**: 
  - State归一化
  - 坐标系转换
  - Action转换逻辑

#### 情况D: 模型输出不变/随机
- ❌ 训练代码有问题
- **需要检查**:
  - 梯度是否正常
  - Loss计算是否正确
  - 数据加载是否正确

## 预期时间线

- **2000步**: 应该看到loss明显下降
- **5000步**: 应该能看到一些正确的动作
- **10000步**: 成功率应该 > 50%
- **20000步**: 成功率应该 > 80%

## 当前状态

### 已修复的问题
1. ✅ State归一化问题（评估时移除了错误的归一化）
2. ✅ 缩放因子修正（0.05 → 0.012）
3. ✅ 6D旋转转换验证（已确认正确）
4. ✅ 坐标系方向验证（已确认一致）

### 待验证
- [ ] 训练loss能否收敛
- [ ] 模型能否学会单任务
- [ ] 评估方向是否正确

## 注意事项

1. **不要提交到Git**: 这些是本地测试文件
2. **快速迭代**: 如果发现问题，立即修改重新训练
3. **记录结果**: 保存每次测试的视频和成功率
4. **对比分析**: 对比模型输出和demo动作的数值

## 下一步计划

一旦单任务测试通过：
1. 增加到3个任务
2. 增加到10个任务
3. 最后用全部90个任务训练





