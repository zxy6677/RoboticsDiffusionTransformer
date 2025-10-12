# RDT在LIBERO上的微调和评估

本项目提供了在LIBERO数据集上对RDT模型进行微调训练和评估的完整流程，基于官方README的Fine-Tuning指导。

## 核心文件

### 训练相关
- `libero_finetune_correct.py` - 基于README指导的正确微调脚本
- `main.py` - 官方训练主脚本
- `train/` - 训练相关模块
  - `train.py` - 训练主逻辑
  - `dataset.py` - 数据集处理
  - `sample.py` - 数据采样
- `models/` - 模型定义
  - `rdt_runner.py` - RDT模型运行器
  - `multimodal_encoder/` - 多模态编码器

### 评估相关
- `libero_evaluate.py` - RDT在LIBERO任务上的评估脚本
- `eval_sim/eval_rdt_libero.py` - 评估模拟器

### 数据处理
- `data/hdf5_libero_dataset.py` - LIBERO数据集HDF5加载器
- `data/datasets/libero_90/` - LIBERO_90数据集

### 配置文件
- `configs/base.yaml` - 基础配置
- `configs/finetune_datasets.json` - 微调数据集配置
- `configs/finetune_sample_weights.json` - 微调采样权重
- `configs/dataset_stat.json` - 数据集统计信息
- `configs/state_vec.py` - 状态向量映射
- `configs/dataset_control_freq.json` - 数据集控制频率

## 使用方法

### 1. 微调训练

使用基于README指导的正确微调脚本：

```bash
# 在LIBERO_90的第一个任务上进行微调
python libero_finetune_correct.py \
    --task_id 0 \
    --max_steps 10000 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --output_dir checkpoints/libero_finetune
```

### 2. 直接使用main.py微调

也可以直接使用官方的main.py脚本：

```bash
python main.py \
    --pretrained_model_name_or_path=checkpoints/rdt-1b \
    --pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl \
    --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 \
    --output_dir=checkpoints/libero_finetune/task_00 \
    --train_batch_size=32 \
    --sample_batch_size=64 \
    --max_train_steps=10000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler=constant \
    --learning_rate=1e-4 \
    --mixed_precision=bf16 \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type=finetune \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=wandb
```

### 3. 模型评估

```bash
# 评估微调后的模型
python libero_evaluate.py \
    --config configs/base.yaml \
    --pretrained checkpoints/libero_finetune/task_00_KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it \
    --text_encoder google/t5-v1_1-xxl \
    --vision_encoder google/siglip-so400m-patch14-384 \
    --benchmark libero_90 \
    --num_tasks 1 \
    --max_steps 100 \
    --record_video \
    --video_output_dir videos
```

## 数据集配置

### LIBERO_90数据集已配置
- **数据集名称**: `libero_90`
- **控制频率**: 20Hz (在`configs/dataset_control_freq.json`中配置)
- **采样权重**: 100 (在`configs/finetune_sample_weights.json`中配置)
- **统计信息**: 已计算并保存在`configs/dataset_stat.json`中

### 数据集加载器
- `data/hdf5_libero_dataset.py` - 实现了LIBERO数据集的HDF5加载
- 支持状态向量映射和6D旋转表示
- 自动处理LIBERO到RDT统一动作空间的转换

## 任务说明

### LIBERO_90任务列表
1. `KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet` - 关闭柜子的顶层抽屉
2. `KITCHEN_SCENE10_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` - 拿起盘子和小碗之间的黑色碗并将其放在盘子上
3. ... (共90个任务)

## 输出文件

### 训练输出
- `checkpoints/libero_finetune/task_XX_*/` - 微调后的模型权重
- `wandb/` - 训练日志和可视化
- 训练过程可通过Wandb监控

### 评估输出
- `libero_evaluation_results.json` - 评估结果JSON文件
- `videos/` - 任务执行视频
- 控制台输出 - 实时评估进度和结果

## 基于README的微调流程

### 1. 数据集准备 ✅
- LIBERO_90数据集已下载到`data/datasets/libero_90/`
- 数据集统计信息已计算完成

### 2. 数据集加载器实现 ✅
- `HDF5LIBERODataset`类已实现
- 支持LIBERO状态到RDT统一动作空间的转换
- 使用6D旋转表示处理末端执行器姿态

### 3. 配置文件设置 ✅
- `configs/finetune_datasets.json` - 包含`libero_90`
- `configs/finetune_sample_weights.json` - 设置采样权重
- `configs/dataset_control_freq.json` - 设置控制频率为20Hz
- `configs/dataset_stat.json` - 包含数据集统计信息

### 4. 微调训练 ✅
- 使用官方`main.py`脚本进行训练
- 支持DeepSpeed和混合精度训练
- 自动保存检查点和EMA权重

## 注意事项

1. **环境要求**: 需要安装LIBERO环境和相关依赖
2. **GPU内存**: 建议使用至少24GB显存的GPU
3. **数据路径**: 确保LIBERO数据集路径正确设置
4. **模型权重**: 确保预训练模型权重文件存在
5. **遵循README**: 严格按照官方README的Fine-Tuning指导进行

## 故障排除

### 常见问题
1. **CUDA内存不足**: 减少batch_size或使用梯度累积
2. **数据集路径错误**: 检查`data/datasets/libero_90/`目录
3. **模型权重加载失败**: 检查预训练模型路径和文件完整性
4. **LIBERO环境问题**: 确保LIBERO正确安装和配置
5. **状态映射问题**: 检查`configs/state_vec.py`中的映射配置

### 调试建议
1. 使用较小的数据集进行测试
2. 检查数据加载和预处理流程
3. 验证模型输入输出维度
4. 监控训练和评估过程中的内存使用
5. 参考官方README的Fine-Tuning部分进行配置
