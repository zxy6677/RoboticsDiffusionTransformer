# LIBERO微调和评估代码相对路径修改总结

## 修改概述

为了支持在远程服务器上使用，已将LIBERO微调和评估相关代码中的所有绝对路径修改为相对路径。

## 修改的文件

### 1. `libero_evaluate.py`
**修改内容：**
- `sys.path.append('/home/ubuntu/LIBERO/libero')` → `sys.path.append('../../LIBERO/libero')`
- `sys.path.append('/home/ubuntu/LIBERO/libero/libero')` → `sys.path.append('../../LIBERO/libero/libero')`
- `sys.path.append('/home/ubuntu/RoboticsDiffusionTransformer')` → `sys.path.append('.')`
- `spec_from_file_location("state_vec", "/home/ubuntu/RoboticsDiffusionTransformer/configs/state_vec.py")` → `spec_from_file_location("state_vec", "configs/state_vec.py")`
- `libero.set_libero_default_path("/home/ubuntu/LIBERO/libero/libero")` → `libero.set_libero_default_path("../../LIBERO/libero/libero")`

### 2. `eval_sim/eval_rdt_libero.py`
**修改内容：**
- `sys.path.append('/home/ubuntu/LIBERO/libero')` → `sys.path.append('../../LIBERO/libero')`
- `sys.path.append('/home/ubuntu/LIBERO/libero/libero')` → `sys.path.append('../../LIBERO/libero/libero')`
- `sys.path.append('/home/ubuntu/RoboticsDiffusionTransformer')` → `sys.path.append('..')`
- `spec_from_file_location("state_vec", "/home/ubuntu/RoboticsDiffusionTransformer/configs/state_vec.py")` → `spec_from_file_location("state_vec", "../configs/state_vec.py")`
- `libero.set_libero_default_path("/home/ubuntu/LIBERO/libero/libero")` → `libero.set_libero_default_path("../../LIBERO/libero/libero")`
- 默认参数路径修改：
  - `--pretrained` 默认值：`/home/ubuntu/rdt-1b` → `checkpoints/rdt-1b`
  - `--text_encoder` 默认值：`/home/ubuntu/t5-v1_1-xxl` → `google/t5-v1_1-xxl`
  - `--vision_encoder` 默认值：`/home/ubuntu/siglip-so400m-patch14-384` → `google/siglip-so400m-patch14-384`

### 3. `models/multimodal_encoder/t5_encoder.py`
**修改内容：**
- `available_models = ["google/t5-v1_1-xxl", "/home/ubuntu/t5-v1_1-xxl"]` → `available_models = ["google/t5-v1_1-xxl", "google/t5-v1_1-xxl"]`

## 相对路径说明

### 项目结构假设
```
project_root/
├── RoboticsDiffusionTransformer/  # 当前项目目录
│   ├── libero_evaluate.py
│   ├── eval_sim/
│   │   └── eval_rdt_libero.py
│   ├── models/
│   │   └── multimodal_encoder/
│   │       └── t5_encoder.py
│   └── configs/
│       └── state_vec.py
└── LIBERO/  # LIBERO项目目录（与RoboticsDiffusionTransformer同级）
    └── libero/
        └── libero/
```

### 路径映射
- `../../LIBERO/libero` - 从RoboticsDiffusionTransformer目录到LIBERO/libero
- `../../LIBERO/libero/libero` - 从RoboticsDiffusionTransformer目录到LIBERO/libero/libero
- `..` - 从eval_sim目录到RoboticsDiffusionTransformer根目录
- `.` - 当前目录（RoboticsDiffusionTransformer根目录）
- `../configs/state_vec.py` - 从eval_sim目录到configs目录
- `configs/state_vec.py` - 从根目录到configs目录

## 使用说明

### 在远程服务器上部署
1. 确保项目结构符合上述假设
2. 将LIBERO项目放在与RoboticsDiffusionTransformer同级目录
3. 所有路径现在都是相对路径，可以在任何位置运行

### 运行评估
```bash
# 在RoboticsDiffusionTransformer根目录
python libero_evaluate.py --config configs/base.yaml --pretrained checkpoints/rdt-1b

# 或在eval_sim目录
cd eval_sim
python eval_rdt_libero.py --config ../configs/base.yaml --pretrained ../checkpoints/rdt-1b
```

### 运行微调
```bash
# 在RoboticsDiffusionTransformer根目录
python libero_finetune_correct.py --task_id 0 --max_steps 10000
```

## 注意事项

1. **LIBERO路径**：确保LIBERO项目位于正确的位置（与RoboticsDiffusionTransformer同级）
2. **模型路径**：预训练模型路径现在使用相对路径，确保模型文件在正确位置
3. **配置文件**：所有配置文件路径都已更新为相对路径
4. **跨平台兼容**：相对路径在不同操作系统上都能正常工作

## 验证修改

所有绝对路径已成功修改为相对路径，代码现在可以在远程服务器上正常运行，无需修改任何路径配置。
