# GitHub上传指南

## 🚀 将RDT LIBERO微调代码上传到GitHub

### 当前状态
✅ 代码已提交到本地git仓库  
✅ 主要文件已准备好  
✅ 文档已更新  

### 上传步骤

#### 1. 在GitHub上创建新仓库

1. 访问 [GitHub新建仓库页面](https://github.com/new)
2. 填写仓库信息：
   - **仓库名称**: `rdt-libero-finetune` (或您喜欢的名称)
   - **描述**: `RDT fine-tuning on LIBERO dataset`
   - **可见性**: 选择 Public 或 Private
   - **初始化选项**: 
     - ❌ 不要勾选 "Add a README file"
     - ❌ 不要勾选 "Add .gitignore"
     - ❌ 不要勾选 "Choose a license"

3. 点击 "Create repository"

#### 2. 获取仓库URL

创建完成后，GitHub会显示仓库URL，类似：
```
https://github.com/YOUR_USERNAME/rdt-libero-finetune.git
```

#### 3. 上传代码

在终端中运行以下命令（替换为您的实际仓库URL）：

```bash
# 方法1: 添加新的远程仓库
git remote add my-repo https://github.com/YOUR_USERNAME/rdt-libero-finetune.git
git push -u my-repo main

# 或者方法2: 替换当前的origin
git remote set-url origin https://github.com/YOUR_USERNAME/rdt-libero-finetune.git
git push -u origin main
```

#### 4. 验证上传

上传完成后，访问您的GitHub仓库页面，应该能看到以下文件：

### 📁 主要文件结构

```
rdt-libero-finetune/
├── README_LIBERO.md                    # 详细使用说明
├── LIBERO_FINETUNE_SUMMARY.md         # 项目总结
├── libero_finetune_correct.py         # 微调脚本
├── libero_evaluate.py                 # 评估脚本
├── data/
│   ├── hdf5_libero_dataset.py         # LIBERO数据集加载器
│   └── datasets/libero_90/            # LIBERO数据集
├── eval_sim/
│   └── eval_rdt_libero.py            # 评估模拟器
├── configs/
│   ├── finetune_datasets.json        # 微调数据集配置
│   ├── finetune_sample_weights.json  # 采样权重配置
│   ├── dataset_control_freq.json     # 控制频率配置
│   └── dataset_stat.json             # 数据集统计信息
└── models/                            # 模型定义
    ├── rdt_runner.py
    └── multimodal_encoder/
```

### 🎯 使用说明

上传完成后，其他人可以通过以下方式使用您的代码：

#### 1. 克隆仓库
```bash
git clone https://github.com/YOUR_USERNAME/rdt-libero-finetune.git
cd rdt-libero-finetune
```

#### 2. 安装依赖
```bash
conda create -n rdt python=3.10.0
conda activate rdt
pip install -r requirements.txt
```

#### 3. 运行微调
```bash
python libero_finetune_correct.py --task_id 0 --max_steps 10000
```

#### 4. 运行评估
```bash
python libero_evaluate.py --pretrained checkpoints/libero_finetune/task_00_* --benchmark libero_90
```

### 📝 仓库描述建议

在GitHub仓库页面，您可以添加以下描述：

```markdown
# RDT Fine-tuning on LIBERO Dataset

This repository contains the implementation for fine-tuning RDT (Robotics Diffusion Transformer) on the LIBERO dataset.

## Features

- ✅ Complete fine-tuning pipeline for LIBERO_90 dataset
- ✅ Model evaluation and video recording
- ✅ HDF5 dataset loader for LIBERO
- ✅ Based on official RDT README guidance
- ✅ Comprehensive documentation

## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run fine-tuning: `python libero_finetune_correct.py --task_id 0`
4. Evaluate model: `python libero_evaluate.py --pretrained checkpoints/libero_finetune/task_00_*`

## Documentation

- [README_LIBERO.md](README_LIBERO.md) - Detailed usage guide
- [LIBERO_FINETUNE_SUMMARY.md](LIBERO_FINETUNE_SUMMARY.md) - Project summary
```

### 🔧 故障排除

如果遇到上传问题：

1. **认证问题**: 确保您已登录GitHub并配置了SSH密钥或Personal Access Token
2. **权限问题**: 确保您有权限推送到目标仓库
3. **网络问题**: 检查网络连接，必要时使用代理

### 📊 项目亮点

- 🎯 **基于官方指导**: 严格按照RDT官方README的Fine-Tuning指导实现
- 🔧 **完整流程**: 包含数据集配置、微调训练、模型评估的完整流程
- 📚 **详细文档**: 提供详细的使用说明和项目总结
- 🧹 **代码清理**: 移除了不必要的文件，只保留核心功能
- 🎥 **视频录制**: 支持任务执行过程的视频录制

现在您的代码已经准备好上传到GitHub了！按照上述步骤操作即可。


