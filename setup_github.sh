#!/bin/bash

# RDT LIBERO Fine-tuning GitHub Setup Script
# 这个脚本将帮助您将代码上传到您的GitHub仓库

echo "🚀 RDT LIBERO Fine-tuning GitHub Setup"
echo "======================================"

# 检查当前git状态
echo "📋 检查当前git状态..."
git status

echo ""
echo "📝 请按照以下步骤操作："
echo ""
echo "1. 在GitHub上创建一个新的仓库："
echo "   - 访问 https://github.com/new"
echo "   - 仓库名称建议: rdt-libero-finetune"
echo "   - 描述: RDT fine-tuning on LIBERO dataset"
echo "   - 选择 Public 或 Private"
echo "   - 不要初始化README（因为我们已经有了）"
echo ""
echo "2. 获取您的仓库URL，然后运行以下命令："
echo ""
echo "   # 添加您的远程仓库"
echo "   git remote add my-repo https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo ""
echo "   # 推送到您的仓库"
echo "   git push -u my-repo main"
echo ""
echo "3. 或者，如果您想替换当前的origin："
echo ""
echo "   # 更改远程仓库URL"
echo "   git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git push -u origin main"
echo ""

# 显示当前提交信息
echo "📊 当前提交信息："
git log --oneline -1

echo ""
echo "✅ 代码已准备好上传到GitHub！"
echo "📁 主要文件包括："
echo "   - libero_finetune_correct.py (微调脚本)"
echo "   - libero_evaluate.py (评估脚本)"
echo "   - data/hdf5_libero_dataset.py (数据集加载器)"
echo "   - README_LIBERO.md (使用说明)"
echo "   - LIBERO_FINETUNE_SUMMARY.md (项目总结)"
echo ""
echo "🎯 下一步：请按照上面的步骤在GitHub上创建仓库并推送代码！"


