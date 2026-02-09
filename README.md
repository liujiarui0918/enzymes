# 数据集介绍
  ## 简介
  ENZYMES 是从 BRENDA 酶数据库获得的 600 个蛋白质三级结构的数据集。 ENZYMES 数据集包含 6 种酶。
  ## 引文
  # OpenDataLab ENZYMES

  这是一个用于复现并对比 GIN（Graph Isomorphism Network）在 ENZYMES 数据集上训练曲线的简单项目。

  目录结构与用途：
  - `raw/GraphRNN/dataset/ENZYMES/`：原始 ENZYMES 数据文件和加载脚本 `load_data.py`。
  - `train_gin.py`：训练脚本，包含一个可变层数的 GIN 实现、训练/验证循环和绘图保存功能。
  - `requirements.txt`：运行所需 Python 包列表。

  快速开始：
  1. 安装依赖：
  ```
  pip install -r requirements.txt
  ```
  2. 运行小规模试验（示例：2 层，30 epochs）：
  ```
  python train_gin.py --layers 2 --epochs 30 --batch-size 16
  ```
  训练结束后会在仓库根目录生成 `training_curves.png`。

  说明：
  - 你可以通过修改 `--layers` 比较不同层数的训练/验证曲线。脚本支持保存最终的训练曲线图像，便于提交和分享。

  如果你希望我自动运行实验并提交生成的图片到远程仓库，我可以在当前环境里执行并尝试 `git add/commit/push`（需要你本地已配置远端和认证）。

  最新对比实验结果
  - 我对 `train_gin.py` 增加了 `--compare-layers` 模式，可以一次性比较多个层数并生成并列对比图。
  - 我在本地跑了 `--compare-layers 2,4,6`（每组 100 epochs，hidden=128，dropout=0.5，weight_decay=5e-4），得到的最好配置为：
    - 最佳层数：2 层
    - 验证集最佳准确率（best val acc）：约 0.7167（在 epoch 45 时达到）
  - 对比图已保存为 `compare_result_compare.png`，并已提交到远程仓库。

  如何运行层数对比
  ```
  python train_gin.py --compare-layers 2,4,6 --epochs 100 --hidden-dim 128 --dropout 0.5 --batch-size 16 --lr 1e-3 --weight-decay 5e-4 --out compare_result.png
  ```

  我也保存了每个配置的最优模型（例如 `best_model_L2.pth`），可用于后续微调或评估。
  
  **交叉验证（k-fold）结果与图表**
  - 针对最优组合 `L2_H128_lr0.001_wd0.0005` 我做了 5-fold 交叉验证，平均验证准确率：约 0.687（std ≈ 0.033）。
  - 每折的最佳验证准确率和对应的 epoch 已保存于 `final_cv/results_L2_H128_lr0.001_wd0.0005.json`。
  - 我还生成了可视化图表，保存在 `figures/`：
    - `figures/fold_best_accuracies.png`：每折的最佳验证准确率柱状图。
    - `figures/fold_epoch_vs_acc.png`：每折最佳准确率对应的 epoch 散点图。

  **模型权重与上传说明**
  - 模型权重位置：`final_cv/best_overall.pth`（以及每折的 `best_...foldN.pth`）。
  - 注意：如果权重文件较大（>50MB），建议使用 Git LFS 或把权重放到 GitHub Releases / 私有云盘再在 README 中添加下载链接。本仓库当前包含训练产生的小型权重文件；如果你希望我把大型权重推到远程仓库，我可以帮你把文件迁移到 Release 或配置 Git LFS（需要你的授权与网络访问）。

@article{borgwardt2005protein,
title={Protein function prediction via graph kernels},
author={Borgwardt, Karsten M and Ong, Cheng Soon and Sch{\"o}nauer, Stefan and Vishwanathan, SVN and Smola, Alex J and Kriegel, Hans-Peter},
journal={Bioinformatics},
volume={21},
number={suppl\_1},
pages={i47--i56},
year={2005},
publisher={Oxford University Press}
}
```
  ‌​‌‌​​​​‌​​​‌‌‌‌‌​​‌‌​‌​‌​​‌​​​‌‌​‌‌‌​‌‌‌​​‌‌‌‌​‌​​​‌​‌‌‌​​‌‌‌‌​‌​‌‌​​‌‌‌​​‌‌‌‌​‌​​‌‌‌​‌