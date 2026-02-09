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