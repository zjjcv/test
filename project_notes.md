# 项目工作记录与反思 (Project Notes & Reflections)

## 2025-12-30
- **任务**: 初始化项目记录文档。
- **状态**: 已创建。
- **背景**: 用户要求建立文档记录工作过程中碰到的问题和执行记录。

## 待办事项 (Backlog)
- [ ] 优化 `utils/plot.ipynb` 中的 WLMD 绘图代码：
    - [ ] X轴改为明确的 Log10 刻度显示 ($10^n$)。
    - [ ] 误差带虚化处理 (调整 alpha 和平滑度)。
    - [ ] 绘制拟合（平滑）后的曲线。

1. 使用/data/zjj/test/src/checkpoint_sp_l2_get.py训练保存好的权重文件，以及种子固定好的数据划分，来计算模型的AGOP (Average Gradient Outer Product)和循环偏差 (Cycle Bias / Circulant Score)随着训练步数的变化，AGOP由于全参数梯度外积矩阵太大（参数量平方级别），建议计算 Input-Gradient AGOP（对输入 Embedding 的梯度协方差），这能直接反映模型关注的输入模式；循环偏差对权重矩阵进行 离散傅里叶变换 (DFT)，计算其在对角线或特定频率上的能量占比。如果能量集中在某些频率上，说明存在强烈的循环偏差。
根据上述要求新建一个utils/AGOP_CycleBias.py，将我需要的数据（步数,训练和测试的loss以及acc加上上述两个指标）保存为csv文件到/data/zjj/test/results/Data/AGOP_CycleBias文件夹下。

在/data/zjj/test/utils/plot.ipynb新建一个cell，利用utils/AGOP_CycleBias.py刚生成的csv文件绘制高质量美观科研绘图，实现：观察这两个指标随着训练/测试的acc和loss的变化趋势。

2. 这是一个关于验证 LLC 测量工具几何鲁棒性的实验研究：该实验旨在探究在保持 Transformer 模型（2层/4头/128维）功能与性能完全不变的前提下，人为的参数缩放（LayerNorm 前权重的 $\alpha$ 缩放与 FFN 层权重的 $\beta$ 配对缩放）是否会干扰 局部学习系数（LLC） 的估算值。使用权重衰减=1设置下最后一步的模型权重文件（加减乘除各一个），来做实验通过 $30 \times 30$ 的网格搜索（共900次前传），对比了 标准 SGLD、pSGLD 和 Riemannian-SGLD 三种采样器。预期结果是通过热力图直观展示：标准 SGLD 可能会随参数变大而估值失准（对缩放敏感），而 R-SGLD 和 pSGLD 理论上应表现出更好的几何不变性（颜色分布均匀），从而确认哪种工具适合用于分析 Grokking 现象中的权重增长问题。此文件只需要输出绘制此图所需的csv文件刀到/data/zjj/test/results/Data/EPR-SGLD文件夹下，加减乘除四种运算各四个csv文件。

本实验旨在验证 LLC（局部学习系数）估算工具在参数几何形变下的鲁棒性，并重点评估新提出的 SI-SGLD（尺度不变 SGLD） 对抗权重衰减（Weight Decay=1）引起参数范数漂移的能力。实验基于 2层 Transformer 架构，分别加载 加、减、乘、除 四种模算术任务的收敛权重，在严格保持模型 Logits 输出不变的前提下，执行 LayerNorm 前权重缩放 ($\alpha$) 与 FFN 层配对缩放 ($\beta$) 的双重变换。通过 $30 \times 30$ 的对数网格搜索（每任务 900 次实验），横向对比了标准 SGLD、pSGLD、Riemannian-SGLD 及 SI-SGLD。特别地，SI-SGLD 采用“层级标量预调节”机制（Layer-wise Scalar Preconditioning），通过计算每个参数张量的 $L_2$ 范数平方（$G = \|W\|_2^2 + \epsilon$）作为度量矩阵，动态调整梯度的更新步长与注入噪声的方差。这种设计强制采样步长随参数模长线性伸缩，从而消除了参数整体尺度变化对 LLC 能量测量的干扰。预期结果将通过热力图证明 SI-SGLD 具有最佳的几何不变性，最终生成的四个任务对应的 CSV 数据文件将保存在 /data/zjj/test/results/Data/EPR-SGLD 目录下，每个任务4各csv代表四个方法，共16各csv文件。

在/data/zjj/test/utils/plot.ipynb新建一个cell，利用刚生成的csv文件绘制高质量美观科研绘图，实现：使用美观热力图，坐标分别是α和β，LLC值越大越亮，由暗紫到橙黄的颜色变化。

3. 