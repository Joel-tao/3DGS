# 多场景：有／无实例特征损失的配对实验

## “有／无特征损失”是什么意思？

这里的特征是高斯携带的三维实例编码（可显示成实例颜色），不是物体的 CLIP 文本／图像语义向量。把这些实例编码渲染到当前视角，与检测、SAM 分割和跨帧关联生成的实例目标图比较：

`L_feature = 0.8 × L1 + 0.2 × (1 − SSIM)`

- **有特征损失**：`L = 0.5 L_RGB + 1.0 L_depth + 0.5 L_feature`。
- **无特征损失**：同样的流程，只令 `L_feature` 的权重为 0。

无特征损失组仍然运行 YOLO、SAM、CLIP、物体关联和实例 ID 维护。它不是“没有物体实例”的普通 3DGS。

例如，椅子与靠垫的边界处，RGB／深度损失关注颜色和几何是否重建正确；实例特征损失还约束投影后的实例编码是否落在对应物体区域。由于渲染可微，它可以通过梯度影响高斯位置、尺度、透明度和实例特征，让边界更一致。但效果应由实验检验，不能保证每个场景都改善。

实例目标来自模型预测及关联，不是真值人工标注；所以这项约束可能同时受到错误分割的影响。

## 本次比较内容

3 个场景 `room0`、`room1`、`office0`，每个分别运行 feature_on／feature_off，共 6 次。每对仅改变特征损失权重，使用 seed=0、相同 200 个采样帧（stride=10）、424×240、真值位姿、20 次建图迭代、相同检测／分割／CLIP 模型和相同评估视角。关闭 DAM、语言模型分类和动态图更新。

- 重建质量：PSNR、MS-SSIM、LPIPS、深度 L1。
- 实例表现：实例数量、每实例高斯支持量、小实例数量、前景高斯比例以及配对 RGB／实例渲染。这些是统计和定性观察，不是分割准确率。
- 资源：总耗时、建图流程耗时、PyTorch 峰值分配／保留显存、采样的进程内存峰值。

每场景评估 21 个采样训练视角，不是独立测试视角。只有一个随机种子，未做显著性分析。关闭组仍计算辅助特征渲染，权重置零是损失消融，不是删除计算分支的速度优化。总耗时包括初始化、建图、评估、保存，可能受温度、桌面负载和文件缓存影响。

## 文件与重跑

实验独立存放：`/media/joe/Data/Dynamic-GSG/experiments/feature_ablation_20261017/`，保留已有 room0 结果。

```bash
cd /home/joe/Dynamic-GSG
# 目标目录必须不存在已有实验，防止覆盖
/home/joe/anaconda3/envs/dgsg/bin/python scripts/run_feature_ablation.py --root /media/joe/Data/Dynamic-GSG/experiments/feature_ablation_new
/home/joe/anaconda3/envs/dgsg/bin/python scripts/summarize_feature_ablation.py /media/joe/Data/Dynamic-GSG/experiments/feature_ablation_new
/home/joe/anaconda3/envs/dgsg/bin/python scripts/plot_feature_ablation.py /media/joe/Data/Dynamic-GSG/experiments/feature_ablation_new
```

`manifest.json` 保存所有配置及训练脚本哈希；每个运行保存展开配置与日志。`status.json` 保存当前进度；`batch_results.json` 保存完成状态。汇总脚本生成 `REPORT.md`、`summary.csv`、`summary.json`、`deltas_on_minus_off.json`；绘图脚本生成 `comparison_metrics.png` 和各场景 `*_paired_views.png`。

请以最终报告判断效果；实例数更多、耗时更短均不自动代表算法更好。论文级语义分割准确率仍需要真值与一致评估协议，本实验不替代该部分。

## 本机已完成结果（2026-10-17）

6/6 次运行正常退出，每次 200 帧，21 个采样训练视角。运行总耗时约 41.44 分钟，单次约 6.58～7.25 分钟，PyTorch 峰值保留显存约 5.95～6.15 GiB。所有配对已再次核对：除了特征损失权重和运行／可视化输出路径，其余配置一致。

| 场景 | 开启 PSNR | 关闭 PSNR | 开启减关闭 | 实例数（开启／关闭） |
|---|---|---|---|---|
| room0 | 29.052 | 29.453 | -0.402 dB | 34／33 |
| room1 | 30.536 | 31.065 | -0.528 dB | 24／24 |
| office0 | 37.201 | 37.735 | -0.533 dB | 20／20 |

当前设置下三场景的 PSNR、MS-SSIM、LPIPS、深度 L1 都没有显示开启组重建更好；开启组高斯数量较少。实例图整体布局接近，局部边界与空洞有差异，尚无真值支持分割准确率结论。因此，本次完成了实验流程和如实对照，尚未复现论文所报告的完整性能与改善趋势。低迭代预算、预测掩膜质量、多目标优化取舍等都可能影响结果，当前实验不能确定具体原因。

完整数据和图表见实验目录的 `REPORT.md`、`summary.csv`、`comparison_metrics.png`、`*_paired_views.png`。未继续追加调参或多种子运行，以保留这次固定协议结果。
