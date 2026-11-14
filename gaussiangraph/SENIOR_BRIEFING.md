# 给师兄的 ramen 实验说明

汇报 PDF：[ramen_experiment_briefing.pdf](results/experiments/senior_briefing/ramen_experiment_briefing.pdf)

## 可直接讲的三分钟版本

我目前做的是 GaussianGraph 在 LERF ramen 上的三维开放词汇分割复现与诊断，范围对应 Table 1 的一个场景。现在已经跑通三维重建、三个对照方法和独立实现的 Control-Follow，但还没有完整复现作者结果，也没有完成场景图和语言关系推理。

数据有 131 张多视角图像以及相机参数。重建用 114 张训练、17 张测试；语义分割用 7 个标注视角的 71 项进行评估。分割评估视角与重建的测试划分不是同一概念。3DGS 最终约 71.4 万高斯，测试视角的 RGB 渲染已经比较接近真实照片。

方法上，我先用掩码监督每个高斯的实例特征，再通过 Control-Follow 聚类。随后把簇渲染到多视角，与二维掩码关联并聚合 CLIP 特征。用文本与 CLIP 的相似度给高斯分类，渲染预测掩码并对比 GT。这个流程能产生真实查询分割，但还没有 LLaVA 场景图。

对照方法包括 LangSplat、LEGaussian 和 OpenGaussian。当前本地协议下 mIoU 分别是 21.81%、8.98%、20.48%。这与论文的排名和数值存在差异，不能直接当成方法优劣结论，评估和解码需要继续核对。

我做了几组诊断：关闭伪特征筛选有明显帮助，SAM2+CF 从 7.70% 提升到 17.66%；归一化聚类和裁剪区域对齐没有带来提升。最严格的一组在同一 67999 步 checkpoint 上比较是否细分，固定所有特征与几何：关闭细分使非零 CLIP 覆盖从 40.02% 到 65.77%，但 mIoU 只从 17.66% 到 18.34%，且 Acc@0.25 下降。因此不能说最后细分是全部问题。

下一步先核准基线和评估，包括 LEGaussian 的解码输入是否归一化、LangSplat 的层选择、背景类别和遮挡处理；随后再做固定簇的关联对照。目标是把 ramen 的方法和评估解释清楚，而不是先把四个场景的表填满。

## 展示图片时怎么讲

1. 3DGS 图：左边是测试照片，中间是模型渲染，右边是相对深度；RGB 重建接近不代表物体语义准确。
2. 分割图：每行一个查询，每列一个方法；绿色来自真实保存的预测掩码，每格的 IoU 与 GT 按同一脚本计算。此页固定展示碗、筷子、鸡蛋、海苔，没有按效果排序。
3. 错误例子：筷子和海苔在该视角较好；鸡蛋常选到更大的碗内区域甚至桌面，提示存在语义混淆和过量预测。单个视角不能代表全场景。
4. CF 两列分别为 70000 步与关闭细分的 67999 步，图片仅供查看实际输出；细分的因果结论应引用单独的同 checkpoint 实验。

## 可打开的材料

- [3DGS 重建图](results/experiments/gs_gallery/ramen_3dgs.png)
- [分割预览](results/experiments/segmentation_gallery/preview.png)
- [完整 71 项查询对比 PDF](results/experiments/segmentation_gallery/all_queries/README.md)
- [逐查询 IoU CSV](results/experiments/segmentation_gallery/per_query_iou.csv)
- [同 checkpoint 细分对照](results/experiments/refinement_ablation_v1/RESULTS.md)
- [关联筛选对照](results/experiments/association_audit_v1/RESULTS.md)
- [归一化与区域对齐对照](results/experiments/cf_unit_aligned_v1/RESULTS.md)

## 不应声称的结论

不能说已经复现完整 GaussianGraph、已经完成 detection/scene graph、超过论文的 LangSplat，或细分是所有低分的原因。当前是单场景独立实现与可追溯的诊断结果。C（包含比例关联）与 A（固定特征重聚类）尚未开展；此前归一化实验包含重训，不能替代 A。
