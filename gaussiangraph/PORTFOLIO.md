# GaussianGraph：Ramen 训练、对照和诊断

原始源码来自 [WangXihan-bit/GaussianGraph](https://github.com/WangXihan-bit/GaussianGraph)，版本见 UPSTREAM.json。保留上游归属，独立本地实验见以下材料。

- [实验汇报](SENIOR_BRIEFING.md)
- [基线与评估进展](TABLE1_BASELINES.md)
- [重建图](results/experiments/gs_gallery/ramen_3dgs.png)
- [分割预览](results/experiments/segmentation_gallery/preview.png)
- [71项查询完整PDF](results/experiments/segmentation_gallery/all_queries/README.md)
- [逐查询IoU](results/experiments/segmentation_gallery/per_query_iou.csv)
- [消融结果](results/experiments/)

`local_scripts/`、`local_configs/` 和 `training_logs/` 从数据盘补入；`results/output_metadata/` 保留各训练输出中的配置与数值。`baselines/OpenGaussian/` 为本地源码快照，原始版本和修改见依赖清单/补丁。`external_dependencies/` 保存其他外部基线的版本及已跟踪文件修改补丁。

没有把重建质量等同于语义准确率，没有声称完成完整GaussianGraph场景图和语言关系推理。所有大型训练权重及逐帧中间结果仍在原始数据盘，未上传，见根目录清单。
