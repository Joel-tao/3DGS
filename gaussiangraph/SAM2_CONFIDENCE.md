# SAM2_L 与置信度监督

本轮将两项方法差异接入，但还不包含 LLaVA/Grounding DINO 的完整前景定位和场景图。
它仍是根据论文实现的分割实验，不能直接标作作者官方实现。

## 已验证

- 下载原始 SAM2 Hiera-L，源码固定为官方 2024-07-29 的 658aaba 提交，兼容当前 PyTorch 2.3.1，未升级或替换渲染器的 PyTorch。
- SAM2 原版不是 SAM2.1。论文写 SAM2_L，作者公开提取脚本默认配置为 SAM2.1；这一版本歧义保留记录。
- 第一张 ramen 图像成功产生 19 个掩码、19 个置信度及 19×512 CLIP 特征，坐标保持原始 988×731。
- CLIP 为 ViT-B-16 / laion2b_s34b_b88k；重新编码的 14 个查询与原文本特征余弦最小值约 0.999996，评估仍保持原文本特征。
- 置信度损失的公式、零置信度梯度和空掩码/索引校验测试通过。

## 提取和文件格式

代码：`baselines/OpenGaussian/experiments/extract_sam2.py`。
数据：`/media/joe/Data/gaussiangraph/data/lerf_ovs_sam2/ramen`。
images、sparse、GT 链接到原始数据，language_features 独立保存，不覆盖旧特征。

- `<frame>_sam2.npz`：原始掩码压缩位图、bbox、predicted_iou、stability_score。
- `<frame>_s.npy`：训练标签图，兼容原读取器的 level 3；level 0–2 全为 -1，不虚称已有四级 SAM2 掩码。
- `<frame>_f.npy`：每个原始 SAM2 掩码的 CLIP 特征。
- `<frame>_c.npy`：与标签/特征行严格对应的置信度。
- `feature_source.json`：源码 commit、权重 SHA256、模型和提取阈值。

SAM2 自动掩码使用 32×32 点网格，每批 8 点，pred_iou_thresh=0.8、stability_score_thresh=0.95、无额外图像裁剪层。
按面积从大到小写标签，小掩码覆盖大掩码的重叠区域；CLIP 裁剪保留各自原始掩码，背景置零后补成正方形。
这些预处理规则不是论文中完整公开的唯一设置。当前也未实现作者的四粒度自定义 SAM2 生成器。
SAM2 与 CLIP 分两阶段加载，避免两个大模型同时占用 8GB 显存。逐帧缓存以支持恢复。

## 置信度定义与损失

论文 Eq.3/4 引入 Conf，但未明确公开其数值来源。本次定义：

`Conf = clamp(predicted_iou,0,1) × clamp(stability_score,0,1)`。

作者二维代码也将两者乘积用于掩码筛选，但这不足以证明其训练 Conf 与本定义完全相同。
两个原始分数均保留，便于后续替换定义。本次把掩码质量分数作为掩码内统一置信度，不声称是逐像素预测的不确定性。

`utils/confidence_loss.py` 实现：

- 平滑项为带 Conf 的掩码内平方特征距离（Eq.3），再除以图像像素数；权重仍为 0.01。
- 对比项使用掩码对置信度均值与均值特征的平方距离倒数（Eq.4），保留 m(m+1) 分母。
- 距离下限 1e-6，避免零距离除零；空掩码不参与掩码对计算。
- 像素数归一化、损失权重和数值稳定项均为明确记录的实现选择，不是论文给出的完整超参数。

训练用 `--confidence-dir .../language_features` 显式启用，未提供该参数时保持原损失，防止污染 v2 对照。
新实验重新从 30000 步几何检查点学习实例特征，不沿用旧 SAM 掩码得到的特征或梯度历史。
该轮最终关联使用论文描述的 IoU 匹配入口（`--association-score iou`）；v2 保持原 OpenGaussian 联合评分。
所以 SAM2 轮是进一步方法对齐的组合实验，不是只改变单一因素的严格消融。

## 自动执行队列与状态

`experiments/run_sam2_confidence.py` 顺序使用 GPU，避免与关联或训练并发导致显存不足：

1. 完成 131 张图像的 SAM2/CLIP/置信度提取并校验对齐。
2. 在实际 SAM2 数据上跑包含所有 CF 阶段的短训练，检查置信度损失可运行。
3. 修复版 CF v2 关联、统一 top-1 评估。
4. 从 30000 步开始训练 SAM2＋置信度＋修复版 CF 至 70000 步。
5. IoU 语言关联、统一 top-1 评估，输出 `experiments/SAM2_CF_COMPARISON.json`。

当前阶段见 `/media/joe/Data/gaussiangraph/logs/ramen-sam2-status.json`，不能仅凭代码存在就认为全部实验已完成。
完整命令保存在 `experiments/SAM2_CONFIDENCE_PIPELINE.json`，执行源码快照为 `experiments/sam2_confidence_source.tar.gz`。
任一步失败即停止并保存对应日志。所有新数据、模型和指标都有独立目录。


## 最终状态与结果

整个队列已完成，状态为 complete。统一 top-1 协议下，基线 mIoU=20.13%，CF v2=5.49%，SAM2＋置信度组合=5.90%（Acc@0.25=2.82%，Acc@0.5=0%，13 项空预测）。本次独立实现明显低于基线，不能视为论文复现成功，也不能把组合实验的退化直接归因于 SAM2 或置信度。完整汇总：`/media/joe/Data/gaussiangraph/experiments/FINAL_RESULTS.md`；原始指标：`SAM2_CF_COMPARISON.json`。
