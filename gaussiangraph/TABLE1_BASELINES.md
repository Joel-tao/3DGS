# Table 1 对照方法进度

本轮目标：先补齐 ramen 的 LangSplat、LEGaussian 原生训练，并用统一的三维解码后渲染流程比较。当前还未完成整张 Table 1。

## LangSplat

- 环境、专用 CUDA 扩展已安装在 Data 盘独立环境。
- ramen 自编码器已完成 100 epochs，最佳 checkpoint 位于 `/media/joe/Data/gaussiangraph/outputs/langsplat_ae/ramen/best_ckpt.pth`。
- 131 张图的三维潜特征编码文件已生成。
- 使用现有 OpenGaussian 的第 30000 步重建作为固定几何起点。属于共享重建的实验设置，不能声称完全照搬原论文全部训练条件。
- 三个特征层依次训练，每层 30000 步，随后自动三维解码并评估。
- 队列：`/media/joe/Data/gaussiangraph/scripts/run_langsplat_ramen.py`
- 状态：`/media/joe/Data/gaussiangraph/logs/langsplat-ramen-status.json`
- 日志：`/media/joe/Data/gaussiangraph/logs/langsplat-ramen-level1-train.log`（其余层同名替换数字）。
- 10 步短程训练与三维评估已通过，仅用于验证程序，不是最终结果。

## LEGaussian

- 使用官方 LEGaussians 的 2024 年版本 `049e12f`，隔离环境及 CUDA 扩展已编译。
- 正在从同一组 ramen 图像重新提取原生 Pyramid CLIP 和 DINO 特征；官方示例数据命名与当前 GT 不同，未混用。
- 当前预处理先运行 1 epoch 验证完整流程；正式量化需 300 epochs，随后还需训练和评估。
- 日志：`/media/joe/Data/gaussiangraph/logs/leg-ramen-preprocess-smoke.log`。
- 显存优化：CLIP 分批推理、DINO attention 按 query 分块、数据集缓存在 CPU。尚需对分块 attention 做数值等价验证。

## 统一评估的边界

实现位于 `baselines/OpenGaussian/experiments/evaluate_table1_3d.py`。

协议名 `table1_3d_classification_v1`：先对每个高斯解码并在 14 个文本类别中分类，再渲染类别区域；alpha > 0.7；空预测显式计零；71 个标注项目。该协议是公开记录的独立实现，不是作者未公开的评估脚本。

此前的 `top1_fixed_v1` 为单簇选择，不能直接与新协议数字拼成对照表。OpenGaussian 和 CF 必须使用相同新协议重评估。LangSplat 三层还需要固定无 GT 的层选择规则后汇总，不能用 GT 挑最优层。

后续仍需：LEGaussian 正式量化和训练、三层汇总、新协议下各方法重评估，再扩展其余三个场景。当前 SAM2 + confidence + CF 的实现仍属于独立实现，已有结果并未达到论文表现。

## 2026-10-29 后续进度

- LangSplat 三层均完成 30000 步及三维评估。固定的无 GT 层选择规则：每个视角、每个查询选择可见高斯的最大文本余弦相似度最高的层；相同分数优先较小层号。三层合并 mIoU **21.81%**，Acc@0.25 **30.99%**，Acc@0.5 **19.72%**；71 项，缺失预测 0。
- 汇总结果：`/media/joe/Data/gaussiangraph/experiments/langsplat_3d_multilevel_v1/metrics.json`；逐查询层选择记录在同目录 `selected_levels.json`。这是独立协议的结果。
- LEGaussian 的 DINO `utils` 导入冲突已修复：仅在 hub 加载期间隔离同名模块，结束后恢复项目模块。
- DINO 分块 attention 与原实现已做 257 token 数值对比，最大绝对误差 **7.45e-8**，测试通过，且确认项目 utils 正确恢复。
- LEGaussian 已恢复 DINO 提取，本次量化目标 **300 epochs**。新日志：`/media/joe/Data/gaussiangraph/logs/leg-ramen-preprocess.log`。
- 自动后续脚本 `scripts/run_legaussian_ramen.py` 已启动：等待预处理成功后，生成训练配置，从 SfM 点云按原生方法联合训练 RGB/语义 30000 步，再做统一三维评估；失败则记录并停止。状态：`/media/joe/Data/gaussiangraph/logs/legaussian-ramen-status.json`。
- 已启动 OpenGaussian、CF、SAM2+confidence+CF 在新协议下的重新评估，避免混用旧的单簇选择结果。

以上覆盖前文的运行中状态。LEGaussian 的正式训练与最终指标尚未完成。

## 统一新协议下 ramen 的阶段结果

单位为百分比；均为 71 项且缺失预测 0。不能与旧协议结果混用，也不是作者原表。

| 方法 | Acc@0.25 | Acc@0.5 | mIoU |
| --- | ---: | ---: | ---: |
| LangSplat | 30.99 | 19.72 | 21.81 |
| OpenGaussian | 25.35 | 16.90 | 20.48 |
| 独立 CF v2 | 9.86 | 7.04 | 6.46 |
| 独立 SAM2 + confidence + CF | 11.27 | 4.23 | 7.70 |

LEGaussian 待正式训练完成后补入。当前独立 CF 及 SAM2+confidence+CF 仍显著低于 OpenGaussian；增加模块并没有自动实现论文效果，还需要继续排查训练和关联方法。机器可读结果：`/media/joe/Data/gaussiangraph/experiments/TABLE1_RAMEN_COMMON.json`。

量化阶段修复：DINO 特征缓存已完成。原 batch_size=32 在首轮量化发生 CUDA OOM；已将量化 batch_size 改为 4 并重新运行 300 epochs，复用缓存。此批次变化影响优化轨迹，属于需要记录的设置差异。自动后续队列已重新启动。

## LEGaussian 最终运行结果（2026-10-30 核查）

ramen 的 300 轮量化、30000 步训练和统一三维评估均已完成，状态为 `complete`。71 项，缺失预测 0；Acc@0.25 **9.86%**，Acc@0.5 **0.00%**，mIoU **8.98%**。结果位于 `/media/joe/Data/gaussiangraph/experiments/legaussian_3d_v1/metrics.json`，已加入统一结果 JSON。

这标志着 ramen 三个对照方法的流程跑通；LEGaussian 数值仍不等于论文复现成功。GaussianGraph 独立实现仍待排查，其他三个场景未完成，整张 Table 1 尚未完成。

## ramen 排查

只做 ramen，暂不扩展其他场景。核对报告见 [RAMEN_ALIGNMENT_AUDIT.md](RAMEN_ALIGNMENT_AUDIT.md)。已完成簇/语义覆盖率统计与视角门槛对照；CF 的主要已证实损失来源包括聚类细碎、关联覆盖不足，以及至少 5 视角筛选。原始主结果保留，诊断结果另存。

## 固定聚类的关联对照（进行中）

已启动 `/media/joe/Data/gaussiangraph/scripts/run_cf_association_audit.py`。CF 与 SAM2+CF 各运行纯 IoU 关联，分别保留/关闭伪特征方差筛选；每次关联完成后自动用至少 1 和 5 视角门槛评估，共 4 个关联实验、8 组指标。几何、聚类、训练特征、GT 均固定，不重新训练，不以 GT 选择参数。

修复 `filter=False` 仍会清空高方差 mask 的问题；默认 `filter=True` 的行为不变。关闭关联调试图输出，保留逐视角匹配索引、分数及有效标志，供后续分析。

- 状态：`/media/joe/Data/gaussiangraph/logs/cf-association-audit-status.json`
- 当前第一组日志：`/media/joe/Data/gaussiangraph/logs/cf_iou_filtered-association.log`
- 所有新结果：`/media/joe/Data/gaussiangraph/experiments/association_audit_v1/`
- 完成后自动生成 `summary.json` 和 `RESULTS.md`。状态 `complete` 表示这批关联对照结束，不表示论文复现成功。

## 关联对照已完成

4 次关联、8 组评估全部结束，状态 complete。固定纯 IoU 与至少 5 视角门槛，关闭伪特征方差筛选使 CF mIoU 从 9.14% 到 15.57%，SAM2+confidence+CF 从 7.70% 到 17.66%。这确认该筛选对现有模型造成明显损失；纯 IoU 本身不保证改善，需与旧联合分数区分。结果详见 `/media/joe/Data/gaussiangraph/experiments/association_audit_v1/RESULTS.md`。

尚未完成单位特征聚类重训、SAM2 训练区域与 CLIP 区域对齐及置信度损失尺度对照；本批 complete 不代表全部问题已解决。

## 归一化聚类与区域对齐实验已启动

模型输出：`/media/joe/Data/gaussiangraph/outputs/ramen_sam2_cf_unit_v1`。从 30000 步固定重建重新训练至 70000 步，Control/Follow 使用单位实例特征，XYZ 权重保持 .5；其他训练设置及置信度损失不变。

自动流程先对该模型用原始 CLIP 特征、纯 IoU、关闭伪特征筛选关联，评估至少 1/5 视角条件；然后重新生成严格对应互斥训练掩码的 CLIP 裁剪（同一编码器、相同 mask ID），固定该模型重新关联并评估。两个分支只改变 CLIP 裁剪来源；均保留旧结果，不按 GT 挑参数。

- 队列：`/media/joe/Data/gaussiangraph/scripts/run_cf_unit_aligned.py`
- 状态：`/media/joe/Data/gaussiangraph/logs/cf-unit-aligned-status.json`
- 训练日志：`/media/joe/Data/gaussiangraph/logs/ramen-sam2-cf-unit-training.log`
- 最终汇总：`/media/joe/Data/gaussiangraph/experiments/cf_unit_aligned_v1/summary.json`（全部完成后共 4 条）

归一化路径已通过尺度不变性测试，两份新脚本通过编译检查。训练效果、对齐特征提取和最终指标仍待运行验证。损失权重暂不改动，待本轮对照后再判断。

## 归一化与区域对齐实验完成

全部流程已完成。至少 5 视角时，归一化聚类+原始 CLIP mIoU 14.89%，归一化聚类+区域对齐 CLIP 14.39%，均低于上一版同关联设置的 17.66%。不采用新版本替换此前较好结果。详见 `/media/joe/Data/gaussiangraph/experiments/cf_unit_aligned_v1/RESULTS.md`。下一项候选为固定细分前 checkpoint 的细分开关对照，尚未启动。

## B：同 checkpoint 细分开关对照（2026-10-31 启动）

顺序执行 B，再根据结果推进 C（包含比例及跨视角检查）、A（固定特征重新聚类），不混合多个改动。B 使用 `ramen_sam2_confidence_v1/cf_checkpoint67999.pth`；两组特征、几何严格相同，无优化器更新，只改变父簇/细分标签。特征 SHA256 存入各组 config.json。采用原始 SAM2 CLIP、纯 IoU、关闭伪特征筛选，每组评估 1/5 视角门槛。

阶段统计：Control（49999 步）205 簇，仅 5000 控制点；Follow（67999 步）3348 簇，714346 高斯；同特征细分后 5263 簇。Follow 已有 1768 个不足 32 点的小簇，细分后这个数量不变。最多分裂的父簇 id=110，15087 点，被拆成 156 个子簇。不能据簇数单独判断语义正确性。

- 状态：`/media/joe/Data/gaussiangraph/logs/refinement-ablation-status.json`
- 日志：`/media/joe/Data/gaussiangraph/logs/refinement-no_refine-association.log`，随后 `refinement-refine-association.log`
- 实验目录：`/media/joe/Data/gaussiangraph/experiments/refinement_ablation_v1/`
- 统计：`stage_stats.json`、`parent_children.json`；全部结束后生成 `RESULTS.md`。

当前仅 B 已启动，C 和 A 尚未启动。

## B 对照完成（2026-10-31）

同一 67999 步 checkpoint、相同特征和几何的细分开关对照全部结束。至少 5 视角时，关闭细分：3348 簇，非零 CLIP 覆盖 65.77%，mIoU 18.34%，Acc@0.25 28.17%，Acc@0.5 18.31%；开启细分：5263 簇，非零 CLIP 覆盖 40.02%，mIoU 17.66%，Acc@0.25 32.39%，Acc@0.5 16.90%。关闭细分提高 mIoU 约 0.68 个百分点，但 Acc@0.25 下降；不能称全部指标改善。细分影响关联覆盖，但不是当前低分的唯一或已证实最大原因。C、A 固定特征对照尚未启动。

完整结果：`/media/joe/Data/gaussiangraph/experiments/refinement_ablation_v1/RESULTS.md`。
