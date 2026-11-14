# ramen 基线审计

所有结果来自同一个 70000 步模型，未重新训练。原基线输出保留。

| 配置 | Acc@0.25 (%) | Acc@0.5 (%) | mIoU (%) | 未生成掩码项 |
|---|---:|---:|---:|---:|
| upstream | 32.39 | 9.86 | 17.60 | 14 |
| no_knn | 30.99 | 8.45 | 17.20 | 14 |
| no_scale_filter | 29.58 | 7.04 | 16.92 | 10 |
| no_scale_or_knn | 29.58 | 7.04 | 16.17 | 10 |

14 项漏检中，10 项的所选高斯在对应视角不可见（bowl、plate、sake cup），另外 4 项 napkin 被尺度/最少点数规则阻止输出。这里的不可见只证明所选簇在该视角不可见，不能单独证明文本选簇错误。关闭尺度过滤减少缺失，但整体指标下降。

原合并条件 candidate_id-max_id<leaf_num 可产生跨父簇合并。统一 top1_fixed_v1 预先采用单簇 argmax，不使用 GT 决定合并或阈值。

统一 top-1 协议：Acc@0.25=39.44%，Acc@0.5=15.49%，mIoU=20.13%。14 项是显式空预测，均计 0；指标分母仍为 71。

后续 CF 使用同一个 top1_fixed_v1 对比。该协议尚未被证实与 GaussianGraph 作者的未公开评估实现完全相同。

逐项记录：baseline_audit/items.csv；选簇记录：baseline_audit/selection.json；统一协议：protocol_top1_v1.json。
