"""Summarize matched ablations; instance statistics are not segmentation accuracy."""
import argparse,csv,gzip,json,pickle
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
a=p.parse_args()
batch=json.loads((a.root/'batch_results.json').read_text())
rows=[]
for record in batch:
    if record['returncode']:
        rows.append(dict(run=record['run'],scene=record['scene'],feature_weight=record['feature_weight'],status='failed'));continue
    r=a.root/record['run'];cfg=json.loads((r/'config.resolved.json').read_text());perf=json.loads((r/'run_metrics.json').read_text())
    with np.load(r/'params_with_idx.npz') as d:
        ids=d['object_idx'].ravel().astype(int);gaussians=len(ids)
        unique,counts=np.unique(ids[ids>0],return_counts=True)
    with gzip.open(r/'objects.pkl.gz','rb') as f:objects=pickle.load(f)
    labels={int(o['idx']):o.get('category','unknown') for o in objects}
    categories={}
    for i in unique:categories[labels.get(int(i),'unknown')]=categories.get(labels.get(int(i),'unknown'),0)+1
    row=dict(run=record['run'],scene=record['scene'],feature_weight=record['feature_weight'],status='complete',frames=cfg['data']['num_frames'],gaussians=gaussians,live_instances=len(unique),small_instances_lt150_gaussians=int((counts<150).sum()),median_gaussians_per_instance=float(np.median(counts)) if len(counts) else None,foreground_gaussian_fraction=float((ids>0).mean()),end_to_end_seconds=record['end_to_end_seconds'],dgsg_wall_seconds=perf['dgsg_wall_seconds'],peak_allocated_gib=perf['peak_cuda_allocated_bytes']/1024**3,peak_reserved_gib=perf['peak_cuda_reserved_bytes']/1024**3,peak_rss_gib_sampled=record['peak_rss_bytes_sampled']/1024**3)
    for metric in ['psnr','ssim','lpips','l1','rmse']:
        values=np.atleast_1d(np.loadtxt(r/'eval'/f'{metric}.txt'));row[metric]=float(values.mean());row['evaluated_views']=len(values)
    (r/'instance_statistics.json').write_text(json.dumps(dict(categories=categories,counts_by_id={str(i):int(n) for i,n in zip(unique,counts)},note='Predicted instance statistics only; not accuracy.'),indent=2))
    rows.append(row)
(a.root/'summary.json').write_text(json.dumps(rows,indent=2))
fields=sorted(set().union(*(r.keys() for r in rows)))
with (a.root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
deltas=[]
for scene in ['room0','room1','office0']:
    matches={r['feature_weight']:r for r in rows if r['scene']==scene and r['status']=='complete'}
    if .5 not in matches or 0. not in matches:continue
    on,off=matches[.5],matches[0.]
    deltas.append(dict(scene=scene,**{k:on[k]-off[k] for k in ['psnr','ssim','lpips','l1','live_instances','gaussians','end_to_end_seconds','peak_reserved_gib']}))
(a.root/'deltas_on_minus_off.json').write_text(json.dumps(deltas,indent=2))
lines=['# 三场景实例特征损失消融','','仅特征损失权重 0.5 / 0 不同；固定 seed=0，424×240，stride=10，200 帧，20 次建图迭代，真值位姿。关闭 DAM 和语言模型后处理。','','| 场景 | 特征损失 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | 深度 L1(cm) ↓ | 实例数 | 高斯数 | 总耗时(s) | PyTorch峰值保留显存(GiB) |','|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    if r['status']!='complete':lines.append(f"| {r['scene']} | {r['feature_weight']} | 失败，见日志 | | | | | | | |");continue
    lines.append(f"| {r['scene']} | {'有' if r['feature_weight'] else '无'} | {r['psnr']:.3f} | {r['ssim']:.4f} | {r['lpips']:.4f} | {r['l1']*100:.3f} | {r['live_instances']} | {r['gaussians']} | {r['end_to_end_seconds']:.1f} | {r['peak_reserved_gib']:.3f} |")
lines+=['','## 这个对照检验什么','','有特征损失：L=0.5 L_RGB + 1.0 L_depth + 0.5 L_feature；无特征损失：最后一项权重为 0。L_feature=0.8 L1+0.2(1-SSIM)，作用于低维实例特征渲染图，不是 CLIP 文本向量。两组均保留检测、分割和关联。','','实例特征损失可通过可微渲染同时影响高斯几何、透明度、尺度及实例特征。它是否改善重建与边界必须看实验；不能预先保证每个指标或每个场景都会改善。','','## 解释限制','','- 重建在每场景21个采样训练视角评估，不是独立测试视角；不是论文完整设置或论文数值复现。','- 实例数、高斯支持量、少量高斯实例比例均为预测统计，没有分割真值，数量多或少都不自动代表更准确。请同时查看配对实例渲染。','- 单个随机种子，不足以推断统计显著性。GPU非确定性、桌面负载和温度影响时间与数值。','- 两组均计算特征渲染，关闭组仅将权重设为0；耗时对比不是删除辅助分支后的加速实验。','- 总耗时包含Python启动、模型加载、数据读取、建图、评估和保存。PyTorch显存不含桌面、驱动和非PyTorch分配；内存RSS为2秒采样近似峰值。','- 实例编号在不同运行之间可能变化，相同颜色或相同ID不保证是同一实物。']
lines+=['','## 本次观测','','开启减关闭的 PSNR 差值：'+', '.join(f"{d['scene']}: {d['psnr']:+.3f} dB" for d in deltas)+'。', '以上是本机低分辨率、低迭代预算、单种子结果；如果重建指标下降，应如实记录，不能推断论文方法整体无效，也不能宣称复现了论文改善趋势。', '', '## 查看结果', '', '[指标图](comparison_metrics.png) · [room0 配对视图](room0_paired_views.png) · [room1 配对视图](room1_paired_views.png) · [office0 配对视图](office0_paired_views.png) · [CSV](summary.csv)', '', '配对图按列排列：开启RGB、关闭RGB、开启实例、关闭实例；按行排列采样帧0、100、199。实例颜色不是跨运行匹配的真值身份。']
(a.root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
