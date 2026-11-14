from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
B=Path('/media/joe/Data/gaussiangraph');out=B/'experiments/table1_snapshot'
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
keys=['Acc@0.25','Acc@0.5','mIoU']
refs=[['LangSplat',11.09,7.25,8.94],['LEGaussian',31.54,26.76,15.79],['OpenGaussian',40.85,21.13,25.41],['GaussianGraph（论文）',43.66,28.63,29.58]]
spec=[('LangSplat（多层汇总）','langsplat_3d_multilevel_v1/metrics.json'),('LEGaussian（argmax）','legaussian_3d_v1/metrics.json'),('OpenGaussian','opengs_3d_v1/metrics.json'),('独立 SAM2＋CF：关闭伪特征筛选','association_audit_v1/sam2_cf_iou_unfiltered/eval_occ5/metrics.json'),('独立 SAM2＋CF：关闭细分¹','refinement_ablation_v1/no_refine/eval_occ5/metrics.json')]
local=[]
for name,path in spec:
 m=json.loads((B/'experiments'/path).read_text());local.append([name]+[100*m[k] for k in keys])
fig=plt.figure(figsize=(13,9),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.set_axis_off()
def txt(x,y,s,size=13,**kw):ax.text(x,y,s,fontproperties=font,fontsize=size,transform=ax.transAxes,**kw)
txt(.05,.955,'TABLE 1  |  ramen 单场景结果',22,weight='bold')
txt(.05,.912,'指标单位：%    ↑ 越高越好    ·    当前结果快照：2026-10-31',12,color='#536171')
def panel(y,title,rows,h):
 txt(.05,y,title,15,weight='bold')
 table=ax.table(cellText=[[r[0]]+[f'{v:.2f}' for v in r[1:]] for r in rows],colLabels=['Method','Acc@0.25 ↑','Acc@0.5 ↑','mIoU ↑'],colWidths=[.58,.14,.14,.14],bbox=[.05,y-h-.022,.90,h],cellLoc='center',edges='horizontal')
 table.auto_set_font_size(False);table.set_fontsize(12)
 for (i,j),c in table.get_celld().items():
  c.get_text().set_fontproperties(font);c.get_text().set_fontsize(12);c.set_edgecolor('#c4ccd5');c.set_linewidth(.6)
  if j==0:c.get_text().set_ha('left');c.PAD=.025
  if i==0:c.get_text().set_weight('bold');c.set_linewidth(1.2)
  if i>0 and j>0 and rows[i-1][j]==max(r[j] for r in rows):c.get_text().set_weight('bold');c.get_text().set_color('#146b65')
 return table
panel(.855,'A. 论文 Table 1 参考值',refs,.255)
panel(.525,'B. 本地实验：独立评估协议（不能视为已复现论文）',local,.305)
txt(.05,.145,'¹ 关闭细分使用第 67999 步固定特征；其同 checkpoint 的开启细分对照仍在运行。',11,color='#536171')
txt(.05,.112,'本地均评估 71 项；簇方法采用至少 5 个匹配视角。后两行为诊断变体，不是作者方法的完整复现。',11,color='#536171')
txt(.05,.079,'两个区域内分别标出最高值；论文与本地在训练、关联及评估细节上仍有差异，不直接判定优劣。',11,color='#536171')
txt(.05,.041,'论文来源：arxiv.org/html/2503.04034v1    |    本地来源：各实验 metrics.json',10,color='#687484')
fig.savefig(out/'table1_ramen.png',dpi=180,facecolor='white');fig.savefig(out/'table1_ramen.pdf',facecolor='white');plt.close(fig)
with (out/'table1_ramen.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['source','method']+keys)
 for source,rows in [('paper',refs),('local',local)]:
  for r in rows:w.writerow([source]+r)
(out/'sources.json').write_text(json.dumps(dict(paper='https://arxiv.org/html/2503.04034v1',local=dict(spec)),ensure_ascii=False,indent=2))
print(out/'table1_ramen.png')
