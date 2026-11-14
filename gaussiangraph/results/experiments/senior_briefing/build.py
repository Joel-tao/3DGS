from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from PIL import Image
B=Path('/media/joe/Data/gaussiangraph/experiments');out=B/'senior_briefing';font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
def page(title,lines=None,image=None,note=None):
 fig=plt.figure(figsize=(16,9),facecolor='white');fig.text(.05,.93,title,fontproperties=font,fontsize=24)
 if image:
  ax=fig.add_axes([.04,.14,.92,.72]);ax.imshow(Image.open(image));ax.axis('off')
 if lines:
  y=.82
  for line in lines:
   fig.text(.06,y,line,fontproperties=font,fontsize=17,va='top',linespacing=1.6);y-=.12
 if note:fig.text(.05,.055,note,fontproperties=font,fontsize=12,color='#526171')
 pdf.savefig(fig);plt.close(fig)
with PdfPages(out/'ramen_experiment_briefing.pdf') as pdf:
 page('GaussianGraph：ramen 单场景复现与诊断',[
 '目标：跑通三维开放词汇分割，并核对 GaussianGraph Table 1 的方法与评估。',
 '范围：仅 LERF ramen；不是完整四场景复现，也尚未完成 scene graph。',
 '数据：131 张图像及相机参数；114 张训练、17 张测试；分割评估 7 视角、71 项。',
 '注意：分割评估视角与重建测试划分不是一回事；当前评估不是作者完整脚本。',
 '已完成：3DGS、三种对照方法、独立 CF/SAM2/置信度版本，以及多组控制变量实验。'])
 page('1. 当前流程：从重建到文本查询分割',[
 'RGB＋相机参数 → 3DGS 重建 → 每个高斯学习实例特征',
 '二维掩码监督 → Control / Follow / 细分 → 高斯簇',
 '渲染簇并匹配二维掩码 → 跨视角聚合 CLIP → 文本类别选择高斯',
 '渲染预测掩码 → 与 GT 比较 → mIoU、Acc@0.25、Acc@0.5',
 '尚未完成：LLaVA＋Grounding DINO 前景流程、属性/关系生成、场景图及关系推理。'])
 page('2. 我们实际训练的 3DGS 重建',image=B/'gs_gallery/ramen_3dgs.png',note='左：未参与重建训练的照片；中：3DGS 渲染；右：相对深度。约 71.4 万高斯；重建好不等于分割好。')
 page('3. 我们实际生成的分割结果',image=B/'segmentation_gallery/preview.png',note='绿色为选中区域。固定 frame_00006 的碗、筷子、鸡蛋、海苔示例；完整 71 项另附 PDF。两列 CF 的 checkpoint 不同。')
 page('4. 当前本地指标：独立评估协议',[
 '方法                                  Acc@0.25        Acc@0.5          mIoU（%）',
 'LangSplat 多层汇总                  30.99              19.72              21.81',
 'LEGaussian argmax                   9.86                0.00                8.98',
 'OpenGaussian                         25.35              16.90              20.48',
 'SAM2＋CF，关闭伪特征筛选        32.39              16.90              17.66',
 'SAM2＋CF，关闭最后细分           28.17              18.31              18.34'],note='簇方法使用至少 5 个匹配视角；末行使用 67999 步；其余模型设置不同。不能据此宣称复现论文排名。')
 page('5. 哪些改动经过实验验证？',[
 '关联筛选：固定纯 IoU，关闭伪特征筛选，SAM2＋CF mIoU 7.70 → 17.66。',
 '归一化聚类重训：mIoU 17.66 → 14.89；本次设置下没有提升。',
 'CLIP 区域对齐：在归一化模型上，14.89 → 14.39；没有提升。',
 'LEGaussian softmax 解码：8.98 → 10.47；提升有限，尚未解决基线差距。',
 '以上属于不同控制变量实验，不能拼成一条持续提升的训练曲线。'])
 page('6. 最严格的细分开关对照：同一份 67999 步模型',[
 '固定实例特征、几何、CLIP、关联与评估；只改变父簇 / 细分后标签。',
 '阶段统计：Control 205 簇（5000 控制点）→ Follow 3348 → 细分 5263。',
 '关闭细分：非零 CLIP 覆盖 65.77%；mIoU 18.34%；Acc@0.25 28.17%。',
 '开启细分：非零 CLIP 覆盖 40.02%；mIoU 17.66%；Acc@0.25 32.39%。',
 '结论：细分显著影响覆盖率，但 mIoU 只差约 0.68 个百分点，且并非所有指标改善。'],note='mIoU/准确率使用至少 5 视角；非零 CLIP 覆盖率统计所有获得特征的点，不是类别正确率。')
 page('7. 当前限制与下一步',[
 '核心限制：方法与评估仍有独立实现选择，尚不能称 GaussianGraph 复现成功。',
 'LangSplat 偏高、LEGaussian 偏低：优先核对各方法解码与评估，而非继续调 CF。',
 '检查逐高斯归一化、层选择、背景拒识、单独渲染对象的遮挡处理及 GT 计算。',
 '后续候选：固定簇的包含比例关联；固定特征重新聚类，逐项验证。',
 '本地已有真实重建图、逐查询掩码、checkpoint、配置和实验日志，可追溯。'])
print(out/'ramen_experiment_briefing.pdf')
