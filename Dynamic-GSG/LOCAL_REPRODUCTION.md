# 本机 DynamicGSG 复现

## 2026-10-17：三场景特征损失消融已完成

`room0`、`room1`、`office0` 各完成有／无实例特征损失对照，共 6 次，均正常退出。固定 200 个采样帧、424×240、20 次建图迭代、真值位姿、seed=0；仅改变特征损失权重 0.5／0。

结果位于 `/media/joe/Data/Dynamic-GSG/experiments/feature_ablation_20261017/`，包含六组地图、配置与日志，以及 `REPORT.md`、`summary.csv`、`comparison_metrics.png` 和三场景 `*_paired_views.png`。完整讲解见 [FEATURE_ABLATION.md](FEATURE_ABLATION.md)。

本设置中开启组 PSNR 分别低 0.402／0.528／0.533 dB；没有复现重建质量改善趋势。实例统计与可视化已完成，但没有语义分割真值，不能报告实例准确率。论文完整配置、多种子统计、语义／关系准确率与动态更新仍未完成。下文较早记录保留为历史进展。


## 2026-10-17：已接入本地模型分类（DAM 关闭）

最新流程：已有 YOLO 物体类别 → 本地 Qwen2.5-1.5B-Instruct 分类 → 三维几何层级图 → 关系查询与高亮。

```bash
cd /home/joe/Dynamic-GSG
./run_hierarchy_model.sh
```

模型版已实际跑通，输出 `scene_graph_hierarchy_model.png`／`.json`／`.svg`、`model_classifications.json`、`hierarchy_model_comparison.json`，结果位于原 `room0_local200` 目录。分类为 8 个 Asset、19 个 Ordinary、7 个 Standalone；16 个几何归属候选，3 个归属未确定。原规则版保留，便于比较。

本地模型读取类别文字，**不是图像 LVLM**，也不是论文原模型。DAM 暂不运行。详细配置、缓存、实测显存与限制见 [MODEL_HIERARCHY.md](MODEL_HIERARCHY.md)。后文规则版说明保留为对照。


## 最新进展：本地静态层级图与关系查询

已补齐本地近似链路：**物体高斯地图 → 房间／家具／物体层级图 → 按父子关系查询 → 在高斯渲染中高亮结果**。

本次没有重建或覆盖原始高斯地图、实例文件。新增图使用类别规则替代 LVLM 分类，使用几何候选归属替代经过验证的承载关系；它不是论文完整方法复现。后文“未运行 LLM 层级图”等记录仍然有效。

```bash
cd /home/joe/Dynamic-GSG
./run_hierarchy_local.sh
```

运行会重建本地层级 JSON、PNG、SVG 和实例清单，并演示查询归属于沙发的抱枕及高亮。

结果均在 `/media/joe/Data/Dynamic-GSG/experiments/room0_local200/`：

- `scene_graph_hierarchy_local.png`／`.svg`：可查看的层级图；实线表示房间成员，虚线表示未验证的几何归属，红边为归属未确定。
- `scene_graph_hierarchy_local.json`：1 个房间＋34 个实例，34 条父子边；包含候选父节点、选择分数、几何范围、坐标转换及配置来源。
- `scene_graph_hierarchy_local.md`：实例类别、类型、父节点、状态清单。
- `query_pillows_on_couch_local.json`：示例检索结果，共 7 个实例。
- `query_pillows_on_couch_local_preview.png`：结果在原高斯地图中的红色高亮。

**实现细节与限制**：

1. 使用原始 Replica `traj.txt` 的首个选中位姿，将第一相机坐标系中的地图恢复为 Replica 世界坐标，Z 轴向上，长度单位米。不能直接把相机坐标 Z 轴当作高度；本工具暂只支持 Replica。
2. 类别规则区分 Asset（家具）、Ordinary（普通物体）、Standalone（独立物体）。本次为 8／18／8 个。规则按上游提示词的思路编写，stool 归 Standalone，lamp 归 Ordinary；实际落地灯等情况可能误分类。
3. 对每个实例使用高斯位置的 5%／95% 分位范围降低离群点影响。普通物体中心在家具水平范围内，且底部高度落在家具高度范围附近，才进入候选父节点；水平余量 8 cm，高度余量 15 cm。选择几何分数最小的候选；这不能证明真实接触、物理支撑或 inside/on 关系。
4. 16 个普通物体有候选归属，2 个（#29 pillow、#33 bag）未确定，暂挂房间并标记 `unresolved`。图中的 `attached_to` 不应直接解释为已验证的“在上面”。
5. 输入的 34 个实例不等于 34 个经过真值确认的独立实物，继承上游分割、检测与跨帧关联的错误。
6. 不调用语言服务，不生成 DAM 描述，不执行自由语言推理；查询使用类别／父节点筛选。`sofa` 与 `couch`、`sofa chair` 与 `chair` 支持别名匹配。

查询示例：

```bash
# 找归属于沙发的抱枕候选
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_hierarchy_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --category pillow --parent couch
# 查看某个实例的父节点
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_hierarchy_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --id 36
# 查看具体桌子 #24 的子节点
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_hierarchy_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --parent-id 24
```

如需修正类别／类型，创建独立的 override JSON，例如 `{"8": {"object_type": "Standalone"}}`，然后运行 `scripts/build_hierarchy_local.py <结果目录> --overrides <文件>`，再运行绘图脚本。不要直接修改原始 `objects.pkl.gz`。再次执行一键脚本会按默认规则重建图。

验证：5 项单元测试通过，涵盖支撑候选、悬空／远处物体、独立物体、类型修正、类别别名、唯一父节点和无环；真实地图也验证了节点／边引用、父节点类型和无环。高亮已实际重新渲染验证。测试不代表语义关系准确率。

仍待论文级复现：DAM/LVLM 物体描述与分类、经过评估的层级关系和语言检索、动态地图／场景图更新、多场景定量与消融实验。


项目：`/home/joe/Dynamic-GSG`，上游提交 `73c2dba33fabcf60f91bf4d1b995cdfc55bb33e6`。

- Python：`/home/joe/anaconda3/envs/dgsg/bin/python`；PyTorch 2.3.1+cu121，CUDA 编译器 12.1，GPU RTX 3070 Laptop 8 GB。
- 数据：`/media/joe/Data/SplaTAM/data/Replica`（复用已有数据）；room0 有 2000 组 RGB、深度、位姿。
- 模型、缓存、结果：`/media/joe/Data/Dynamic-GSG/`。运行前确保 Data 盘已经挂载。
- 新增依赖列表 `requirements-local.txt` 修复尾逗号、NumPy/SciPy 版本矛盾，并避免 `clip` 同名包覆盖 OpenAI CLIP。GroundingDINO、RAM、DAM 改为按需导入。

## 运行

在项目根目录执行：

```bash
./run_replica_local.sh 2>&1 | tee logs/room0_smoke.log
# 覆盖完整 room0 轨迹：从 2000 帧中每隔 10 帧取样，共 200 帧
DGSG_NUM_FRAMES=200 DGSG_RUN_NAME=room0_local200 ./run_replica_local.sh 2>&1 | tee logs/room0_local200.log
```

本地配置 `configs/replica/dgsg_local.py` 使用 424×240、真值位姿、不扰动位姿、默认 10 帧、每帧 20 次建图迭代、ViT-B-32 CLIP；关闭动态更新及 DAM/语言服务描述。这些修改用于验证运行链路，**不是论文完整参数或定量结果复现**。物体描述保存检测器类别，不冒充 DAM 描述。

```bash
/home/joe/anaconda3/envs/dgsg/bin/python scripts/export_scene_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_smoke
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_objects_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_smoke 'a chair'
```

导出的 PLY 是高斯中心点云，不是网格；`scene_graph_geometric.json` 是中心距离阈值构成的几何邻接图，**不是论文语言层级场景图**。

## 完整模式的后续条件

完整描述需要安装仓库要求的 describe-anything、RAM（GroundingDINO 路线）、相应权重，并配置语言服务。环境变量 `DGSG_API_KEY` 和 `DGSG_API_BASE` 用于建图后描述的服务；查询服务另用 `DGSG_QUERY_API_KEY`（或 `OPENAI_API_KEY`）及 `DGSG_QUERY_API_BASE`；当前上游描述请求模型为 `qwen2.5-vl-72b-instruct`。不要把密钥写入仓库或聊天。

仅修改 `enable_object_captions=True` 不足以完成完整模式：需恢复 ViT-H-14、论文分辨率和迭代数，评估显存，核对描述服务和查询服务的模型兼容性。

Replica 是静态数据；动态更新需要另外的变化序列。上游公开代码只支持单物体更新。静态 smoke 成功不表示已经复现动态实验。

## 无界面预览

```bash
/home/joe/anaconda3/envs/dgsg/bin/python scripts/render_preview_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --frames 0 50 100 199
```

输出 `reconstruction_instances.png`；左侧是真正的高斯渲染 RGB，右侧按保存的实例 ID 着色，灰色表示背景。

若重启后数据盘未挂载：`udisksctl mount -b /dev/disk/by-uuid/F03AD16D3AD130F8`。此次未修改开机挂载配置。

## 已确认的短序列结果

10 帧 smoke 已完成：188678 个高斯、9 个实例；保存地图、关键帧和实例后，可独立重新加载进行 RGB/实例渲染、点云导出与 CLIP 查询。`a chair` 查询返回 stool 和 sofa chair 等候选。原 smoke 的 `eval_every=100` 只评估首视角，PSNR 32.11 dB 不代表 10 帧平均或论文指标。后续 200 帧运行改为每 10 帧评估。

`logs/environment-freeze.txt`、`logs/conda-explicit.txt` 和 `logs/model-sha256.txt` 记录环境与权重；`logs/room0_local200.log` 为长序列日志。

200 帧结果导出后，可执行 `./view_replica_local.sh` 打开上游交互查看器，`r` 为 RGB、`p` 按实例编号、`c` 为 CLIP。图形交互需桌面显示环境；已提供无需图形窗口的预览/查询命令。LLM 相关按键需要额外服务配置。

## 200 帧本地流程验收（2026-10-16）

- 200/200 帧正常退出；来自 room0 原始 2000 帧、stride=10。
- 保存 755742 个高斯中心、34 个有高斯支持的物体实例；几何邻接图 42 条边。
- 21 个采样训练视角：PSNR 29.0459 dB，MS-SSIM 0.95705，LPIPS 0.10103，深度 L1 0.01329 m。使用真值位姿，ATE=0 不说明定位能力。
- 已重新加载最终文件，完成 RGB/实例渲染、两种 PLY 点云导出、CLIP 文本查询和 JSON 导出。
- `a lamp` 前两位为 lamp；`a chair` 首位误匹配 pillow，其后有 sofa chair。这仅验证检索链路，未做标注检索准确率评估。
- 结果目录：`/media/joe/Data/Dynamic-GSG/experiments/room0_local200`。
- 预览：`reconstruction_instances.png`；几何图：`scene_graph_geometric.json`；检索：`query_lamp.json`、`query_chair.json`；配置快照：`config.resolved.json`、`config_replay.py`。
- 未运行 DAM/LLM 层级图、动态更新、论文完整分辨率或消融实验。
