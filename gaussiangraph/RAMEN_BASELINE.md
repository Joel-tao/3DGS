# ramen OpenGaussian 基线

目标：先验证 OpenGaussian 的训练、文本选择和 LERF 评估链路。
本实验尚不包含 GaussianGraph Control-Follow，不能称为 GaussianGraph Table 1 已复现。

## 目录与环境

- 代码：`/home/joe/gaussiangraph/baselines/OpenGaussian`
- 数据、环境、构建与结果：`/media/joe/Data/gaussiangraph`
- Python：`/media/joe/Data/gaussiangraph/envs/opengs/bin/python`
- 环境通过 `venv --system-site-packages` 复用 `/home/joe/anaconda3/envs/dgsg` 的
  Python 3.10 / PyTorch 2.3.1+cu121；依赖原环境存在。
- GPU：RTX 3070 Laptop 8GB，CUDA 构建架构 8.6。
- Data 分区已通过 udisksctl 挂载。本次未修改系统开机挂载设置。

## 数据来源和实验限制

1. 图像、COLMAP 与多边形 GT：LangSplat 作者发布的
   <https://huggingface.co/datasets/MinghanQin/lerf_ovs>。
   官方 Google Drive 链接也可下载同名归档。
2. OpenGaussian README 中的 OneDrive 预处理数据链接本次访问失败。
   先采用第三方预计算特征验证基线：
   <https://huggingface.co/datasets/JUNHAKBAE/THGS-lerf-ovs-language-features>。
   下载脚本固定 HF revision，并校验每个文件的 SHA256；记录在
   `data/lerf_ovs/feature_source.json`。发布脚本使用 LangSplat SAM ViT-H、
   OpenCLIP ViT-B-16 / laion2b_s34b_b88k。
   **这些不是 OpenGaussian 作者的原始预处理文件，也不是 GaussianGraph 的 SAM2_L 输入。**
3. 只使用 level 3 掩码。图像与掩码尺寸经检查后才能启动训练。
4. GT 来自 7 个标注视角，按 LangSplat 的 `cv2.fillPoly` 和同类别合并规则
   转为 71 个二值掩码。JPEG 保存与 OpenGaussian 原评估读取方式一致。
5. 使用 OpenGaussian 的选择与评估方式，不能据此保证与 GaussianGraph
   Table 1 的未公开评估细节完全一致。指标保留 upstream 的 `IoU > 阈值`，
   缺失预测计 0，最终 JSON 使用 0–1 单位。

## 运行

```bash
cd /home/joe/gaussiangraph/baselines/OpenGaussian
bash scripts/run_ramen.sh check
bash scripts/run_ramen.sh train
bash scripts/run_ramen.sh render
bash scripts/run_ramen.sh eval
```

`train` 保留官方 ramen 参数：70000 步，阶段边界 30000/40000/50000，
64 个粗簇、每粗簇 10 个细簇、pos_weight=0.5、loss_weight=0.01。
额外使用 `--save_memory --data_device cpu`，保持默认图像分辨率。
如果显存仍不足，需要另建输出目录运行低分辨率实验并明确记录变更。
`RAMEN_DATA`、`RAMEN_MODEL`、`RAMEN_GT` 和 `OPENGS_PYTHON` 可覆盖路径。
训练会保存上游 commit、补丁、依赖列表、完整命令和特征来源。

## 本地修正

- `scene/cameras.py`：根据数据实际设备初始化 `data_on_gpu`，让 CPU 存储模式
  在训练时正确转移相机张量。
- `train.py`：启用 save_memory 时，每轮结束释放该视角的 GPU 数据，涵盖阶段 0/1。
- `compute_lerf_iou.py`：GT/预测路径可通过命令行指定，保存 metrics JSON，
  空 GT 报错。保留原有阈值及聚合方式。
- 渲染器延迟加载 PyTorch3D，仅文本选择后的 KNN 后处理需要它；普通训练
  可以在该依赖编译期间开始。
- 新增单场景启动、输入检查、特征下载和 GT 转换脚本。

## 已完成验证

- CUDA 渲染器编译安装且导入成功。
- PyTorch3D 0.7.7 编译安装成功，CUDA KNN 计算验证通过。
- 完整输入检查通过：131 张图像、262 个特征文件、全部评估 GT 与运行依赖。
- GT 转换得到 71 项；GT 自比较的 mIoU 和两个 Acc 均为 1，
  仅用于验证评估程序，**不是模型实验结果**。
- 启动脚本通过 bash 语法检查；Python 修改通过语法编译检查。
- 20 步 RGB 重建检查通过，模型和 checkpoint 成功保存。
- 40 步短流程检查通过（第 11 步进入连续实例特征学习），确认掩码监督、
  六维特征渲染与反向传播能执行。它不用于报告分割效果。

## 本次完整训练

70000 步完整训练已结束；最终关联显存错误修复后，文本选择、渲染与评估均已完成。
进度日志：`/media/joe/Data/gaussiangraph/logs/ramen-baseline-train.log`。
后续进程 `scripts/finish_ramen.py` 等待训练成功后自动渲染并评估；任一步
失败则将状态写为 failed，不生成虚假的成功结果。
状态文件：`/media/joe/Data/gaussiangraph/logs/ramen-status.json`。
render/eval 日志在同目录，最终指标为 `outputs/ramen_baseline/metrics.json`。

训练是否成功以训练日志、`cluster_lang.npz`、最终 codebook 和
`outputs/ramen_baseline/metrics.json` 为准。

## 2026-10-29：最终关联阶段显存修复

70000 步训练已结束并保存 checkpoint、PLY 和两级 codebook。首次执行最终
语言关联时，`mask_feature_mean` 分配完整的 `[掩码数, 6, H, W]` 中间张量，
在 mapping 阶段发生 CUDA OOM，原自动评估流程因此停止。

修复将掩码均值与方差按每批 2 个掩码归约，去掉仅供可视化使用的全掩码
特征展开。统计值及梯度通过 CPU/CUDA 对照测试；原分辨率 64 掩码方差计算
额外峰值显存约 147MiB。测试命令：

```bash
/media/joe/Data/gaussiangraph/envs/opengs/bin/python scripts/test_mask_feature_mean.py
```

无需重训。`associate_ramen.py` 从最终 checkpoint 恢复连续与量化特征，
不加载优化器，只重跑语言关联，保持原分辨率、训练视角划分及匹配参数：

```bash
/media/joe/Data/gaussiangraph/envs/opengs/bin/python -u associate_ramen.py \
  -s /media/joe/Data/gaussiangraph/data/lerf_ovs/ramen \
  -m /media/joe/Data/gaussiangraph/outputs/ramen_baseline \
  --data_device cpu --eval
```

恢复日志：`logs/ramen-association-recovery.log`。关联成功后再执行 render/eval。


## 本次基线结果（已完成）

- 评估项：71；其中 14 项无输出预测文件，按 upstream 规则计 IoU=0，全部计入分母。
- Acc@0.25：32.39%（23/71）
- Acc@0.5：9.86%（7/71）
- mIoU：17.60%
- 模型与原始数值：`outputs/ramen_baseline/metrics.json`。
- 预测二值掩码：`outputs/ramen_baseline/text2obj/ours_70000/renders_cluster_silhouette/`。

初次训练日志末尾的 OOM 是历史失败记录，已保留；后续恢复、渲染和评估日志
单独保存，当前状态见 `logs/ramen-status.json`（complete）。原渲染器在目标簇被
可见性或后处理筛掉、没有渲染对象时不写文件；本次保留其行为并显式统计缺失数。
恢复补丁和源码快照保存在 `association_recovery_changes.patch` 与
`association_recovery_source.tar.gz`。结果属于使用第三方预计算特征的
OpenGaussian 单场景基线，尚未达到论文原数值，也未加入 Control-Follow。


## 后续进展

基线漏检审计与 Control-Follow 实现、独立参数及当前训练状态，见 `CONTROL_FOLLOW.md`。原始基线模型和 metrics.json 保持不变。
