# 模型分类与静态层级图（不运行 DAM）

本流程使用本地 **Qwen2.5-1.5B-Instruct 文本语言模型**读取已有 YOLO 类别，按仓库的 `configs/prompts/parsing_objects_caption.txt` 判断 Asset／Ordinary／Standalone，然后结合三维几何建立层级图。

它不读取物体图像，不生成颜色／材质描述，不是视觉语言模型推理，也不是论文 GPT-4o 或上游 Qwen2.5-VL-72B 的等价复现。模型来源：https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct 。

## 运行

```bash
cd /home/joe/Dynamic-GSG
./run_hierarchy_model.sh
```

默认读取 `/media/joe/Data/Dynamic-GSG/experiments/room0_local200/`，顺序执行：类别模型分类（单独占用 GPU）→ 图构建 → 绘图 → 查询沙发的抱枕候选 → 在高斯地图中高亮 → 与规则版比较。

模型权重位于 Data 盘的 `Dynamic-GSG/models/Qwen2.5-1.5B-Instruct/`。首次运行后，分类按类别缓存；相同类别复用一次模型推理结果，不代表模型依据每个实例的外观作出独立判断。改变类别、模型版本或提示词会使相应缓存失效。已有有效缓存时无需重新加载分类模型。

## 输出

- `model_classifications.json`：每个实例的模型类型、模型版本、原始回答、提示词哈希和是否使用 DAM。
- `model_type_cache.json`：类别级可恢复缓存；无效 JSON 会停止并另存失败记录，不回退到手写规则。
- `scene_graph_hierarchy_model.json`／`.md`／`.png`／`.svg`：模型版层级图；与原 `scene_graph_hierarchy_local.*` 分开保存。
- `query_pillows_on_couch_model.json`、`query_pillows_on_couch_model_preview.png`：模型版图查询与高亮。
- `hierarchy_model_comparison.json`：与规则版的类型和父节点差异。差异多少均不代表准确率。

## 查询

```bash
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_hierarchy_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --graph-stem scene_graph_hierarchy_model --category pillow --parent couch
/home/joe/anaconda3/envs/dgsg/bin/python scripts/query_hierarchy_local.py /media/joe/Data/Dynamic-GSG/experiments/room0_local200 --graph-stem scene_graph_hierarchy_model --parent-id 24
```

## 保留的局限

模型只替代类别到物体类型的手写映射。连边仍依据 Z 轴向上的 Replica 世界坐标、实例位置分位范围、水平覆盖和高度条件推断，`attached_to` 是未验证的几何归属候选。模型可能误分类；分割、关联错误也会传递到图中。

这是静态后处理流程，尚未实现流式图更新、动态物体变化维护、自由语言关系推理或论文级准确率评估。DAM 暂不安装和运行；以前准备的源码、空环境不参与本流程。

## 本机实测（2026-10-17）

- 模型版本：`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；完整权重大小及 SHA256 已核对模型仓库清单。
- 实际推理 16 个类别，覆盖 34 个实例；Asset=8、Ordinary=19、Standalone=7。
- 模型版图包含 1 个房间、34 个实例、34 条父子边，其中 16 条为几何归属候选，3 个普通物体归属未确定。
- PyTorch 峰值分配显存 2.923 GiB，峰值保留显存 3.105 GiB；不包括桌面及其他进程的显存，未与建图并行测试。
- 与规则版相比，仅 #18 `picture` 从 Standalone 变成 Ordinary，父节点未变化，归属状态变为 unresolved。保留模型原始判断，未人工覆盖；这不是模型比规则更准确的证据。
- 模型图查询“归属于沙发的抱枕”返回 7 个实例，已在原始高斯地图重渲染高亮。
- 8 项单元测试通过；真实图已检查分类来源、实例覆盖、唯一父节点和无环。再次执行分类时全部命中缓存，不加载模型；缓存重放保留首次实测显存数据。
- DAM 未运行，未生成详细视觉描述；原始高斯地图、实例文件和规则版图保留。
