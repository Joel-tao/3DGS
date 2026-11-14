# Dynamic-GSG：本地复现与成果

源码来自 [GeLuzhou/Dynamic-GSG](https://github.com/GeLuzhou/Dynamic-GSG)，版本见 UPSTREAM.json。本目录保留本地代码修改、配置、日志和实际实验报告。

- [200帧复现说明](LOCAL_REPRODUCTION.md)：room0、755742个高斯中心、34个实例；使用真值位姿，不代表已验证定位能力。
- [模型辅助层级图](MODEL_HIERARCHY.md)：类别文字分类与几何归属近似，不是完整论文LVLM/DAM流程。
- [三场景消融](FEATURE_ABLATION.md)：六次运行；本地设置下未复现开启实例特征损失带来的重建质量改善。
- [结果目录](results/)：指标、配置、层级图、查询结果与组合可视化。

大型地图、关键帧、模型权重留在本地，详见仓库根目录清单。`DEPENDENCIES.json` 记录下载依赖版本；不包含安装环境和编译产物。原始子模块不随此快照自动初始化，请按对应地址/版本恢复。
