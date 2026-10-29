# 频域分析结果索引

原始运行结果统一放在远程 `/home/user/kpm/kpm/results/SDSTL/frequency/`。每次运行创建独立子目录，不覆盖旧实验，也不写入数据集目录。这里保留少量审阅用结论和远程运行索引。

- [stage1_cepdvs/](stage1_cepdvs/README.md)：CEP-DVS 的 RGB、双 Sobel 与事件输入分析。
- [edge_pair_audit_20261014T081612Z/](edge_pair_audit_20261014T081612Z/analysis.md)：迁移训练索引配对与离线 Edge 输入核对。
- [checkpoint_inventory/](checkpoint_inventory/analysis.md)：下载的 CEP-DVS baseline 权重元数据、候选选择与局限；大文件 `.pth` 保留本地但不纳入版本控制。
- [stage2_baseline/](stage2_baseline/README.md)：CEP-DVS 当前测试集 baseline 推理与后续层间分析结果。

只有取得对应实验的 `run.json`、摘要和必要的逐样本数据并完成核对后，才在此记录数据结论；图像仅在该实验产生图像时核对。合成数据测试只用于验证脚本，不作为研究结果。
