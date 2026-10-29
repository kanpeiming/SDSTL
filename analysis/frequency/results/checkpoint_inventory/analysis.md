# CEP-DVS baseline checkpoint 本地清单

2026-10-15。输入为从远程 `/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints_CEP-DVS` 下载的 5 个 `.pth`。使用本地 PyTorch 2.5.1 的 `torch.load(..., map_location='cpu', weights_only=True)` 只读提取元数据；完整机器可读结果与各文件 SHA-256 在 [`baseline_metadata.json`](baseline_metadata.json)。文件没有被修改或执行训练。本报告审查 checkpoint 可用性和候选身份，并对首选模型做了本地严格加载及零输入前向检查；尚未核对真实运行日志或实际数据预测。

| 文件（省略公共前缀/后缀） | 保存 epoch 字段 | `pretrained_path` | `best_test_acc` 字段 | 初步判断 |
|---|---:|---|---:|---|
| `Baseline_...epoch50_ft` | 20 | CEP-DVS T10 Edge 预训练路径 | 1.0000 | 微调候选；预训练文件是否实际加载待日志核实 |
| `Baseline_...epoch80_baseline` | 34 | CEP-DVS T2 Edge 预训练路径 | 1.0000 | 名称含 baseline，但不能视为已证明的从零训练 |
| `Baseline_...epoch80_ft` | 5 | CEP-DVS T10 Edge 预训练路径 | 0.7336 | 微调候选；与 `epoch50_ft` 为不同训练记录 |
| `SDSTL_...namebaseline` | 22 | `null` | 1.0000 | **阶段 2 的首选纯 DVS baseline 候选** |
| `SDSTL_...nameft` | 18 | CEP-DVS T10 Edge 预训练路径 | 1.0000 | 微调候选；需核实预训练与载入范围 |

五份文件均约 39.75 MiB，SHA-256 各不相同；均含 `net` 模型状态、60 个张量、约 1,041.6 万个张量元素。首层 `dvs_input.fwd.module.0.weight` 为 `[64,2,3,3]`，分类层 `classifier.module.weight` 为 `[20,256]`；元数据均指 CEP-DVS、48×48、T=10、seed=1000、DVS ratio=1、`vgg16`/TET。首选 `SDSTL_...namebaseline.pth` 已在本地按当前 `VGGSNN(cls_num=20, img_shape=48, use_event_attention=False)` **严格载入，60 个键全部匹配**；零输入 `[1,10,2,48,48]` 前向输出 `[1,10,20]` 且全为有限值。这只证明结构与基本前向兼容，不证明当前真实 DVS 上的准确率或历史数据来源。

`epoch` 字段是权重保存时的 epoch 索引，不是文件名中 `epoch50/80` 所指的计划总训练轮数。路径字段也只记录训练时指定的路径，不证明该文件实际存在或被成功加载。尤其 `Baseline_...epoch80_baseline` 保存了非空 T2 预训练路径；当前 `baseline.py` 在该路径存在时会加载匹配参数，因此此文件不能直接作为从零训练对照。

**不要把 `best_test_acc` 当成测试准确率。** 当前 `tl_utils/trainer.py` 的 `save_model()` 将 `self.best_train_acc` 写入名为 `best_test_acc` 的字段；`BaselineTrainer` 在没有验证集时按训练准确率保存模型。四个文件中出现的 `1.0` 因此不能证明测试集达到 100%。历史源码版本和日志尚未核实，稳妥的表述是这个字段不具备独立测试精度证据。

## TensorBoard 训练记录

从远程 `results/SDSTL/baseline/log_dir_CEP-DVS` 下载的 12 个事件文件已在本地逐文件解析，机器可读摘要见 [`tensorboard_summary.json`](tensorboard_summary.json)。`SDSTL_...namebaseline` 目录中有 1 个包含 80 轮完整曲线的文件，另 3 个文件只有事件头；`SDSTL_...nameft` 目录中有 1 个完整文件。事件目录名与权重名相同，权重的 epoch 字段恰好是对应曲线**首次达到训练准确率 1.0** 的轮次，因而日志与当前“按最佳训练准确率保存”机制相符，但仅凭目录名不能证明精确的历史源码或数据版本。

| 日志对应模型 | 权重保存轮次的 `train/accuracy` | 同轮 `val/accuracy` | 80 轮末 `val/accuracy` | 末轮单次 `test/accuracy` | 全程最高 `val/accuracy` |
|---|---:|---:|---:|---:|---:|
| `SDSTL_...namebaseline` | 第 22 轮：1.000 | 第 22 轮：0.331 | 0.340 | 0.340 | 0.355（第 65 轮） |
| `SDSTL_...nameft` | 第 18 轮：1.000 | 第 18 轮：0.411 | 0.399 | 0.399 | 0.422（第 11 轮） |

这些是 TensorBoard 记录值。**末轮 `test/accuracy` 并非保存于第 22/18 轮的 checkpoint 重新评估结果**；保存轮次的 `val/accuracy` 也只能称为日志验证指标。两组日志中末轮 `val/accuracy` 与最终 `test/accuracy` 完全相同，与复用同一批评估数据相符，但没有原始启动命令或历史源码，不能确认其划分关系。微调日志更高也不能直接归因于预训练：其权重确实记录非空预训练路径，但是否成功加载及加载哪些模块仍待历史日志核实。

首选 checkpoint 内 `data_dir=/data/zhan/Event_Camera_Datasets` 只是保存的通用参数；当前 CEP-DVS 分支不使用该参数指定数据根。日志只包含指标曲线，没有记录实际 `pathFile.csv` 或 DVS 目录。因此历史数据来源仍是关键未决项。

后续已在远程使用**当前明确的数据根**和该 baseline 权重做 1,000 个测试样本的只读前向评估，终端报告为 331/1000，恰与保存轮次验证准确率 0.331 相同；见[阶段 2 复测记录](../stage2_baseline/eval_20261015T053933Z/analysis.md)。现进入同一模型前两次下采样附近的频谱、幅度及发放率采集。
