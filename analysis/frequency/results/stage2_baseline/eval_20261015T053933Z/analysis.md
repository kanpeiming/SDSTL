# CEP-DVS 当前测试集 baseline 复测

2026-10-15。依据用户提供的远程终端输出；原始 `run.json`、`predictions.csv` 和 `per_class.csv` 尚未拉回本地，本页不声称已逐条复核预测。

- 权重：`SDSTL_CEP-DVS_TET_img48_T10_seed1000_dvs1_enc-TET_lr0.001_VGGSNN_namebaseline.pth`。
- 当前数据根：`/home/user/Datasets/kpm-own/CEP-DVS`；预处理 DVS 10,000 个样本，测试集为排序后的索引 9000–9999，共 1,000 个。
- 远程结果目录：`/home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline/eval_20261015T053933Z`。
- 终端报告：`331/1000 = 0.3310`，设备 `cuda:0`；没有训练或参数更新。

该结果与该权重保存时第 22 轮 TensorBoard 的 `val/accuracy = 0.331` 一致。这为当前数据管线和权重的组合提供了可用的推理基线，但不能单凭相同准确率证明历史训练和当前数据版本完全相同。80 轮末日志的 `test/accuracy = 0.340` 属于另一个模型状态，不能拿来否定这次结果。

下一步固定阶段 1 已审计的训练样本和这份权重，采集前两次池化附近的层间频谱、幅度、发放率。20 个样本各类一个，只用于定位现象，不据此估计模型总体准确率或挑选频带。
