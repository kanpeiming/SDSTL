# 阶段 2：CEP-DVS baseline 早层频谱探针

已在当前 1,000 个测试样本上复测 baseline 权重，准确率为 331/1000，与保存轮次日志的验证准确率一致；[本地记录](results/stage2_baseline/eval_20261015T053933Z/analysis.md)。下一步使用阶段 1 已固定的训练集清单，每类取第一个 ID，共 20 个样本。此次只做推理和观测，不训练、不修改数据或模型参数。

采集点为输入、`dvs_input`、`features[0]`、`features[2]`、`features[3]` 的 Conv/BN/放电前膜电位/脉冲，以及共享池化模块的第 1、2 次调用输出。膜电位由 BN 输出和实际脉冲按本项目 LIF 递推式重建，记录脉冲不一致个数；若非零，先检查重建值，不解释膜电位频谱。每样本分别计算空间去均值 FFT 的低/中/高频能量占比，同时记录 RMS、去均值 RMS、活跃比例、脉冲率和零空间能量比例。不同空间分辨率用相同归一化频率半径描述，频带占比变化不能直接称为信息增益或分类因果证据。

本地已用随机输入验证采集前后 logits 完全一致、19 个阶段均被捕获、膜电位递推与实际脉冲一致；实际 CEP-DVS 数据仍需在远程运行。把以下两个文件上传到远程现有 `analysis/frequency/` 目录（在本地 PowerShell 执行，密码在交互提示中输入）：

```text
sftp -P 56089 root@10.8.135.1
sftp> cd /home/user/kpm/kpm/ESVAE/ESVAE/analysis/frequency
sftp> put C:/coding/SDSTL/analysis/frequency/probe_cepdvs_early_layers.py
sftp> put C:/coding/SDSTL/analysis/frequency/run_probe_cepdvs_early_layers.sh
sftp> bye
```

在远程 SSH 终端运行：

```bash
cd /home/user/kpm/kpm/ESVAE/ESVAE
bash analysis/frequency/run_probe_cepdvs_early_layers.sh
```

脚本会打印 `Results: /home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline/probe_<UTC>`。输出在独立目录：`summary.md`、`per_sample_stage.csv`、`predictions.csv`、`run.json`、`console.log`。按脚本打印的真实目录名拉回：

```text
sftp -P 56089 root@10.8.135.1
sftp> lcd C:/coding/SDSTL/analysis/frequency/results/stage2_baseline
sftp> get -r /home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline/probe_<实际时间戳>
sftp> bye
```

只需拉回这个 `probe_` 目录即可分析；此前 `eval_` 结果的终端摘要已经足够决定本步。若运行失败，保留并回传 `console.log` 的报错，不必重新执行阶段 1。
