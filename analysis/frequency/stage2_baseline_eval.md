# 阶段 2 前置：当前 CEP-DVS 测试集 baseline 推理

这一步只用现有 `SDSTL_...namebaseline.pth` 权重，对当前 `/home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed` 中按加载器排序后的末 1,000 个样本做完整推理。运行在远程服务器；输出独立存放于 `/home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline/eval_<UTC>/`。不训练、不改权重、不写数据集。

本地已通过 Python 语法、命令行帮助和 Bash 语法检查；缺少本地 CEP-DVS 数据，真实推理需在远程完成。当前模型的本地严格加载和零输入前向检查见[checkpoint 清单](results/checkpoint_inventory/analysis.md)。

本地 PowerShell 中连接 SFTP，上传本次两个新文件：

```text
sftp -P 56089 root@10.8.135.1
sftp> cd /home/user/kpm/kpm/ESVAE/ESVAE/analysis/frequency
sftp> put C:/coding/SDSTL/analysis/frequency/eval_cepdvs_baseline.py
sftp> put C:/coding/SDSTL/analysis/frequency/run_eval_cepdvs_baseline.sh
sftp> bye
```

在远程 SSH 终端执行：

```bash
cd /home/user/kpm/kpm/ESVAE/ESVAE
bash analysis/frequency/run_eval_cepdvs_baseline.sh
```

脚本运行前会验证项目、数据、权重和结果目录的解析路径；模型按当前 `VGGSNN` 严格加载。输出包含 `summary.md`、`run.json`、`predictions.csv`、`per_class.csv` 和 `console.log`。成功时终端最后打印 `Results:` 目录；若失败，保留该目录的 `console.log` 用于诊断。

回到本地 PowerShell，使用脚本打印的实际 `eval_<UTC>` 名称拉回结果：

```text
sftp -P 56089 root@10.8.135.1
sftp> lcd C:/coding/SDSTL/analysis/frequency/results/stage2_baseline
sftp> get -r /home/user/kpm/kpm/results/SDSTL/frequency/stage2_baseline/<实际 eval_ 目录名>
sftp> bye
```

这次推理使用测试集，只用于核验 checkpoint 在当前输入管线的预测，不用于挑选频带、调模型或选择权重。待结果核对后，再固定同一模型和样本设计层间采集。
