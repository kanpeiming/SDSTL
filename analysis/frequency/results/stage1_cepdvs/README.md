# CEP-DVS 阶段 1 结果

状态：远程 20 样本 smoke、每类 5 个样本的 full，以及同 100 样本离线 Edge 与配对审计均已下载并核对；修正默认路径后的加载器单批次检查已通过（依据用户提供的终端日志）。远程运行根目录：`/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/`。

[加载器单批次检查记录](loader_smoke_report_20261014.md)：终端输出表明三种数据各 10,000 个样本、迁移划分为 9,000/1,000、首批 Edge/DVS 标签一致；尚未取得远程 `check.log` 和实际 run ID。

| Run ID | 远程目录 | 样本数 | 原始事件 | 核对状态 |
|---|---|---:|---|---|
| [`smoke_20261014T072630Z`](smoke_20261014T072630Z/analysis.md) | `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/smoke_20261014T072630Z/` | 20；每类 1 | 20/20 可重建 | 输入诊断已核对；迁移结论待验证 |
| [`full_20261014T102251Z`](full_20261014T102251Z/analysis.md) | `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/full_20261014T102251Z/` | 100；每类 5 | 100/100 可重建 | 输入诊断已核对；离线 Edge 见下一行，模型层待验证 |
| [`edge_pair_audit_20261015T050226Z`](edge_pair_audit_20261015T050226Z/analysis.md) | `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/edge_pair_audit_20261015T050226Z/` | 100 个离线 Edge；9,000 个训练配对 | 不适用 | 当前路径、配对和离线输入已核对；历史模型待验证 |

分析顺序：先检查 `run.json` 的样本清单和代码指纹，再读 `summary.md` 与逐样本 CSV；核对重建误差和异常样本图像后，才写关于输入频谱、极性与时间分箱的结论。
