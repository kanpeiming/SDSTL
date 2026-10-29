# 阶段 1 补充：CEP-DVS 迁移配对与离线 Edge 审计

目标：按 `get_edge2dvs_cepdvs` 的排序、Edge 长度划分、前缀采样和位置/取模规则，核对实际训练配对；在阶段 1 smoke 的相同 20 个样本上统计离线 Edge `.pt`。该脚本不训练模型，不修改数据集或现有结果。

## 1. 远程预检（VS Code 集成终端）

手动登录，密码仅输入交互提示：

```bash
ssh root@10.8.135.1 -p 56089
cd /home/user/kpm/kpm/ESVAE/ESVAE
pwd -P
test -f baseline.py && test -f analysis/frequency/spectrum.py && echo 'project OK'
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/edge
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed
realpath /home/user/Datasets/kpm-own/CEP-DVS/pathFile.csv
realpath /home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/smoke_20261014T072630Z/run.json
test ! -e analysis/frequency/audit_cepdvs_transfer.py && echo 'new audit script path OK'
test ! -e analysis/frequency/run_audit_cepdvs_transfer.sh && echo 'new wrapper path OK'
```

若项目目录不是上述路径，先定位运行阶段 1 时的项目根目录；若 Edge 数据不存在、`realpath` 落在 `/home/user/Datasets/kpm-own` 之外，或远程已有同名审计脚本，先停止并核实，不要覆盖文件。旧的 `dataloader/cepdvs.py` 可能仍指向 `/home/user/kpm/kpm/Dataset/CEP-DVS`；脚本会报告该默认 `pathFile.csv` 是否存在、是否与本次审计的 CSV 一致。

## 2. 上传两个新文件（本地 PowerShell 终端）

```text
sftp -P 56089 root@10.8.135.1
sftp> cd /home/user/kpm/kpm/ESVAE/ESVAE/analysis/frequency
sftp> put C:/coding/SDSTL/analysis/frequency/audit_cepdvs_transfer.py
sftp> put C:/coding/SDSTL/analysis/frequency/run_audit_cepdvs_transfer.sh
sftp> bye
```

两个文件分别是 [审计脚本](audit_cepdvs_transfer.py)和[路径检查与运行封装](run_audit_cepdvs_transfer.sh)。如果远程项目根目录不同，仅调整 SFTP 的 `cd`，不要改本地文件路径。

## 3. 运行（远程 SSH 终端）

```bash
cd /home/user/kpm/kpm/ESVAE/ESVAE
bash analysis/frequency/run_audit_cepdvs_transfer.sh
```

默认读取 `/home/user/Datasets/kpm-own/CEP-DVS/data/edge`、`data/dvs_processed`、`pathFile.csv` 和现有 smoke 结果的 `run.json`、`per_sample_channel.csv`。新结果位于 `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/edge_pair_audit_<UTC时间>/`。运行完成会打印 `Results:` 绝对路径及摘要。

如果结果显示 `Dataloader default pathFile.csv ... matches audited CSV=False`，配对表只代表本次指定的 CSV，不能直接代表未修改的训练入口；先核实旧路径和实际训练启动命令。`Same file ID` 与 `same label` 必须分别看：标签相同不代表同一个 RGB/Edge-DVS 实例。`stored_edge_label_mismatches` 只检查 smoke 的 20 个 Edge 文件，不是全数据集检查。

## 4. 拉回结果（本地 PowerShell 终端）

将 `<实际目录名>` 替换为上一步输出的 `edge_pair_audit_...`：

```text
sftp -P 56089 root@10.8.135.1
sftp> lcd C:/coding/SDSTL/analysis/frequency/results/stage1_cepdvs
sftp> get -r /home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/<实际目录名>
sftp> bye
```

回传整个目录后，核对 `summary.md`、`run.json`、`pair_audit.csv`、`offline_edge_sample_channel.csv`。下一步才决定是否扩大每类样本数及开展分箱修正对照。
