# CEP-DVS 加载器路径修正与单批次核验

[数据路径说明](../../Docs/point1/数据集路径与格式说明.txt)和[配对审计](results/edge_pair_audit_20261014T081612Z/analysis.md)都使用 `/home/user/Datasets/kpm-own/CEP-DVS`；审计时远程 `dataloader/cepdvs.py` 的默认根目录仍是旧路径，且旧 `pathFile.csv` 不存在。为后续实验，本地仅更新了 `dataloader/cepdvs.py` 的五个 CEP-DVS 默认路径，没有修改数据划分、标签解析或训练逻辑。远程运行前要确认该文件没有用户改动。

在 VS Code 的远程 SSH 终端预检：

```bash
cd /home/user/kpm/kpm/ESVAE/ESVAE
pwd -P
git rev-parse HEAD
git status --short -- dataloader/cepdvs.py
test ! -e analysis/frequency/run_cepdvs_loader_smoke.sh && echo 'new wrapper path OK'
realpath /home/user/Datasets/kpm-own/CEP-DVS/pathFile.csv
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/img
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/edge
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed
realpath -m /home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs
```

只有远程 HEAD 仍为本地补丁基线 `21e246da6d1d82e80b2fdcc135f1b4091827a313`、`git status --short -- dataloader/cepdvs.py` **无输出**、数据路径均位于 `/home/user/Datasets/kpm-own` 内且结果路径位于 `/home/user/kpm/kpm` 内，才在**本地 PowerShell**上传本地路径补丁和运行封装。还应确认远程没有同名 `run_cepdvs_loader_smoke.sh`，避免覆盖已有文件：

```text
sftp -P 56089 root@10.8.135.1
sftp> cd /home/user/kpm/kpm/ESVAE/ESVAE/dataloader
sftp> put C:/coding/SDSTL/dataloader/cepdvs.py
sftp> cd /home/user/kpm/kpm/ESVAE/ESVAE/analysis/frequency
sftp> put C:/coding/SDSTL/analysis/frequency/run_cepdvs_loader_smoke.sh
sftp> bye
```

然后在**远程 SSH 终端**执行单批次只读检查。它从当前加载器读取数据，结果独立保存在 `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/loader_smoke_<UTC时间>/check.log`；不会启动训练或写入数据集。

```bash
cd /home/user/kpm/kpm/ESVAE/ESVAE
bash analysis/frequency/run_cepdvs_loader_smoke.sh
```

成功标志是日志末尾出现 `LOADER_SMOKE_OK`。若出现依赖导入或数据形状错误，保留 `check.log` 并回传，不要安装包、改写数据或启动训练。若远程文件已有未提交改动，也不要直接覆盖；先对比该文件与本地补丁。

拉回日志时，在本地 PowerShell 输入下列 SFTP 命令，把 `<实际目录名>` 换成脚本打印的 `Results:` 路径最后一段：

```text
sftp -P 56089 root@10.8.135.1
sftp> lcd C:/coding/SDSTL/analysis/frequency/results/stage1_cepdvs
sftp> get -r /home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/<实际目录名>
sftp> bye
```
