# 阶段 1：CEP-DVS 输入与双 Sobel 诊断

当前交付：CEP-DVS 的最小数据诊断脚本、频谱指标、样本选择与数学验证。远程 20 样本 smoke 已核对，结果见 [独立报告](results/stage1_cepdvs/smoke_20261014T072630Z/analysis.md)；本脚本的双 Sobel 指在线预训练输出，离线迁移训练所读 Edge `.pt` 尚未纳入对照。

## 范围

脚本以同一文件编号读取 RGB `.jpg` 与预处理 DVS `.pt`，按 CEP-DVS 的 `pathFile.csv` 映射类别。可选读取对应原始事件 `.mat`，分析事件计数、分箱、归一化和缩放前后的差异。双 Sobel 通过当前 `pretrain.Edge.SobelEdgeExtractionModule` 在模型标准化后的 RGB 上产生；两个 Edge 通道相同，保持现有预训练设定。

固定子集按 DVS baseline 的排序后前 90% 为 train、末 10% 为 test，再按类别用固定 seed 抽样。这个 train/test 定义用于 DVS 数据诊断，不等同于 RGB 预训练的随机 90/10 划分。样本 ID 可以用于 CEP-DVS 的文件对应；具体拍摄几何关系仍须结合数据集来源判断。

## 远程运行准备

根据 [sdstl-remote-ssh 技能](../../.codex/skills/sdstl-remote-ssh/SKILL.md)，在 VS Code 集成终端建立 SSH 会话。先核实项目真实目录、源码版本和下面三个数据目录的 `realpath`，并确认它们均位于允许的远程根目录。尤其不要把旧文档里的 `/home/user/kpm/kpm/Dataset/...` 当作已验证路径。

代码同步到远程项目后，推荐运行包装脚本。它会验证项目与输入路径解析结果，并为每次运行创建独立结果目录。如果远程还没有 `analysis/frequency/`，先在 VS Code 集成终端连接服务器，确认项目根目录及 `analysis/` 后，从本地终端复制本目录；若远程已有同名目录，先检查其中的修改，再同步本次文件。

首次检查可在远程会话中运行：

```bash
cd /home/user/kpm/kpm
find . -maxdepth 4 -name baseline.py -type f -print
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/img
realpath /home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed
realpath /home/user/Datasets/kpm-own/CEP-DVS/pathFile.csv
```

找到并确认项目根目录后，如远程尚无 `analysis/frequency/`，在本地 PowerShell 集成终端执行（将变量改为刚确认的实际路径）：

```powershell
$remoteProject = '/home/user/kpm/kpm/ESVAE/ESVAE'
scp -P 56089 -r 'C:\coding\SDSTL\analysis\frequency' "root@10.8.135.1:${remoteProject}/analysis/"
```

远程会话中执行分析：

```bash
cd <核实后的远程项目根目录>
bash analysis/frequency/run_stage1_cepdvs.sh smoke
bash analysis/frequency/run_stage1_cepdvs.sh full
```

如 CEP-DVS 实际根目录不同，但仍位于允许的数据树，可作为第二参数传入。输出集中在 `/home/user/kpm/kpm/results/SDSTL/frequency/stage1_cepdvs/<run_id>/`，审阅结论索引在 [results/stage1_cepdvs/](results/stage1_cepdvs/README.md)。

下例保留直接调用方式，仅在路径确实存在且解析结果位于允许目录后执行：

```bash
cd /home/user/kpm/kpm/SDSTL
python -m analysis.frequency.stage1_cepdvs \
  --rgb-dir /home/user/Datasets/kpm-own/CEP-DVS/data/img \
  --dvs-dir /home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed \
  --csv-dir /home/user/Datasets/kpm-own/CEP-DVS \
  --raw-event-dir /home/user/Datasets/kpm-own/CEP-DVS/data/MAT/img \
  --output-dir /home/user/kpm/kpm/results/SDSTL/frequency/cep_dvs_stage1_first \
  --split train --per-class 5 --seed 1000 --size 48 --plot-count 3
```

若原始 `.mat` 不在允许目录，包装脚本会省略它；原始事件部分标记为未完成。直接调用时也可省略 `--raw-event-dir`。`--output-dir` 必须是不存在的新目录，避免覆盖旧结果。运行需要项目原有的 PyTorch、torchvision、NumPy、Pillow、SciPy 和 Matplotlib 环境。

首次远程运行建议先 `--per-class 1 --plot-count 1` 检查标签、尺寸、空帧和重建差异，再以固定设置做完整抽样。运行不会训练模型或写入数据目录。

## 输出

- `run.json`：参数、样本及数据路径、环境版本、源码 commit 与文件指纹、频带边界。
- `summary.md`：可直接回传的数值摘要，尚未给出因果解释。
- `per_sample_channel.csv`：每个样本、模态、通道的功率与活动统计。
- `per_class_channel.csv`：类别均值、标准差及有效样本数。
- `radial_power_density.csv`：各模态逐通道的径向平均功率密度。
- `sample_auxiliary.csv`：双通道相关性、逐时间箱响应量、原始事件计数、重建误差。
- `example_<id>.png`：RGB、一个 Sobel 通道、DVS 两极性中间箱及对应二维频谱。

模态名说明：`rgb_pixel` 是缩放后的 `[0,1]` RGB；`rgb_model_normalized` 复现 CEP-DVS RGB 加载器的通道标准化；`rgb_gray_pixel` 仅作灰度参照；`edge_dual_sobel` 是两个相同 Sobel 幅值通道；`dvs_processed` 是模型可读取的预处理事件帧；`dvs_sum` 和 `dvs_difference` 分别是两极性和与差。提供原始 `.mat` 时还会记录 `dvs_native_count`、`dvs_native_normalized` 和 `dvs_rebuilt_processed`。

## 指标解释

- 二维 FFT 在每个时间箱、通道的空间轴上计算，使用正交归一化。频带能量先减去各帧空间均值，DC 不进入频带能量；`raw_energy` 保留均值影响。
- 频率半径按对角 Nyquist 半径归一化，预设低频 `[0,0.15)`、中频 `[0.15,0.35)`、高频 `[0.35,1]`。这是诊断约定，非 MaxFormer 官方频带阈值。所有边界写入 `run.json`。
- `*_fraction` 是单个样本逐通道的频带能量占比。中心化能量为零时，占比为 NaN，并记录零能量帧；聚合结果提供有效样本数。
- `activity_fraction` 是非零像素比例。标准化 RGB 通常几乎全非零，不能与事件稀疏度按同一物理意义比较。
- `response_mass` 对预处理 DVS 是归一化响应之和，不是原始事件计数。原始计数仅来自 `.mat`；无原始事件时不可反推计数。
- 原始 180×240 与处理后 48×48 的 cycles/pixel 对应不同空间尺度。可检查各自频谱，不直接把频带数值差解释为处理链的传递增益。
- `rebuilt_max_abs_error` 比较当前代码重建的 48×48 事件帧与磁盘 `.pt`；误差不为零时先核查历史预处理版本，不强行归因于数据本身。
- 图像使用 `log1p` 显示中心化功率；CSV 中所有能量保持线性尺度。两极性相关性仅在两通道都有变化时定义。

## 阶段完成前的人工检查

1. `run.json` 的样本数、类别数、路径与远程真实目录一致；抽样 ID 可复现。
2. `edge_dual_sobel` 两通道数值一致，相关系数在非退化样本上接近 1。
3. 检查原始事件逐箱计数是否末箱明显偏稀，并与处理后逐箱响应质量分开解读。
4. 检查重建误差；若大于数值舍入误差，记录实际预处理版本。
5. 每类检查至少一个代表样本与异常样本，确认高频占比不是空帧、噪点或标准化造成。
6. 汇总空间、极性、密度、时间分布四方面观察，并决定阶段 2 首先采集哪两处层间特征。

本地执行了 `python -m unittest analysis.frequency.test_spectrum -v`，覆盖 FFT 能量、已知频率、零能量、样本选择、原始事件分箱；还在 PyTorch 2.5.1 环境用临时合成 `.jpg/.pt/.mat` 跑通完整输入、输出及绘图。远程真实数据 smoke 的完整性与数值核对已完成，更大固定子集和迁移源域对照仍待验证。
