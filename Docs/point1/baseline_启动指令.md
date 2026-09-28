# Baseline 启动指令

> 直接在 DVS 事件数据上训练 SNN（`VGGSNN` + `TET_loss`），作为迁移学习方法的性能对照基准。
> 入口脚本：`baseline.py`，默认 `val_strategy=none`（基于 train_acc 保存最佳）。
> 日志/权重输出到：`/home/user/kpm/kpm/results/SDSTL/baseline/{log_dir,checkpoints}_{data_set}/{log_name}/`

---

## CIFAR10

```bash
python baseline.py --data_set CIFAR10
```

> 数据路径：`dataloader/cifar.py` 的 `get_cifar10_DVS`（TET 模式）硬编码读取
> `/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat/{train,test}`，
> 要求结构为 `{train,test}/{类别文件夹}/*.pt`（标签按类别文件夹字母序分配），当前环境已确认匹配，可直接跑。

> 预训练（RGB）：`get_cifar10` 已改为从按类别组织的 .pt 读取（与 DVS 目录形式对称），
> 路径 `cifar10_rgb/{train,test}/{类别}/*.pt`，存纯张量 `(3,48,48)`，标签由文件夹名推断。
> 首次需先生成数据：
> ```bash
> python data_process/preprocess_cifar10_rgb2cls.py
> ```

---

## Caltech101

```bash
python baseline.py --data_set Caltech101
```

> 默认走 `get_n_caltech101`（TET 模式），读取 `/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/{train,test}`（.pt）。
> 若要用 `.pt/.bin` 格式 + 事件注意力 + 选择性加载预训练模块（原 `train_caltech101_baseline.py` 行为）：
> ```bash
> python baseline.py --data_set Caltech101 --val_strategy test_as_val \
>     --caltech101_dvs_path /home/user/kpm/kpm/Dataset/Caltech101/NCALTECH101/NCALTECH101 \
>     --use_event_attention --load_features --load_bottleneck --load_classifier
> ```

---

## CEP-DVS

```bash
python baseline.py --data_set CEP-DVS
```

> 路径已在 `dataloader/cepdvs.py` 配置：`/home/user/kpm/kpm/Dataset/CEP-DVS/data/dvs_processed`，可直接跑。

---
