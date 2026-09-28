# ESVAE

> **E**dge-based **S**piking transfer learning for event cameras via structural decoupling（基于结构解耦的事件相机脉冲神经网络迁移学习）

以 RGB 图像的边缘（Edge）作为结构桥梁，把 RGB 域学到的空间结构知识迁移到 DVS 事件域的脉冲神经网络（SNN）。配套论文：*A Spiking Transfer Learning Method for Event Camera Based on Structural Decoupling*（见 `Docs/`）。

- License: MIT
- Python: 3.8+ ｜ PyTorch: 2.4.1 (CUDA 12.1) ｜ SNN 框架: [spikingjelly](https://github.com/fangwei123456/spikingjelly)

---

## 核心思想

DVS 事件数据本质反映场景边缘/运动信息（2 通道，正负极性），与 RGB（3 通道）域差异大。本项目用 **RGB 经 Sobel/Canny 提取的 2 通道边缘图**作为中间桥梁——它和 DVS 同为 2 通道、且语义近似——从而把域差异降到最小，实现 SNN 的有效跨域迁移。

```
┌──────────────────────────────────────────────────────────┐
│  阶段1  RGB → Edge 预训练        train_rgb2edge.py       │
│  产出   rgb_edge_pretrained_best.pth                     │
│  (Sobel幅值 + 简化Canny 二值 → 2通道，近似DVS双极性)     │
└────────────────────────────────┬─────────────────────────┘
                                 │ 预训练权重
                                 ▼
┌──────────────────────────────────────────────────────────┐
│  阶段2  Edge/RGB → DVS 迁移学习  train_edge2dvs.py       │
│  --source_mode edge (默认)  Edge(2ch)→DVS  域差异最小    │
│  --source_mode rgb           RGB(3ch) →DVS  直接迁移       │
│  产出   best_model.pth + Top-1/Top-5                     │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  阶段3  基线对照 (直接 DVS 训练)  baseline.py            │
│  作为迁移学习方法性能增益的对照基准                       │
└──────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
ESVAE/
├── baseline.py              # 阶段3: DVS 基线训练入口（4 数据集）
├── train_rgb2edge.py        # 阶段1: RGB→Edge 预训练入口（4 数据集）
├── train_edge2dvs.py        # 阶段2: Edge/RGB→DVS 迁移入口（3/2 数据集）
├── models/
│   ├── snn_models/VGG.py    # 主模型 VGGSNN / VGGSNNwoAP（TET 风格 VGG-SNN）
│   ├── ann_models/          # ANN 模型 + ann2snn 转换层
│   └── tl_models/           # 迁移学习专用模型（CifarNet/ResNet/VGG 等）
├── pretrain/
│   ├── Edge.py              # SobelEdgeExtractionModule / CannyEdgeDetectionModule
│   ├── pretrainModel.py     # 含 edge_extractor 的 VGGSNN
│   ├── pretrainer.py        # AlignmentTLTrainer_Edge_1（阶段1 训练器）
│   └── edge2dvs_trainer.py  # AlignmentTLTrainer_Edge2DVS（阶段2 训练器）
├── tl_utils/
│   ├── trainer.py           # BaselineTrainer / TLTrainer
│   ├── loss_function.py     # TET_loss / TRT_loss / CKA / TCKA / MMD / MSE
│   └── common_utils.py      # seed_all 等工具
├── dataloader/              # CIFAR10/MNIST/Caltech101/CEP-DVS 数据加载
├── data_process/           # 预处理脚本（RGB→Edge、RGB 按类别、DVS 预处理）
├── analysis/               # 特征子空间分析（参数对比/t-SNE/CKA）
├── metrics/                # FID / Inception Score
├── visualization/          # DVS 可视化
├── datasets/               # ANN/SNN 通用数据加载
├── Docs/                   # 论文 PDF + 训练脚本功能说明 + point1（指令/记录表）
└── init_fid_stats.py       # FID 统计量初始化
```

---

## 环境与依赖

实测环境：torch 2.4.1+cu121 ｜ torchvision 0.19.1 ｜ spikingjelly ｜ numpy 1.23.5 ｜ scipy 1.13.1 ｜ pandas 2.2.3 ｜ matplotlib 3.10.3 ｜ scikit-learn 1.6.1 ｜ opencv-python 4.12.0 ｜ tqdm 4.67.1 ｜ tensorboard 2.10.1 ｜ prefetch_generator

```bash
# 核心依赖（CUDA 按你的环境调整）
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install spikingjelly numpy scipy tqdm tensorboard prefetch_generator
# 分析工具额外依赖
pip install pandas matplotlib scikit-learn opencv-python
```

> 要求 torchvision ≥ 0.8（transform 对张量操作）、torch ≥ 1.13（`torch.load(weights_only=True)`）。

---

## 数据集准备

支持 4 个事件相机基准数据集：

| 数据集 | 类别数 | 用途 | 数据路径 |
|--------|--------|------|----------|
| CIFAR10-DVS | 10 | baseline / 预训练 RGB | DVS: `/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat`；RGB: `/home/user/kpm/kpm/Dataset/CIFAR10/cifar10` |
| N-MNIST | 10 | baseline / 预训练 RGB | torchvision MNIST + spikingjelly n_mnist |
| N-Caltech101 | 101 | baseline / 预训练 RGB | DVS: `/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101/{train,test}`；RGB: `.../caltech101/101_ObjectCategories` |
| CEP-DVS | 20 | baseline / 预训练 RGB | `/home/user/kpm/kpm/Dataset/CEP-DVS/data/{dvs_processed, MAT/img, img, edge}` |

**路径配置位置**：
- CIFAR10：`dataloader/cifar.py` 顶部 `DIR` 字典（`CIFAR10DVS` 硬编码于 `get_cifar10_DVS` 的 TET 分支）
- CEP-DVS：`dataloader/cepdvs.py` 顶部 `CEPDVS_*` 配置区
- Caltech101：`dataloader/caltech101.py` 的 `get_n_caltech101` 内硬编码

**CIFAR10 RGB 按类别预处理（新）**：预训练的 `get_cifar10` 现从按类别组织的 `.pt` 读取（与 DVS 目录形式对称），首次需先生成：

```bash
python data_process/preprocess_cifar10_rgb2cls.py
# 读: /home/user/kpm/kpm/Dataset/CIFAR10/cifar10 (torchvision 标准)
# 写: /home/user/kpm/kpm/Dataset/CIFAR10/cifar10_rgb/{train,test}/{类别}/*.pt
#     纯张量 (3,48,48) float32 [0,1]，标签由文件夹名按字母序分配 (airplane=0...truck=9)
```

---

## 快速开始

### 流程 A：完整迁移学习（Edge 源域，推荐）

```bash
# 1. 预训练 RGB→Edge
python train_rgb2edge.py --data_set Caltech101 --epoch 50 --lr 0.001

# 2. Edge→DVS 迁移学习（加载阶段1 权重）
python train_edge2dvs.py --source_mode edge --data_set Caltech101 \
    --pretrained_path /path/to/rgb_edge_pretrained_best.pth --epoch 100

# 3. 基线对照
python baseline.py --data_set Caltech101
```

### 流程 B：RGB 直接迁移

```bash
python train_rgb2edge.py --data_set Caltech101 --epoch 50
python train_edge2dvs.py --source_mode rgb --data_set Caltech101 \
    --pretrained_path /path/to/rgb_edge_pretrained_best.pth --epoch 100
```

### 流程 C：CIFAR10 基线 + TRT 预训练

```bash
python baseline.py --data_set CIFAR10
python train_rgb2edge.py --data_set CIFAR10 --use_trt --trt_lambda 1e-5 --epochs 50
```

> 详细参数说明见 `Docs/训练脚本功能说明.md`，启动指令速查见 `Docs/point1/baseline_启动指令.md`。

### 输出路径

| 阶段 | TensorBoard | 模型权重 |
|------|-------------|----------|
| baseline | `results/SDSTL/baseline/log_dir_{DS}/{log_name}/` | `results/SDSTL/baseline/checkpoints_{DS}/{log_name}.pth` |
| 预训练 | `results/SDSTL/pretrain/log_dir/...` | `results/SDSTL/pretrain/checkpoints/{DS}_EdgePretrain_{NC}_.../rgb_edge_pretrained_best.pth` |
| 迁移 | `results/SDSTL/transfer/log_dir/{Tag}2DVS_{DS}_{NC}/...` | `results/SDSTL/transfer/checkpoints/{Tag}2DVS_{DS}_{NC}_.../best_model.pth` |

（结果根目录默认 `/home/user/kpm/kpm/results/SDSTL/`）

---

## 关键特性

- **双 Sobel 核边缘提取器**（`pretrain/Edge.py`）：`SobelEdgeExtractionModule`（连续梯度幅值）+ `CannyEdgeDetectionModule`（简化 Canny 二值），叠加成 2 通道边缘图，近似 DVS 双极性。
- **TET / TRT Loss**（`tl_utils/loss_function.py`）：源自 ICLR 2022 的时序高效训练；TRT 为带时序正则化变体（`--use_trt` + `trt_decay/lambda/epsilon/eta`）。
- **CKA / TCKA 迁移损失**：`TCKA` 对 SNN 时序特征逐时间步计算 CKA 后取平均，专门适配脉冲网络时序特性；另支持 MMD / MSE / TMSE / TMMD。
- **事件注意力**（`EventMidFrameAttention`）：在 `dvs_input` 后和 `features[0]` 后插入，用事件中间帧引导注意力（`--use_event_attention`）。
- **EventRPG 增强**：几何增强 + RPGDrop + RPGMix（`--use_eventrpg --eventrpg_mix_prob`）。
- **选择性模块加载**（`baseline.py`）：`--load_dvs_input/features/bottleneck/classifier` 独立开关，便于消融"迁移哪一层收益最大"。
- **分层学习率**（预训练）：输入层 `lr×10` 加速适配新域，其余层基础 lr。

---

## 分析工具（`analysis/`）

用于论证"结构解耦迁移有效"的实验闭环：

```bash
# 1. 参数对比（L2 范数 + 余弦相似度，逐层）
python -m ESVAE.analysis.compare_parameters \
    --baseline_ckpt baseline.pth --finetuned_ckpt finetuned.pth \
    --output_dir results/param_comparison

# 2. 特征 t-SNE 可视化
python -m ESVAE.analysis.extract_features_tsne \
    --baseline_ckpt baseline.pth --pretrained_ckpt pretrained.pth \
    --data_path /path/to/n-caltech101 --output_dir results/tsne --max_samples 2000

# 3. CKA/TCKA 子空间定量对比
python -m ESVAE.analysis.compare_subspace_cka \
    --baseline_ckpt baseline.pth --pretrained_ckpt pretrained.pth \
    --data_path /path/to/n-caltech101 --output_dir results/cka --max_samples 2000
```

详见 `analysis/README.md`。

---

## 实验记录

`Docs/point1/实验结果记录表.xlsx`：按数据集分 sheet（CIFAR10 / Caltech101 / CEP-DVS），列含数据集名、实验名、配置、Top-1/Top-5 准确率、训练准确率、损失、日志路径、权重路径等，结果列预填路径模板，跑完填数即可。

---

## Citation

```bibtex
@article{esvae2026,
  title={A Spiking Transfer Learning Method for Event Camera Based on Structural Decoupling},
  year={2026}
}
```

---

## License

MIT License (Copyright © 2021 Hiromichi Kamata)。见 `LICENSE`。
