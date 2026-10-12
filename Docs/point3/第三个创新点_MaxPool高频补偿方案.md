# 第三个创新点：MaxPool 高频补偿（思路源自 MaxFormer）

## 一、思路来源：MaxFormer（NeurIPS 2025）

- 论文：*Spiking Neural Networks Need High Frequency Information*（arxiv 2505.18608）
- 仓库：https://github.com/bic-L/MaxFormer

### 核心发现
SNN 性能瓶颈根源是**频率偏置**（而非二值激活的信息损失）：
- 脉冲神经元（LIF）本质是**一阶 IIR 低通滤波器**（充电过程传递函数证明），高频无法有效传播
- 低通特性逐层串联被**急剧放大**，深层高频消散（论文类比为"人眼散光"），丢失局部细节/纹理

### 两个高通算子补偿
1. **Patch Embedding + Max-Pool**：源头注入高频（Max-Pool 是高通算子）
2. **DWC（深度卷积）替代早期自注意力**：保留局部高频，线性复杂度（vs 注意力二次）

### 关键反直觉结论
| 架构 | AvgPool（低通） | MaxPool（高通） |
|------|----------------|----------------|
| ANN Transformer | 更优（全局/低频） | — |
| **SNN** | CIFAR-100 76.73% | **79.12%（+2.39%）** |

- 卷积架构同样适用（Max-ResNet）：CIFAR-10 +2.25%，CIFAR-100 +6.65%
- 结论：高通算子实现 SNN **性能+能效兼得**

## 二、ESVAE 现状分析

ESVAE VGGSNN 用 `pool = SeqToANNContainer(nn.AvgPool2d(2))`（`pretrainModel.py:22`、`snn_models/VGG.py:16`）。

问题：
- AvgPool 是**低通**（平均/平滑），与 LIF 低通叠加 → **加剧低频偏置，丢失高频（边缘/纹理/细节）**
- ESVAE 核心是"**边缘桥梁**"（RGB Sobel 边缘作桥梁缩小 RGB↔DVS 域差异），而边缘是**高频信息** → AvgPool 低通丢失边缘高频，与边缘思路**矛盾**
- 现有 EventMidFrameAttention 用中间帧先验，未补偿频率偏置

## 三、创新点命题

**用 MaxPool（高通算子）替代 AvgPool，补偿 ESVAE VGGSNN 的低频偏置，增强高频/边缘特征表达**

理论依据：
1. MaxFormer 证明 SNN 用 MaxPool（高通）优于 AvgPool（低通），补偿 LIF 低频偏好
2. ESVAE 边缘桥梁依赖高频（边缘），MaxPool 保留高频 → **与边缘思路协同**
3. 样本对齐后，合成 DVS 的边缘与 RGB 对应（高频对应），MaxPool 保留的高频更有意义

SNN 脉冲(0/1)语义：MaxPool = 逻辑或（OR，任一脉冲→输出，保留强响应/高频）；AvgPool = 平均/密度（低通平滑）。

## 四、与现有创新点的协同

| 创新点 | 作用 | 与高频/边缘关系 |
|--------|------|----------------|
| 1 样本对齐 | 合成 DVS ↔ RGB，边缘对应 | 高频对应 |
| 2 注意力 | 中间帧先验引导 | 对齐后先验高质量 |
| **3 MaxPool 高频补偿** | **保留高频边缘，补偿低频偏置** | **直接保留高频** |

三者协同：对齐保证高频对应 → MaxPool 保留高频 → 注意力用高质量先验。

## 五、实施方案

### 改动（最小，方案阶段不执行）
```python
# 原: pretrainModel.py:22 / snn_models/VGG.py:16
pool = SeqToANNContainer(nn.AvgPool2d(2))
# 改:
pool = SeqToANNContainer(nn.MaxPool2d(2))
```
涉及 `pretrainModel.py:22`（预训练）、`snn_models/VGG.py:16`（baseline/迁移）。SeqToANNContainer 对任何池化都适用（时序每帧池化）。

### 实验设计（AvgPool vs MaxPool × 真实/合成）
| 组 | 池化 | DVS 数据 | 关键参数 |
|----|------|---------|---------|
| A1 | AvgPool | 真实 | （默认） |
| A2 | MaxPool | 真实 | 改 pool |
| B1 | AvgPool | 合成 | `--dvs_data_root` |
| B2 | MaxPool | 合成 | 改 pool + `--dvs_data_root` |

CIFAR10 / Caltech101 / CEP-DVS，80 epoch，seed=1000。
关键对比：
- A2 vs A1：MaxPool 高频补偿增益（预期正向，MaxFormer 已证）
- B2 vs B1：对齐+MaxPool 协同增益
- B2 vs A2：对齐带来的额外增益
- 可叠加注意力：B2+attn 为全创新点组合

### 进阶方向
1. **DWC 替代早期注意力**：MaxFormer 用 DWC 替代早期自注意力（保留高频+降复杂度）。ESVAE 可用 DWC 替代/补充 `event_attn_1`（前层高频敏感区）
2. **MaxPool 位置优化**：不全部替换，仅前层（高频敏感）用 MaxPool，后层保留 AvgPool
3. **MaxPool + 注意力结合**：MaxPool 高通改变特征分布，与中间帧注意力先验交互，需验证

## 六、待确认
- **MaxPool 在 SNN 脉冲(0/1)下的梯度传播**：max 不可导处用次梯度，需确认 `SeqToANNContainer(MaxPool2d)` 在 SNN 反向训练的兼容性（PyTorch MaxPool2d 有次梯度，应可用，但需验证 TET loss 下稳定）
- 全部 4 个 pool 替换 vs 部分层替换
- MaxPool 改变特征分布对 EventMidFrameAttention 先验的影响（三创新点叠加时）
- Max-ResNet 风格（stride=2 卷积，VGGSNNwoAP）是否更受益
