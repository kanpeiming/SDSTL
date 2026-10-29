# 实验 01：边缘提取方式对创新点 1 的影响

## 目标

在相同 RGB→Edge 预训练和 Edge→DVS 迁移流程下，比较 Sobel、Canny、Laplacian 三种边缘表示对分类精度和迁移效果的影响。

## 当前代码状态

- `train_rgb2edge.py` 的 `--edge_method {Sobel,Canny,Laplacian}` 已连接到模型内的实际提取器，默认仍为 Sobel。
- 三种方式均先生成单通道响应，再复制成 2 通道以适配现有 Edge 编码器；这两个通道不表示 DVS 的 ON/OFF 极性。
- Canny 使用 `pretrain/Edge.py` 中的简化实现：非极大值抑制直接返回输入，滞后阈值步骤只按高阈值二值化。
- Laplacian 使用四邻域核，对灰度卷积响应取绝对值并除以 8；Sobel 保留原有的跨 RGB 通道平均梯度幅值。

预训练入口现已可分别运行三组。下游 `train_edge2dvs.py` 读取独立的 Edge 数据路径，因此要进行完整迁移对比，还需按相同算子和样本划分生成各组 Edge 数据。

预训练调用示例（以 CEP-DVS 为例，其余训练参数按同一设置填写）：

```bash
python train_rgb2edge.py --data_set CEP-DVS --edge_method Sobel --experiment_name edge_sobel
python train_rgb2edge.py --data_set CEP-DVS --edge_method Canny --experiment_name edge_canny
python train_rgb2edge.py --data_set CEP-DVS --edge_method Laplacian --experiment_name edge_laplacian
```

## 对比设计

| 组别 | 边缘方式 | 实现与记录要求 |
|---|---|---|
| E1 | Sobel | 默认参考组：梯度幅值跨 RGB 通道平均 |
| E2 | Canny | 现有简化版：高斯平滑、Sobel 梯度、高阈值二值化 |
| E3 | Laplacian | 四邻域核：灰度响应绝对值除以 8 |

三组保持相同的 RGB/DVS 样本编号划分、图像尺寸、时间步、网络结构、损失、训练轮数、优化器及随机种子。统一边缘输出形状与通道数，记录各方法的数值分布；分别保存对应的边缘数据与预训练权重，下游迁移使用同组权重及边缘数据。测试集只用于最终评估。

## 待实现

1. 用样例图检查三种响应的强度、稀疏度和空间对齐。
2. 按同一划分生成每组下游 Edge 数据，分别运行 RGB→Edge 预训练与 Edge→DVS 迁移；给数据、日志和权重使用独立名称。
3. 在本目录记录实际命令、代码版本、数据路径、划分、权重路径及最终结果，再汇总三组差异。

## 结果记录

| 数据集 | 方法 | 种子 | 预训练权重 | 边缘数据路径 | DVS 测试准确率 | 备注 |
|---|---|---:|---|---|---:|---|
| 待填写 | Sobel | 1000 | 待填写 | 待填写 | 待填写 | |
| 待填写 | Canny | 1000 | 待填写 | 待填写 | 待填写 | |
| 待填写 | Laplacian | 1000 | 待填写 | 待填写 | 待填写 | |
