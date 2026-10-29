# 阶段 1 的开源参考审查

检索日期：2026-10-14。这里记录已经核对过的参考，不表示复制其代码。

| 来源 | 核对结果 | 本阶段采用情况 |
|---|---|---|
| [MaxFormer 论文](https://arxiv.org/html/2505.18608v3)与[作者仓库](https://github.com/bic-L/MaxFormer) | 论文用频谱观察与池化/局部卷积消融建立论点；仓库 README 说明可视化代码未随仓库提供 | 借鉴“频谱观察与任务消融分开”的研究结构；不声称复现其频谱实现或阈值 |
| [PyTorch `torch.fft.fft2` 文档](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html) | `norm="ortho"` 对二维 FFT 作正交归一化 | 参考其归一化定义；本地独立实现使用 NumPy FFT 以便无 GPU 计算，Parseval 用测试验证 |
| [NumPy `fftfreq` 文档](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html) | 频率坐标是每像素采样间隔的 cycles/pixel，正负频率位置由 FFT 长度决定 | 用于定义二维径向频带，代码未复制 |
| [event_representation](https://github.com/LarryDong/event_representation) | 罗列事件帧、极性累积及其他事件表示；仓库标注 GPL-3.0 | 仅用于术语和方案对照，不引入其代码；本阶段沿用 SDSTL 自身的事件分箱 |

后续阶段如需采集 SNN 中间状态或加入频带干预，将再次检索对应的作者实现并记录具体 commit、文件、许可证和输入约定。

阶段 2 早层探针参考 [PyTorch `Module.register_forward_hook` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) 确认前向钩子只读取模块输出，并查看 [SpikingJelly 的状态监测示例](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/examples/Spiking_DQN_state.py) 作为 SNN 中间状态观测方式的对照。探针按 SDSTL 自有 `VGGSNN` / `LIFSpike` 实现编写，没有复制外部代码；共享池化的调用次序依据本地模型结构逐次记录。

补充配对审计时复核了 [event_representation](https://github.com/LarryDong/event_representation) 与 [DVS Benchmark 2021](https://github.com/holesond/dvs-benchmark-2021) 的公开事件格式说明。它们可帮助确认极性与事件帧术语，但没有 SDSTL `get_edge2dvs_cepdvs` 的索引/标签规则；审计逻辑因此直接依照本项目数据加载器实现，没有复制外部代码。
