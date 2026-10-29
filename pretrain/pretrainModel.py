# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""
# 移除错误的cv2.dnn.Layer导入，使用正确的SNN Layer
import models.TET__layer
from tl_utils.loss_function import *
from pretrain.Edge import *
from models.TET__layer import *




class VGGSNN(nn.Module):
    # 与下游 VGGSNN 的注意力位置和参数名一致。
    FEATURE_ATTENTION = {
        0: (2, 128), 2: (3, 256), 3: (4, 256),
        5: (5, 512), 6: (6, 512), 8: (7, 512), 9: (8, 512),
    }

    def __init__(self, cls_num=10, img_shape=48, device='cuda', use_event_attention=False,
                 event_attention_reduction=8, use_edge_extractor=False, rgb_to_gray=False,
                 edge_method='Sobel', event_attention_layers=2):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)  # RGB 3通道输入
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS/Edge 2通道输入

        # 事件注意力开关
        self.use_event_attention = use_event_attention
        if event_attention_layers not in (2, 8):
            raise ValueError('event_attention_layers must be 2 or 8')
        self.event_attention_layers = event_attention_layers

        # 预训练：边缘提供 Q/V，同层 RGB 提供 K；下游推理：事件自身提供 Q/K/V。
        if self.use_event_attention:
            from models.TET__layer import EventMidFrameAttention
            self.event_attn_1 = EventMidFrameAttention(64, reduction=event_attention_reduction)
            for _, (number, channels) in self.FEATURE_ATTENTION.items():
                if number > self.event_attention_layers:
                    continue
                setattr(self, f'event_attn_{number}',
                        EventMidFrameAttention(channels, reduction=event_attention_reduction))

        # 边缘提取器（仅用于 RGB->Edge 预训练，默认关闭）
        self.use_edge_extractor = use_edge_extractor
        if self.use_edge_extractor:
            extractors = {
                'Sobel': SobelEdgeExtractionModule,
                'Canny': CannyEdgeDetectionModule,
                'Laplacian': LaplacianEdgeExtractionModule,
            }
            if edge_method not in extractors:
                raise ValueError(f"不支持的边缘提取方式: {edge_method}")
            self.edge_method = edge_method
            self.edge_extractor1 = extractors[edge_method](device=device, in_channels=3)
            self.rgb_to_gray = RGB2GrayscaleModule() if rgb_to_gray else None
        else:
            self.rgb_to_gray = None

        self.features = nn.Sequential(
            Layer(64, 128, 3, 1, 1, False),
            pool,
            Layer(128, 256, 3, 1, 1, False),
            Layer(256, 256, 3, 1, 1, False),
            pool,
            Layer(256, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 1, 1, False),
            pool,
            Layer(512, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 1, 1, False),
            pool,
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # 传统瓶颈层设计，基于动态计算的特征图尺寸
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.bottleneck_lif_node = LIFSpike()
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def extract_edge(self, rgb_data):
        """
        RGB -> 2通道边缘图。所选算子生成 1 通道响应，再复制到 2 通道；
        两通道只是模型输入形状适配，不代表事件的 ON/OFF 极性。

        Args:
            rgb_data: (N, 3, H, W) RGB图像
        Returns:
            edge: (N, 2, H, W) 双通道边缘图
        """
        if not self.use_edge_extractor:
            raise RuntimeError("extract_edge() 需要模型以 use_edge_extractor=True 构造")

        # 可选灰度转换
        if self.rgb_to_gray is not None:
            rgb_data = self.rgb_to_gray(rgb_data)

        edge = self.edge_extractor1(rgb_data)
        return edge.repeat(1, 2, 1, 1)

    def forward(self, source, target, encoder_tl_loss_type='TCKA', feature_tl_loss_type='MSE',
                return_attention_loss=False):
        """
        Args:
            source: 源域输入，(N, T, C, H, W) C可以是3（RGB）或2（Edge）
            target: 目标域输入，(N, T, 2, H, W) 为DVS数据
            encoder_tl_loss_type: 包括 'TCKA'(分别计算各时间步mem的CKA，最后求平均), 'CKA'(将各时间步的spike求频率后计算CKA)
            feature_tl_loss_type: 包括 'TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'.
        Returns:
            Returns loss values during training:
                encoder_tl_loss: 编码器计算的迁移损失
                feature_tl_loss: 提取特征计算的迁移损失
        """
        # print("Source shape:", source.shape)
        # print("Target shape:", target.shape)

        if self.training:
            batch_size, T = source.shape[0:2]
            attention_losses = []

            # 根据通道数选择输入层
            # 处理源域输入 (3通道RGB 或 2通道Edge)
            if source.shape[2] == 3:
                source, source_mem = self.rgb_input(source)
            elif source.shape[2] == 2:
                source, source_mem = self.dvs_input(source)
            else:
                raise ValueError(f"Unexpected source channel number: {source.shape[2]}, expected 2 or 3")
            
            # 处理目标域输入 (2通道DVS)
            if target.shape[2] == 2:
                target, target_mem = self.dvs_input(target)
            else:
                raise ValueError(f"Unexpected target channel number: {target.shape[2]}, expected 2")
            
            # 第1处：Edge 提供 Q/V，RGB 提供 K。
            if self.use_event_attention:
                if return_attention_loss:
                    target, attention_loss = self.event_attn_1(
                        target, key_source=source, return_aux_loss=True)
                    attention_losses.append(attention_loss)
                else:
                    target = self.event_attn_1(target, key_source=source)

            # 计算编码器迁移损失
            if encoder_tl_loss_type == 'TCKA':
                encoder_tl_loss = 1 - temporal_linear_CKA(source_mem.view((batch_size, T, -1)),
                                                          target_mem.view((batch_size, T, -1)))
            elif encoder_tl_loss_type == 'CKA':
                encoder_tl_loss = 1 - linear_CKA(source.view((batch_size, T, -1)),
                                                 target.view((batch_size, T, -1)), "SUM")
            else:
                raise Exception(f"Invalid encoder_tl_loss_type: {encoder_tl_loss_type}")

            # 共享卷积逐层提取，两路在每处注意力保持相同空间尺度。
            for index, layer in enumerate(self.features):
                source = layer(source)
                target = layer(target)
                if self.use_event_attention and index in self.FEATURE_ATTENTION:
                    number, _ = self.FEATURE_ATTENTION[index]
                    if number <= self.event_attention_layers:
                        attention = getattr(self, f'event_attn_{number}')
                        if return_attention_loss:
                            target, attention_loss = attention(
                                target, key_source=source, return_aux_loss=True)
                            attention_losses.append(attention_loss)
                        else:
                            target = attention(target, key_source=source)

            source = torch.flatten(source, 2)
            source = self.bottleneck(source)
            source, source_mem = self.bottleneck_lif_node(source, return_mem=True)
            source_clf = self.classifier(source)

            target = torch.flatten(target, 2)
            target = self.bottleneck(target)
            target, target_mem = self.bottleneck_lif_node(target, return_mem=True)
            target_clf = self.classifier(target)

            # 计算特征迁移损失
            if feature_tl_loss_type == 'TMSE':
                feature_tl_loss = temporal_MSE(source_mem, target_mem)
            elif feature_tl_loss_type == 'MSE':
                feature_tl_loss = MSE(source, target, "SUM")
            elif feature_tl_loss_type == 'CKA':
                feature_tl_loss = 1 - linear_CKA(source, target, "SUM")
            elif feature_tl_loss_type == 'TCKA':
                feature_tl_loss = 1 - temporal_linear_CKA(source_mem, target_mem)
            elif feature_tl_loss_type == 'MMD':
                feature_tl_loss = MMD_loss(source, target, "SUM")
            else:
                raise Exception(f"Invalid feature_tl_loss_type: {feature_tl_loss_type}")

            if return_attention_loss:
                if not attention_losses:
                    raise RuntimeError('注意力引导损失需要启用注意力模块')
                return (source_clf, target_clf, encoder_tl_loss, feature_tl_loss,
                        torch.stack(attention_losses).mean())
            return source_clf, target_clf, encoder_tl_loss, feature_tl_loss

        else:
            # 测试模式：只处理目标域（DVS）数据
            if target.shape[2] == 3:
                target, _ = self.rgb_input(target)
            else:
                target, _ = self.dvs_input(target)
            
            # 测试及下游只有目标模态，由自身提供 Q/K/V。
            if self.use_event_attention:
                target = self.event_attn_1(target)

            for index, layer in enumerate(self.features):
                target = layer(target)
                if self.use_event_attention and index in self.FEATURE_ATTENTION:
                    number, _ = self.FEATURE_ATTENTION[index]
                    if number <= self.event_attention_layers:
                        target = getattr(self, f'event_attn_{number}')(target)

            target = torch.flatten(target, 2)
            target = self.bottleneck(target)
            target = self.bottleneck_lif_node(target)
            target_clf = self.classifier(target)
            return target_clf


class VGGSNNwoAP(VGGSNN):
    def __init__(self, cls_num=10, img_shape=32, use_event_attention=False,
                 event_attention_reduction=8, use_edge_extractor=False, rgb_to_gray=False,
                 edge_method='Sobel', event_attention_layers=2):
        """
        VGGSNNwoAP: 使用stride=2卷积替代平均池化

        Args:
            cls_num: 分类数量，默认10（CIFAR10）
            img_shape: 输入图像大小，默认32（与tl.py保持一致，CIFAR10使用32×32）
            use_event_attention: 是否启用事件注意力
            event_attention_reduction: 事件注意力通道压缩比
            use_edge_extractor: 是否实例化边缘提取器（仅 RGB->Edge 预训练时为 True）
            rgb_to_gray: 边缘提取前是否将 RGB 转灰度
        """
        super(VGGSNNwoAP, self).__init__(cls_num=cls_num, img_shape=img_shape, device='cuda',
                                         use_event_attention=use_event_attention,
                                         event_attention_reduction=event_attention_reduction,
                                         use_edge_extractor=use_edge_extractor,
                                         rgb_to_gray=rgb_to_gray,
                                         edge_method=edge_method,
                                         event_attention_layers=event_attention_layers)
        self.rgb_input = Layer(3, 64, 3, 1, 1, True)  # RGB 3通道输入
        self.dvs_input = Layer(2, 64, 3, 1, 1, True)  # DVS/Edge 2通道输入
        self.features = nn.Sequential(
            Layer(64, 128, 3, 2, 1, False),
            Layer(128, 256, 3, 1, 1, False),
            Layer(256, 256, 3, 2, 1, False),
            Layer(256, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 2, 1, False),
            Layer(512, 512, 3, 1, 1, False),
            Layer(512, 512, 3, 2, 1, False),
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # 关键修复：与tl.py完全一致，LIFSpike放在SeqToANNContainer内部
        # 注意：父类的bottleneck_lif_node仍然存在，会导致两次LIFSpike激活
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256),
                                            LIFSpike())
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
