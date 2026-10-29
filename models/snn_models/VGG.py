# -*- coding: utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: VGG.py
@time: 2022/4/19 20:54
The code in this file is form Temporal Efficient Training of Spiking Neural Network via Gradient Re-weighting https://github.com/Gus-Lab/temporal_efficient_training
"""

from ..TET__layer import *


class VGGSNN(nn.Module):
    # features 中卷积层的位置、注意力模块编号和输出通道数。
    FEATURE_ATTENTION = {
        0: (2, 128),
        2: (3, 256),
        3: (4, 256),
        5: (5, 512),
        6: (6, 512),
        8: (7, 512),
        9: (8, 512),
    }

    def __init__(self, in_channel=2, cls_num=10, img_shape=32, use_event_attention=False,
                 event_attention_reduction=8, event_attention_layers=8,
                 attention_source='mid_event'):
        super(VGGSNN, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        self.dvs_input = Layer(2, 64, 3, 1, 1)
        
        # 事件注意力开关
        self.use_event_attention = use_event_attention
        if event_attention_layers not in (2, 8):
            raise ValueError('event_attention_layers must be 2 or 8')
        self.event_attention_layers = event_attention_layers
        if attention_source not in ('mid_event', 'gray_rgb', 'rgb'):
            raise ValueError('attention_source must be mid_event, gray_rgb or rgb')
        self.attention_source = attention_source
        self.gray_prior_projections = nn.ModuleDict()
        
        # 在事件输入层和后续每个卷积层之后添加事件注意力。
        if self.use_event_attention:
            self.event_attn_1 = EventMidFrameAttention(64, reduction=event_attention_reduction)
            if attention_source in ('gray_rgb', 'rgb'):
                image_channels = 1 if attention_source == 'gray_rgb' else 3
                self.gray_prior_projections['1'] = nn.Conv2d(image_channels, 64, 3, padding=1)
            for _, (number, channels) in self.FEATURE_ATTENTION.items():
                if number > self.event_attention_layers:
                    continue
                setattr(self, f'event_attn_{number}',
                        EventMidFrameAttention(channels, reduction=event_attention_reduction))
                if attention_source in ('gray_rgb', 'rgb'):
                    self.gray_prior_projections[str(number)] = nn.Conv2d(image_channels, channels, 3, padding=1)

        self.features = nn.Sequential(
            Layer(64, 128, 3, 1, 1),
            pool,
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool,
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 1, 1),
            pool,
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # self.T = 4
        self.bottleneck = SeqToANNContainer(nn.Linear(512 * W * W, 256))
        self.classifier = SeqToANNContainer(nn.Linear(256, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _image_query(self, guidance_frame, number, event_features):
        expected_channels = 1 if self.attention_source == 'gray_rgb' else 3
        if guidance_frame is None or guidance_frame.ndim != 4 or guidance_frame.size(1) != expected_channels:
            raise ValueError(f'{self.attention_source} 注意力需要 (N,{expected_channels},H,W) 的配对图像')
        image = torch.nn.functional.interpolate(guidance_frame,
            size=event_features.shape[-2:], mode='bilinear', align_corners=False)
        return self.gray_prior_projections[str(number)](image).unsqueeze(1)

    def forward(self, input, return_attention_loss=False, guidance_frame=None):
        # input = add_dimention(input, self.T)
        x = self.dvs_input(input)
        attention_losses = []

        if self.use_event_attention:
            query = self._image_query(guidance_frame, 1, x) if self.attention_source in ('gray_rgb', 'rgb') else None
            if return_attention_loss:
                x, loss = self.event_attn_1(x, return_aux_loss=True, query_source=query)
                attention_losses.append(loss)
            else:
                x = self.event_attn_1(x, query_source=query)

        for index, layer in enumerate(self.features):
            x = layer(x)
            if self.use_event_attention and index in self.FEATURE_ATTENTION:
                number, _ = self.FEATURE_ATTENTION[index]
                if number <= self.event_attention_layers:
                    attention = getattr(self, f'event_attn_{number}')
                    query = self._image_query(guidance_frame, number, x) if self.attention_source in ('gray_rgb', 'rgb') else None
                    if return_attention_loss:
                        x, loss = attention(x, return_aux_loss=True, query_source=query)
                        attention_losses.append(loss)
                    else:
                        x = attention(x, query_source=query)

        x = torch.flatten(x, 2)
        x = self.bottleneck(x)
        x = self.classifier(x)
        if return_attention_loss:
            if not attention_losses:
                raise RuntimeError('return_attention_loss requires use_event_attention=True')
            return x, torch.stack(attention_losses).mean()
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self, in_channel=2, cls_num=10, img_shape=32):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(in_channel, 64, 3, 1, 1),
            Layer(64, 128, 3, 2, 1),
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 2, 1),
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
        )
        W = int(img_shape / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, cls_num))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class AttentionVGGSNNwoAP(nn.Module):
    def __init__(self, T):
        super(AttentionVGGSNNwoAP, self).__init__()

        self.rgb_input_layer = Layer(3, 64, 3, 1, 1)
        self.dvs_input_layer = Layer(2, 64, 3, 1, 1)

        self.features = nn.Sequential(
            LayerWithAttention(64, 128, 3, 2, 1, T, 'T'),
            LayerWithAttention(128, 256, 3, 1, 1, T, 'T'),
            LayerWithAttention(256, 256, 3, 2, 1, T, 'T'),
            LayerWithAttention(256, 512, 3, 1, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 2, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 1, 1, T, 'T'),
            LayerWithAttention(512, 512, 3, 2, 1, T, 'T'),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, source, target):
        if source.shape[2] == 3:
            source_x = self.rgb_input_layer(source)
        elif source.shape[2] == 2:
            source_x = self.dvs_input_layer(source)
        target_x = self.dvs_input_layer(target)

        source_x, target_x = self.features((source_x, target_x))
        source_x = torch.flatten(source_x, 2)
        target_x = torch.flatten(target_x, 2)

        source_x = self.classifier(source_x)
        target_x = self.classifier(target_x)
        return source_x, target_x

