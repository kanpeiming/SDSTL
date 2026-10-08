# -*- coding: utf-8 -*-
"""
Edge/RGB -> DVS 迁移学习入口（合并版）

合并自:
  - train_edge2dvs.py  Edge(2ch)->DVS 迁移学习
  - train_rgb2dvs.py   RGB(3ch)->DVS 迁移学习

设计说明:
  - --source_mode edge (默认): Edge 数据（2通道）作为源域，域差异最小
    使用 get_edge2dvs_* 系列数据加载函数
    支持 Caltech101, CIFAR10, CEP-DVS
  - --source_mode rgb: RGB 数据（3通道）作为源域，RGB 直接迁移
    使用 get_tl_* 系列数据加载函数
    支持 Caltech101, CIFAR10
    Caltech101 会自动移除 Faces 类，保持 101 类（含 BACKGROUND_Google）

  - 原有命令行完全兼容:
    train_edge2dvs.py 的参数 → 默认 --source_mode edge
    train_rgb2dvs.py 的参数 → 加 --source_mode rgb 即可
"""

import argparse
import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter

# 添加 ESVAE 根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
esvae_root = os.path.dirname(current_dir)
if esvae_root not in sys.path:
    sys.path.insert(0, esvae_root)

from dataloader.caltech101 import get_edge2dvs_caltech101, get_tl_caltech101
from dataloader.cifar import get_edge2dvs_cifar10, get_tl_cifar10
from dataloader.cepdvs import get_edge2dvs_cepdvs
from pretrain.edge2dvs_trainer import AlignmentTLTrainer_Edge2DVS
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from tl_utils import common_utils
from tl_utils.loss_function import TET_loss

parser = argparse.ArgumentParser(description='Edge/RGB -> DVS 迁移学习（合并版）')
# ======================== 数据集与源域模式 ========================
parser.add_argument('--data_set', type=str, default='Caltech101',
                    choices=['Caltech101', 'CIFAR10', 'CEP-DVS'],
                    help='数据集名称')
parser.add_argument('--source_mode', type=str, default='edge',
                    choices=['edge', 'rgb'],
                    help='源域模式: edge=Edge(2ch)源域 | rgb=RGB(3ch)源域')

# ======================== 基本训练参数 ========================
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=100, type=int, help='Training epochs')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='Multi-GPU parallelism')
parser.add_argument('--T', default=10, type=int, help='SNN simulation time')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='编码器类型')
parser.add_argument('--seed', type=int, default=1000, help='Random seed')

# ======================== 迁移学习损失参数 ========================
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='编码器迁移损失类型')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='特征迁移损失类型')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float, help='编码器迁移损失比例')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float, help='特征迁移损失比例')

# ======================== 模型参数 ========================
parser.add_argument('--use_woap', default=False, type=bool, help='Use VGGSNNwoAP')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_classes', type=int, default=None, help='类别数（默认自动检测）')
parser.add_argument('--img_size', type=int, default=48, help='Image size')

# ======================== 数据采样参数 ========================
parser.add_argument('--edge_sample_ratio', type=float, default=1.0, help='Edge 训练集比例 (source_mode=edge)')
parser.add_argument('--dvs_sample_ratio', type=float, default=1.0, help='DVS 训练集比例')
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0, help='RGB 训练集比例 (source_mode=rgb)')
parser.add_argument('--split_ratio', type=float, default=0.9, help='Train/test 划分比例 (source_mode=rgb 时用于 DVS 数据)')
parser.add_argument('--num_workers', type=int, default=8, help='Data loading workers')

# ======================== 预训练模型参数 ========================
parser.add_argument('--pretrained_path', type=str, default='', help='RGB->Edge 预训练模型路径')

# ======================== 数据路径参数 (source_mode=edge) ========================
parser.add_argument('--edge_root', type=str, default='', help='Edge 数据根目录 (source_mode=edge)')
parser.add_argument('--dvs_root', type=str, default='', help='DVS 数据根目录 (source_mode=edge)')

# ======================== 事件注意力参数 ========================
parser.add_argument('--use_event_attention', action='store_true', default=False,
                    help='启用事件中间帧引导注意力 (Event Mid-Frame Guided Attention)')
parser.add_argument('--event_attention_reduction', type=int, default=8,
                    help='事件注意力通道压缩比 (默认 8)')

# ======================== 路径参数 ========================
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/transfer/log_dir',
                    help='TensorBoard 日志目录')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/transfer/checkpoints',
                    help='Checkpoint 目录')

args = parser.parse_args()

# 创建 args.epochs 别名（兼容 AlignmentTLTrainer_Edge2DVS 中可能的 args.epochs 引用）
args.epochs = args.epoch

# 根据数据集和源域模式设置默认路径和类别数
if args.data_set == 'Caltech101':
    if args.num_classes is None:
        args.num_classes = 101
    if args.source_mode == 'edge':
        if not args.edge_root:
            args.edge_root = '/home/user/kpm/kpm/Dataset/Caltech101/caltech101_edge'
        if not args.dvs_root:
            args.dvs_root = '/home/user/kpm/kpm/Dataset/Caltech101/NCALTECH101/NCALTECH101/Caltech101'
        if not args.pretrained_path:
            args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/Caltech101_EdgePretrain_101_Caltech101_RGB2Edge_Pretrain_AP_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_RGB1.0_TWoSobelEdge_img_shape48/rgb_edge_pretrained_best.pth'
elif args.data_set == 'CIFAR10':
    if args.num_classes is None:
        args.num_classes = 10
    if args.source_mode == 'edge':
        if not args.edge_root:
            args.edge_root = '/home/user/kpm/kpm/Dataset/CIFAR10/cifar10_edge'
        if not args.dvs_root:
            args.dvs_root = '/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat'
        if not args.pretrained_path:
            args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CIFAR10_10_Feature-Alignment_CIFAR10_enc-time_encoder_opt-Adam_lr0.001_T2_seed1000_TwoChannelBaseOnlySobel/best_model.pth'
elif args.data_set == 'CEP-DVS':
    if args.num_classes is None:
        args.num_classes = 20
    if args.source_mode == 'edge':
        if not args.edge_root:
            from dataloader.cepdvs import CEPDVS_EDGE_ROOT
            args.edge_root = CEPDVS_EDGE_ROOT
        if not args.dvs_root:
            from dataloader.cepdvs import CEPDVS_DVS_PROCESSED_ROOT
            args.dvs_root = CEPDVS_DVS_PROCESSED_ROOT
        if not args.pretrained_path:
            args.pretrained_path = '/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints/CEP-DVS_EdgePretrain_20_CEP-DVS_RGB2Edge_Pretrain_AP_enc-time_encoder_opt-Adam_lr0.001_T10_seed1000_RGB1.0_TWoSobelEdge_img_shape48/rgb_edge_pretrained_best.pth'
    else:
        raise ValueError(f"source_mode=rgb 不支持 CEP-DVS 数据集，请使用 --source_mode edge")
else:
    raise ValueError(f"Unsupported dataset: {args.data_set}")

device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
source_tag = 'Edge' if args.source_mode == 'edge' else 'RGB'
source_ratio = args.edge_sample_ratio if args.source_mode == 'edge' else args.RGB_sample_ratio
log_name = (
    f"{source_tag}2DVS_{args.data_set}_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"{source_tag}{source_ratio}_"
    f"DVS{args.dvs_sample_ratio}_"
    f"img{args.img_size}_"
    f"{'EventAttn' if args.use_event_attention else 'NoAttn'}"
)

log_dir = os.path.join(args.log_dir, f"{source_tag}2DVS_{args.data_set}_{args.num_classes}", log_name)
checkpoint_dir = os.path.join(args.checkpoint, f"{source_tag}2DVS_{args.data_set}_{args.num_classes}_{log_name}")

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")


if __name__ == "__main__":
    common_utils.seed_all(args.seed)

    print("\n" + "=" * 80)
    print(f"{source_tag}->DVS 迁移学习 ({args.data_set})")
    print(f"源域模式: {args.source_mode} ({'2通道' if args.source_mode == 'edge' else '3通道'} 输入)")
    print("=" * 80)

    # ============================================================================
    # 加载数据
    # ============================================================================
    print(f"\n加载 {source_tag} 和 DVS 数据集...")
    print(f"数据集: {args.data_set}")

    if args.source_mode == 'edge':
        print(f"Edge 数据路径: {args.edge_root}")
        print(f"DVS 数据路径: {args.dvs_root}")

        if args.data_set == 'Caltech101':
            train_loader, test_loader = get_edge2dvs_caltech101(
                batch_size=args.batch_size,
                edge_root=args.edge_root,
                dvs_root=args.dvs_root,
                edge_ratio=args.edge_sample_ratio,
                dvs_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_size
            )
        elif args.data_set == 'CIFAR10':
            train_loader, test_loader = get_edge2dvs_cifar10(
                batch_size=args.batch_size,
                edge_root=args.edge_root,
                dvs_root=args.dvs_root,
                edge_ratio=args.edge_sample_ratio,
                dvs_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_size
            )
        elif args.data_set == 'CEP-DVS':
            train_loader, test_loader = get_edge2dvs_cepdvs(
                batch_size=args.batch_size,
                edge_root=args.edge_root,
                dvs_root=args.dvs_root,
                edge_ratio=args.edge_sample_ratio,
                dvs_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_size,
                split_ratio=0.9,
                time_bins=args.T
            )
        else:
            raise ValueError(f"不支持的数据集: {args.data_set}")
    else:
        # source_mode == 'rgb'
        if args.data_set == 'Caltech101':
            print(f"注意: Caltech101 RGB数据将自动移除 Faces 类，保持 101 类（含 BACKGROUND_Google）")
            train_loader, test_loader = get_tl_caltech101(
                batch_size=args.batch_size,
                train_set_ratio=args.RGB_sample_ratio,
                dvs_train_set_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_size,
                split_ratio=args.split_ratio
            )
        elif args.data_set == 'CIFAR10':
            train_loader, test_loader = get_tl_cifar10(
                batch_size=args.batch_size,
                train_set_ratio=args.RGB_sample_ratio,
                dvs_train_set_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_size
            )
        else:
            raise ValueError(f"source_mode=rgb 不支持 {args.data_set} 数据集")

    # 检查数据集信息
    print("\n检查数据集信息...")
    try:
        sample_batch = next(iter(train_loader))
        if args.source_mode == 'edge':
            (source_data, dvs_data), (source_labels, dvs_labels) = sample_batch
            print(f"  Edge 数据形状: {source_data.shape}")
            print(f"  DVS 数据形状: {dvs_data.shape}")
            print(f"  Edge 标签范围: [{source_labels.min().item()}, {source_labels.max().item()}]")
            print(f"  DVS 标签范围: [{dvs_labels.min().item()}, {dvs_labels.max().item()}]")
        else:
            (rgb_data, dvs_data), labels = sample_batch
            print(f"  RGB 数据形状: {rgb_data.shape}")
            print(f"  DVS 数据形状: {dvs_data.shape}")
            print(f"  标签范围: [{labels.min().item()}, {labels.max().item()}]")
        print(f"  模型类别数: {args.num_classes}")
    except Exception as e:
        print(f"  无法检查数据集信息: {e}")

    # ============================================================================
    # 准备模型
    # ============================================================================
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=args.img_size,
                          use_event_attention=args.use_event_attention,
                          event_attention_reduction=args.event_attention_reduction)
        print(f"\n使用 VGGSNNwoAP 模型 (without Average Pooling)")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=args.img_size, device=device,
                      use_event_attention=args.use_event_attention,
                      event_attention_reduction=args.event_attention_reduction)
        print(f"\n使用标准 VGGSNN 模型 (with Average Pooling)")

    if args.use_event_attention:
        print(f"✓ 事件注意力已启用:")
        print(f"  - 中间稳定帧引导的时序注意力")
        print(f"  - 插入位置: dvs_input后 + features[0]后")
        print(f"  - 通道压缩比: {args.event_attention_reduction}")
    else:
        print("✗ 事件注意力未启用（使用标准训练）")

    # ============================================================================
    # 加载预训练参数（如果提供）
    # ============================================================================
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"\n加载 RGB->Edge 预训练参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            print(f"加载 epoch {checkpoint.get('epoch', 'unknown')} 的预训练模型")
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()

        # 过滤掉不匹配的键和 edge_extractor/rgb_to_gray 模块
        exclude_modules = ['edge_extractor', 'rgb_to_gray']
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape
                          and not any(exclude_module in k for exclude_module in exclude_modules)}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"成功加载 {len(pretrained_dict)}/{len(model_dict)} 个预训练参数")
        skipped_params = set(model_dict.keys()) - set(pretrained_dict.keys())
        if skipped_params:
            print(f"跳过的参数数量: {len(skipped_params)}")
            edge_extractor_params = [k for k in skipped_params if 'edge_extractor' in k or 'rgb_to_gray' in k]
            if edge_extractor_params:
                print(f"  - edge_extractor/rgb_to_gray 相关参数（未加载）: {len(edge_extractor_params)} 个")
    else:
        if args.pretrained_path:
            print(f"\n警告: 预训练模型路径不存在: {args.pretrained_path}")
        print("  从头开始训练（未使用预训练参数）")

    if args.parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    # ============================================================================
    # 优化器
    # ============================================================================
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"\n使用 Adam 优化器，学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                   weight_decay=args.weight_decay, nesterov=False)
        print(f"\n使用 SGD 优化器，学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    # 学习率调度器
    if args.data_set == 'Caltech101':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    elif args.data_set == 'CIFAR10':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n迁移学习配置:")
    print(f"  源域模式: {args.source_mode} ({'Edge 2ch' if args.source_mode == 'edge' else 'RGB 3ch'})")
    print(f"  编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}")
    print(f"  特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}")
    print(f"  训练轮数: {args.epoch}")
    print(f"  编码器类型: {args.encoder_type}")

    criterion = TET_loss
    print(f"\n使用 TET (Temporal Efficient Training) Loss")

    # ============================================================================
    # 训练
    # ============================================================================
    print(f"\n开始 {source_tag}->DVS 迁移学习...")
    trainer = AlignmentTLTrainer_Edge2DVS(
        args, device, writer, model, optimizer, criterion, scheduler, model_path
    )

    best_train_acc, best_train_loss = trainer.train(train_loader)
    test_loss, test_acc1, test_acc5 = trainer.test(test_loader)

    print(f'\n最终测试结果:')
    print(f'  test_loss={test_loss:.5f}')
    print(f'  test_acc1={test_acc1:.4f} ({test_acc1 * 100:.2f}%)')
    print(f'  test_acc5={test_acc5:.4f} ({test_acc5 * 100:.2f}%)')

    writer.add_scalar(tag="final_test/accuracy1", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="final_test/accuracy5", scalar_value=test_acc5, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)

    writer.close()

    print(f"\n训练完成！模型已保存到: {os.path.join(model_path, 'best_model.pth')}")
    print(f"结果已记录到: {args.data_set}_{args.seed}_{('edge' if args.source_mode == 'edge' else 'rgb')}2dvs_result.txt")
