# -*- coding: utf-8 -*-
"""
RGB->Edge 预训练入口（合并版）

合并自:
  - train_rgb2edge.py  Caltech101/CEP-DVS 预训练（双Sobel核边缘提取）
  - train.py           CIFAR10/MNIST 预训练（支持 TRT Loss）

设计说明:
  - 支持 4 个数据集: CIFAR10, MNIST, Caltech101, CEP-DVS
  - 兼容 --epoch 和 --epochs 两个参数名（train.py 用 --epochs，train_rgb2edge.py 用 --epoch）
  - 支持 TET Loss 和 TRT (Temporal Regularization Training) Loss
  - 使用双Sobel核边缘提取器(SobelEdgeExtractionModule + CannyEdgeDetectionModule)生成 2 通道边缘图
    注: 两个提取器核心梯度算子均为Sobel核，extractor1输出连续梯度幅值，extractor2为简化Canny(高斯模糊→Sobel梯度→硬阈值二值化)
  - 产出 rgb_edge_pretrained_best.pth 供后续 train_edge2dvs.py 使用
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

from dataloader.caltech101 import get_caltech101
from dataloader.cepdvs import get_cepdvs
from dataloader.cifar import get_cifar10
from dataloader.mnist import get_mnist
from pretrain.pretrainer import AlignmentTLTrainer_Edge_1
from pretrain.pretrainModel import VGGSNN, VGGSNNwoAP
from tl_utils.loss_function import TET_loss, TRT_loss
from tl_utils import common_utils

parser = argparse.ArgumentParser(description='RGB->Edge 预训练（合并版）')
# ======================== 数据集与基本训练参数 ========================
parser.add_argument('--data_set', type=str, default='Caltech101',
                    choices=['Caltech101', 'CEP-DVS', 'CIFAR10', 'MNIST'],
                    help='数据集名称')
parser.add_argument('--batch_size', default=32, type=int, help='Batchsize')
parser.add_argument('--optim', default='Adam', type=str, choices=['SGD', 'Adam'], help='Optimizer')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=30, type=int, help='Training epochs')
parser.add_argument('--epochs', default=None, type=int, help='Training epochs (兼容 train.py 的 --epochs 参数名)')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='是否使用多GPU并行')
parser.add_argument('--T', default=10, type=int, help='snn simulation time')
parser.add_argument('--encoder_type', type=str, default='time_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='编码器类型')
parser.add_argument('--seed', type=int, default=1000, help='随机种子')

# ======================== 迁移学习损失参数 ========================
parser.add_argument('--encoder_tl_loss_type', type=str, default='CKA', choices=['TCKA', 'CKA'],
                    help='编码器迁移损失类型')
parser.add_argument('--feature_tl_loss_type', type=str, default='TCKA',
                    choices=['TCKA', 'CKA', 'TMSE', 'MSE', 'TMMD', 'MMD'],
                    help='特征迁移损失类型')
parser.add_argument('--encoder_tl_lamb', default=0.1, type=float,
                    help='编码器迁移损失比例')
parser.add_argument('--feature_tl_lamb', default=0.1, type=float,
                    help='特征迁移损失比例')

# ======================== 模型与边缘提取参数 ========================
parser.add_argument('--img_shape', type=int, default=48, help='图像尺寸')
parser.add_argument('--use_woap', default=False, type=bool,
                    help='是否使用 VGGSNNwoAP (无 AvgPool 版)')
parser.add_argument('--rgb_to_gray', action='store_true', default=False,
                    help='是否在边缘提取前将 RGB 转灰度')
parser.add_argument('--edge_method', type=str, default='Sobel',
                    help='边缘提取方法标记（用于日志名称）')

# ======================== TRT Loss 参数（合并自 train.py）========================
parser.add_argument('--use_trt', action='store_true', default=False,
                    help='是否使用 TRT (Temporal Regularization Training) Loss')
parser.add_argument('--trt_decay', type=float, default=0.5,
                    help='TRT decay factor δ (默认 0.5)')
parser.add_argument('--trt_lambda', type=float, default=1e-5,
                    help='TRT 正则化系数 λ (默认 1e-5)')
parser.add_argument('--trt_epsilon', type=float, default=1e-5,
                    help='TRT epsilon ε (默认 1e-5)')
parser.add_argument('--trt_eta', type=float, default=0.05,
                    help='TRT eta η (MSE loss weight, 默认 0.05)')

# ======================== RGB 采样参数（合并自 train.py）========================
parser.add_argument('--RGB_sample_ratio', type=float, default=1.0,
                    help='RGB 训练集使用比例')

# ======================== 路径参数 ========================
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/pretrain/log_dir',
                    help='TensorBoard 日志目录')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/pretrain/checkpoints',
                    help='模型 checkpoint 目录')
parser.add_argument('--GPU_id', type=int, default=0, help='GPU ID')
parser.add_argument('--num_classes', type=int, default=None, help='类别数（默认自动检测）')

args = parser.parse_args()

# 兼容 --epochs 和 --epoch（--epochs 优先，回退到 --epoch）
if args.epochs is not None:
    args.epoch = args.epochs

# 根据数据集设置类别数
dataset_cls = {
    'Caltech101': 101,
    'CEP-DVS': 20,
    'CIFAR10': 10,
    'MNIST': 10,
}
if args.data_set not in dataset_cls:
    raise ValueError(f"Unsupported dataset: {args.data_set}")
if args.num_classes is None:
    args.num_classes = dataset_cls[args.data_set]

device = torch.device(f"cuda:{args.GPU_id}")

# 生成日志名称
gray_tag = 'Gray_' if args.rgb_to_gray else ''
loss_tag = 'TRT' if args.use_trt else 'TET'
log_name = (
    f"{args.data_set}_RGB2Edge_Pretrain_"
    f"{'woAP' if args.use_woap else 'AP'}_"
    f"enc-{args.encoder_type}_"
    f"opt-{args.optim}_"
    f"lr{args.lr}_"
    f"T{args.T}_"
    f"seed{args.seed}_"
    f"RGB{args.RGB_sample_ratio}_"
    f"{gray_tag}"
    f"TWoSobelEdge_"
    f"{loss_tag}_"
    f"img_shape{args.img_shape}"
)

# 日志目录设置
log_dir = os.path.join(
    args.log_dir,
    f"{args.data_set}_EdgePretrain_{args.num_classes}",
    log_name
)

# 模型保存路径
checkpoint_dir = os.path.join(
    args.checkpoint,
    f"{args.data_set}_EdgePretrain_{args.num_classes}_{log_name}"
)

os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

model_path = checkpoint_dir
writer = SummaryWriter(log_dir=log_dir)

print(f"训练配置: {log_name}")
print(f"日志目录: {writer.log_dir}")
print(f"模型保存: {model_path}")

if __name__ == "__main__":
    common_utils.seed_all(args.seed)
    f = open(f"{args.data_set}_{args.seed}_rgb2edge_pretrain_result.txt", "a")

    print("\n" + "=" * 80)
    print(f"RGB->Edge 预训练 (双Sobel核边缘提取: Sobel幅值 + 简化Canny) - {args.data_set}")
    print(f"损失函数: {'TRT' if args.use_trt else 'TET'}")
    print("=" * 80)

    # ============================================================================
    # 准备 RGB 数据
    # ============================================================================
    print(f"\nLoading {args.data_set} RGB dataset for RGB->Edge pretraining...")
    print(f"图像尺寸设置: {args.img_shape}×{args.img_shape}")

    if args.data_set == 'Caltech101':
        rgb_full_loader, _ = get_caltech101(
            args.batch_size,
            train_set_ratio=1.0,
            img_size=args.img_shape
        )
        # 手动划分 train/test (90% train, 10% test) 防止数据泄露
        print(f"\n手动划分数据集（避免数据泄露）:")
        full_dataset = rgb_full_loader.dataset
        dataset_size = len(full_dataset)
        train_size = int(0.9 * dataset_size)
        test_size = dataset_size - train_size
        torch.manual_seed(args.seed)
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        print(f"  训练集: {len(train_dataset)} 样本 (90%)")
        print(f"  测试集: {len(test_dataset)} 样本 (10%)")

        if args.RGB_sample_ratio < 1.0:
            sampled_size = int(len(train_dataset) * args.RGB_sample_ratio)
            train_indices = torch.randperm(len(train_dataset))[:sampled_size]
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            print(f"  采样后训练集: {len(train_dataset)} 样本 (ratio={args.RGB_sample_ratio})")

    elif args.data_set == 'CEP-DVS':
        rgb_train_dataset_loader, rgb_test_dataset_loader = get_cepdvs(
            args.batch_size,
            train_set_ratio=args.RGB_sample_ratio,
            img_size=args.img_shape
        )
        train_dataset = rgb_train_dataset_loader.dataset
        test_dataset = rgb_test_dataset_loader.dataset
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")

    elif args.data_set == 'CIFAR10':
        # 来自 train.py：只使用训练集进行预训练
        train_loader = get_cifar10(args.batch_size, args.RGB_sample_ratio)
        train_dataset = train_loader.dataset
        test_dataset = None  # CIFAR10 预训练阶段不划分 test
        print(f"训练集RGB数量: {len(train_dataset)} 样本")
        print(f"数据来源: CIFAR10训练集")
        print(f"注意: 预训练阶段只使用训练集，测试在后续阶段进行")

    elif args.data_set == 'MNIST':
        # 来自 train.py
        train_loader = get_mnist(args.batch_size, args.RGB_sample_ratio)
        train_dataset = train_loader.dataset
        test_dataset = None
        print(f"训练集RGB数量: {len(train_dataset)} 样本")
        print(f"数据来源: MNIST训练集")
        print(f"注意: 预训练阶段只使用训练集，测试在后续阶段进行")

    else:
        raise ValueError(f"Unsupported dataset: {args.data_set}")

    # 创建数据加载器
    from dataloader.dataloader_utils import DataLoaderX
    rgb_train_loader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    rgb_test_loader = None
    if test_dataset is not None:
        rgb_test_loader = DataLoaderX(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            drop_last=False,
            pin_memory=True
        )

    print(f"\n=== RGB->Edge 预训练数据集信息 ===")
    print(f"数据集: {args.data_set}")
    print(f"RGB训练集数量: {len(train_dataset)}")
    if test_dataset is not None:
        print(f"RGB测试集数量: {len(test_dataset)}")
    print(f"类别数量: {args.num_classes}")
    print(f"训练模式: RGB作为源域 -> RGB边缘信息作为目标域")
    print(f"损失函数: {'TRT' if args.use_trt else 'TET'}")
    print(f"===========================\n")

    # ============================================================================
    # 准备模型（边缘提取器已内聚到模型类，通过构造函数启用）
    # ============================================================================
    if args.use_woap:
        model = VGGSNNwoAP(cls_num=args.num_classes, img_shape=args.img_shape,
                          use_edge_extractor=True, rgb_to_gray=args.rgb_to_gray)
        print("使用 VGGSNNwoAP 模型 (without Average Pooling)")
        print(f"  架构: stride=2卷积替代AvgPool2d")
    else:
        model = VGGSNN(cls_num=args.num_classes, img_shape=args.img_shape, device=device,
                      use_edge_extractor=True, rgb_to_gray=args.rgb_to_gray)
        print("使用标准 VGGSNN 模型 (with Average Pooling)")
        print(f"  架构: AvgPool2d下采样")
    print(f"  图像尺寸: {args.img_shape}×{args.img_shape}")
    print(f"  输入通道: RGB=3通道, Edge=2通道(Sobel幅值 + 简化Canny二值)")

    # 边缘提取器已内聚到模型类（use_edge_extractor=True 时自动实例化）
    if args.rgb_to_gray:
        print("✓ 已启用 RGB 到灰度转换:")
        print("  - 使用 ITU-R BT.601 标准 (0.299*R + 0.587*G + 0.114*B)")
        print("  - 保持 3 通道输出（每个通道值相同）")
        print("  - 在边缘提取前应用")
    else:
        print("✓ RGB到灰度转换: 未启用（直接使用RGB图像）")

    print("\n✓ 模型已内聚双边缘提取器（均基于 Sobel 核）:")
    print("  - edge_extractor1: SobelEdgeExtractionModule")
    print("      算子: Sobel_x + Sobel_y → sqrt(gx²+gy²) → 跨通道平均")
    print("      输出: 1通道连续梯度幅值")
    print("  - edge_extractor2: CannyEdgeDetectionModule (简化版)")
    print("      流程: RGB→灰度 → 高斯模糊(5×5) → Sobel梯度 → NMS(空操作) → 硬阈值二值化")
    print("      注: 核心梯度算子与 extractor1 相同(Sobel核)，NMS 未实际抑制")
    print("      输出: 1通道二值边缘")
    print("  - 叠加后: 2通道边缘图 (Sobel幅值 + Canny二值)，近似DVS双通道特性")
    print("  - 调用方式: model.extract_edge(rgb_data) → (N, 2, H, W)")

    if args.parallel and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # ============================================================================
    # 优化器
    # ============================================================================
    if args.optim == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr}
        ])
        print(f"RGB->Edge预训练使用 Adam 优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': [p for n, p in model.named_parameters() if 'input' in n], 'lr': args.lr * 10,
             'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False},
            {'params': [p for n, p in model.named_parameters() if 'input' not in n], 'lr': args.lr * 1,
             'momentum': 0.9, 'weight_decay': args.weight_decay, 'nesterov': False}
        ])
        print(f"RGB->Edge预训练使用 SGD 优化器，输入层学习率: {args.lr * 10}, 其他层学习率: {args.lr}")
    else:
        raise Exception(f"优化器应为 ['SGD', 'Adam']，输入为 {args.optim}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n迁移学习配置:")
    print(f"  编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}")
    print(f"  特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}")
    print(f"  编码器类型: {args.encoder_type}")
    print(f"  RGB转灰度: {'启用' if args.rgb_to_gray else '未启用'}")
    print(f"  边缘提取: 双Sobel核提取器 (SobelEdgeExtractionModule梯度幅值 + CannyEdgeDetectionModule简化Canny二值, 均使用Sobel核)")

    # ============================================================================
    # 选择损失函数：TRT 或 TET（合并自 train.py）
    # ============================================================================
    if args.use_trt:
        print(f"\n使用 TRT (Temporal Regularization Training) Loss")
        print(f"  - TRT decay (δ): {args.trt_decay}")
        print(f"  - TRT lambda (λ): {args.trt_lambda}")
        print(f"  - TRT epsilon (ε): {args.trt_epsilon}")
        print(f"  - TRT eta (η): {args.trt_eta}")
        criterion = lambda outputs, labels: TRT_loss(
            model, outputs, labels,
            criterion=torch.nn.CrossEntropyLoss(),
            decay=args.trt_decay,
            lamb=args.trt_lambda,
            epsilon=args.trt_epsilon,
            eta=args.trt_eta
        )
    else:
        print(f"\n使用 TET (Temporal Efficient Training) Loss")
        criterion = TET_loss

    # ============================================================================
    # 训练
    # ============================================================================
    print("\n开始 RGB->Edge 预训练...")
    trainer = AlignmentTLTrainer_Edge_1(
        args, device, writer, model, optimizer, criterion, scheduler,
        os.path.join(model_path, "rgb_edge_pretrained.pth")
    )

    best_train_acc, best_train_loss = trainer.train(rgb_train_loader)

    # 预训练测试（如果有测试集）
    if rgb_test_loader is not None:
        test_loss, test_acc1, test_acc5 = trainer.test(rgb_test_loader)
        print(f'\nRGB->Edge预训练结果: test_loss={test_loss:.5f} test_acc1={test_acc1:.4f} test_acc5={test_acc5:.4f}')
    else:
        # CIFAR10/MNIST 预训练阶段不进行测试（来自 train.py 行为）
        test_loss, test_acc1, test_acc5 = 0.0, best_train_acc, 0.0
        print(f'\n预训练阶段不进行测试（{args.data_set} 预训练模式）')
        print(f'最佳训练准确率: {best_train_acc:.3f}')
        print(f'最佳训练损失: {best_train_loss:.5f}')

    # 保存 RGB->Edge 预训练模型
    pretrained_path = os.path.join(model_path, "rgb_edge_pretrained_best.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_train_acc': best_train_acc,
        'best_train_loss': best_train_loss,
        'test_acc1': test_acc1,
        'test_acc5': test_acc5,
        'test_loss': test_loss,
        'args': args
    }, pretrained_path)
    print(f"\nRGB->Edge 预训练模型已保存到: {pretrained_path}")

    # 记录结果到 TensorBoard
    writer.add_scalar(tag="final/rgb_edge_accuracy", scalar_value=test_acc1, global_step=0)
    writer.add_scalar(tag="final/rgb_edge_loss", scalar_value=test_loss, global_step=0)
    writer.add_scalar(tag="final/rgb_edge_train_accuracy", scalar_value=best_train_acc, global_step=0)

    # 保存结果到文件
    write_content = (
        f'=== {args.data_set} RGB->Edge预训练 结果 ===\n'
        f'数据集: {args.data_set}\n'
        f'种子: {args.seed}\n'
        f'损失函数: {"TRT" if args.use_trt else "TET"}\n'
        f'边缘提取器: 双Sobel核 (SobelEdgeExtractionModule梯度幅值 + CannyEdgeDetectionModule简化Canny二值, 2通道输出)\\n'
        f'RGB转灰度: {"启用" if args.rgb_to_gray else "未启用"}\n'
        f'模型: {"VGGSNNwoAP" if args.use_woap else "VGGSNN"}\n'
        f'预训练epochs: {args.epoch}, 学习率: {args.lr}\n'
        f'编码器迁移损失: {args.encoder_tl_lamb} × {args.encoder_tl_loss_type}\n'
        f'特征迁移损失: {args.feature_tl_lamb} × {args.feature_tl_loss_type}\n'
        f'RGB样本比例: {args.RGB_sample_ratio}\n'
        f'RGB->Edge预训练准确率: {test_acc1:.4f}%\n'
        f'预训练模型保存路径: {pretrained_path}\n'
        f'=====================================\n\n'
    )
    f.write(write_content)
    f.close()

    writer.close()
    print(f"\n预训练完成！模型已保存到: {pretrained_path}")
    print(f"结果已记录到: {args.data_set}_{args.seed}_rgb2edge_pretrain_result.txt")
    print(f"\n使用预训练参数进行 DVS 微调:")
    print(f"  python train_edge2dvs.py --data_set {args.data_set} --pretrained_path {pretrained_path}")
