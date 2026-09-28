# -*- coding: utf-8 -*-
"""
Baseline 训练入口（合并版）

合并自:
  - baseline.py                  通用基线：CIFAR10/MNIST/Caltech101/CEP-DVS
  - train_caltech101_baseline.py Caltech101 专用基线

设计说明:
  - val_strategy='none'（默认）: 不使用验证集，基于训练集准确率保存最佳模型
    （对应原 baseline.py 行为）
  - val_strategy='test_as_val': 用测试集充当每轮验证集选择最佳模型，训练集用全量
    （对应原 train_caltech101_baseline.py 行为）

  - Caltech101 数据集有两种加载方式:
    (1) 不提供 --caltech101_dvs_path: 使用 get_n_caltech101（原始 .mat 格式）
    (2) 提供 --caltech101_dvs_path:    使用 create_caltech101_dataloaders（.pt/.bin 格式）
"""

import os
import torch
import argparse
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import BaselineTrainer
from tl_utils.loss_function import TET_loss
from dataloader.mnist import get_n_mnist
from dataloader.cifar import get_cifar10_DVS
from dataloader.caltech101 import get_n_caltech101, create_caltech101_dataloaders
from dataloader.cepdvs import get_cepdvs_dvs
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='Baseline: 直接在DVS数据上训练SNN（合并版）')
# ======================== 数据集与基本训练参数 ========================
parser.add_argument('--data_set', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'Caltech101', 'MNIST', 'CEP-DVS'],
                    help='数据集类型')
parser.add_argument('--batch_size', default=64, type=int, help='Batchsize')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
parser.add_argument('--epoch', default=80, type=int, help='Training epochs')
parser.add_argument('--id', default='test', type=str, help='Model identifier')
parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
parser.add_argument('--parallel', default=False, type=bool, help='是否使用多GPU并行')
parser.add_argument('--T', default=10, type=int, help='snn simulation time')
parser.add_argument('--encoder_type', type=str, default='lap_encoder',
                    choices=['lap_encoder', 'poison_encoder', 'time_encoder'],
                    help='RGB数据的SNN编码器类型')
parser.add_argument('--seed', type=int, default=1000, help='随机种子')
parser.add_argument('--dvs_sample_ratio', type=float, default=1,
                    help='DVS训练集使用比例')
parser.add_argument('--dvs_encoding_type', type=str, default='TET', choices=['TET', 'spikingjelly'])
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--lamb', default=1e-3, type=float, metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--img_shape', type=int, default=None,
                    help='图像尺寸（默认: CIFAR10=48, MNIST=34, Caltech101=48, CEP-DVS=48）')
parser.add_argument('--num_workers', type=int, default=8,
                    help='DataLoader 工作进程数 (Windows 遇到问题可设 0)')

# ======================== 验证集策略（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--val_strategy', type=str, default='none',
                    choices=['none', 'test_as_val'],
                    help='验证集策略: none=不用验证集(基于train_acc保存) | test_as_val=用测试集当验证集(基于val_acc保存)')

# ======================== Caltech101 专用参数（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--caltech101_dvs_path', type=str, default=None,
                    help='N-Caltech101 DVS 数据集路径 (.pt/.bin 格式); 不提供则使用 get_n_caltech101')
parser.add_argument('--fine_tuning', default='no', type=str, help='Fine-tuning 模式标识')

# ======================== 预训练模型加载（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--pretrained_path', type=str, default=None,
                    help='预训练模型参数路径')
parser.add_argument('--load_dvs_input', action='store_true', default=False,
                    help='是否加载 dvs_input 相关参数 (默认 False)')
parser.add_argument('--load_features', action='store_true', default=False,
                    help='是否加载 features 相关参数 (默认 False)')
parser.add_argument('--load_bottleneck', action='store_true', default=False,
                    help='是否加载 bottleneck 相关参数 (默认 False)')
parser.add_argument('--load_classifier', action='store_true', default=False,
                    help='是否加载 classifier 相关参数 (默认 False)')

# ======================== 事件注意力（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--use_event_attention', action='store_true', default=False,
                    help='启用事件中间帧引导注意力 (Event Mid-Frame Guided Attention)')
parser.add_argument('--event_attention_reduction', type=int, default=8,
                    help='事件注意力通道压缩比 (默认 8)')

# ======================== EventRPG 数据增强（原 baseline.py）========================
parser.add_argument('--use_eventrpg', action='store_true', default=False,
                    help='是否使用 EventRPG 数据增强 (Geometric + RPGDrop + RPGMix)')
parser.add_argument('--eventrpg_mix_prob', type=float, default=0.5,
                    help='EventRPG RPGMix 概率 (默认 0.5)')

parser.add_argument('--experiment_name', type=str, default='baseline')

# ======================== 路径参数 ========================
parser.add_argument('--log_dir', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/log_dir',
                    help='TensorBoard 日志目录')
parser.add_argument('--checkpoint', type=str, default='/home/user/kpm/kpm/results/SDSTL/baseline/checkpoints',
                    help='模型 checkpoint 目录')
parser.add_argument('--data_dir', type=str, default='/data/zhan/Event_Camera_Datasets',
                    help='数据集根目录')

args = parser.parse_args()

# 参数预设值
device = torch.device("cuda:0")

# 注意：log_name、writer 和 model_path 在 main 函数中生成
writer = None
model_path = None

if __name__ == "__main__":
    # 设置随机数种子
    seed_all(args.seed)

    # 设置图像尺寸默认值
    if args.img_shape is None:
        if args.data_set == 'CIFAR10':
            args.img_shape = 48
        elif args.data_set == 'MNIST':
            args.img_shape = 34
        elif args.data_set == 'Caltech101':
            args.img_shape = 48
        elif args.data_set == 'CEP-DVS':
            args.img_shape = 48
        else:
            args.img_shape = 48

    print(f"\n{'=' * 60}")
    print(f"Baseline 实验配置 (直接在 DVS 数据上训练 SNN) - 合并版")
    print('=' * 60)
    print(f"数据集: {args.data_set}")
    print(f"图像尺寸: {args.img_shape}×{args.img_shape}")
    print(f"时间步数: {args.T}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epoch}")
    print(f"学习率: {args.lr}")
    print(f"验证集策略: {args.val_strategy}")
    print(f"数据增强: {'EventRPG (mix_prob=' + str(args.eventrpg_mix_prob) + ')' if args.use_eventrpg else '传统增强'}")
    print(f"训练集使用比例: {args.dvs_sample_ratio}")
    print(f"事件注意力: {'启用 (reduction=' + str(args.event_attention_reduction) + ')' if args.use_event_attention else '未启用'}")
    if args.pretrained_path:
        print(f"预训练模型: {args.pretrained_path}")
        print(f"  加载 dvs_input: {args.load_dvs_input}, features: {args.load_features}, "
              f"bottleneck: {args.load_bottleneck}, classifier: {args.load_classifier}")
    print('=' * 60 + '\n')

    # 生成日志名称
    eventrpg_tag = f"_EventRPG-mix{args.eventrpg_mix_prob}" if args.use_eventrpg else ""
    attn_tag = "_EventAttn" if args.use_event_attention else ""
    ft_tag = f"_FT{args.fine_tuning}" if args.fine_tuning != 'no' else ""
    log_name = (f"Baseline_{args.data_set}_"
                f"img{args.img_shape}_"
                f"T{args.T}_"
                f"seed{args.seed}_"
                f"ratio{args.dvs_sample_ratio}_"
                f"lr{args.lr}_"
                f"epoch{args.epoch}"
                f"{eventrpg_tag}"
                f"{attn_tag}"
                f"{ft_tag}_"
                f"{args.experiment_name}")

    print(f"实验日志名称: {log_name}\n")

    # 创建 TensorBoard writer 和模型保存路径
    log_dir = os.path.join(args.log_dir + '_' + args.data_set, log_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    checkpoint_dir = os.path.join(args.checkpoint + '_' + args.data_set)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'{log_name}.pth')

    # ============================================================================
    # 准备数据集
    # ============================================================================
    print("加载数据集...")
    val_loader = None  # 默认不使用验证集

    if args.data_set == 'CIFAR10':
        train_loader, test_loader = get_cifar10_DVS(
            args.batch_size, args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type=args.dvs_encoding_type,
            use_eventrpg=args.use_eventrpg,
            eventrpg_mix_prob=args.eventrpg_mix_prob
        )
    elif args.data_set == 'Caltech101':
        if args.caltech101_dvs_path is not None:
            # 使用 create_caltech101_dataloaders (.pt/.bin 格式) — 来自 train_caltech101_baseline.py
            print(f"  使用 create_caltech101_dataloaders (.pt/.bin 格式)")
            print(f"  数据路径: {args.caltech101_dvs_path}")
            print(f"  数据增强: 水平翻转(50%) + 随机平移")
            train_loader, _, test_loader = create_caltech101_dataloaders(
                data_path=args.caltech101_dvs_path,
                batch_size=args.batch_size,
                train_ratio=args.dvs_sample_ratio,
                num_workers=args.num_workers,
                img_size=args.img_shape,
                use_nda=False,
                use_eventrpg=False,
                eventrpg_mix_prob=0.5,
                val_split=0.0
            )
        else:
            # 使用 get_n_caltech101 (.mat 格式) — 原始 baseline.py 方式
            train_loader, test_loader = get_n_caltech101(
                args.batch_size, args.T,
                train_set_ratio=args.dvs_sample_ratio,
                encode_type=args.dvs_encoding_type,
                size=args.img_shape,
                use_eventrpg=args.use_eventrpg,
                eventrpg_mix_prob=args.eventrpg_mix_prob
            )
    elif args.data_set == 'MNIST':
        train_loader, test_loader = get_n_mnist(
            args.batch_size, args.T,
            train_set_ratio=args.dvs_sample_ratio,
            encode_type=args.dvs_encoding_type,
            use_eventrpg=args.use_eventrpg,
            eventrpg_mix_prob=args.eventrpg_mix_prob
        )
    elif args.data_set == 'CEP-DVS':
        train_loader, test_loader = get_cepdvs_dvs(
            args.batch_size,
            train_set_ratio=args.dvs_sample_ratio,
            img_size=args.img_shape,
            time_bins=args.T,
            split_ratio=0.9
        )
    else:
        raise ValueError(f"不支持的数据集: {args.data_set}")

    # 根据验证集策略设置 val_loader
    if args.val_strategy == 'test_as_val':
        val_loader = test_loader
        print(f"  验证集策略: 使用测试集作为每轮验证集选择最佳模型")
    else:
        print(f"  验证集策略: 不使用验证集，基于训练集准确率保存最佳模型")

    print(f"✓ 训练集样本数: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"✓ 测试集样本数: {len(test_loader.dataset)} ({len(test_loader)} batches)")
    print()

    # ============================================================================
    # 准备模型
    # ============================================================================
    dataset_config = {
        'CIFAR10': 10,
        'MNIST': 10,
        'Caltech101': 101,
        'CEP-DVS': 20
    }

    if args.data_set not in dataset_config:
        raise ValueError(f"不支持的数据集: {args.data_set}")

    cls_num = dataset_config[args.data_set]
    img_shape = args.img_shape

    print(f"初始化 VGGSNN 模型...")
    print(f"  类别数: {cls_num}")
    print(f"  输入尺寸: {img_shape}×{img_shape}")

    model = VGGSNN(
        cls_num=cls_num, img_shape=img_shape,
        use_event_attention=args.use_event_attention,
        event_attention_reduction=args.event_attention_reduction
    )

    if args.parallel and torch.cuda.device_count() > 1:
        print(f"  使用 {torch.cuda.device_count()} 个GPU进行并行训练")
        model = torch.nn.DataParallel(model)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print()

    # ============================================================================
    # 加载预训练模型参数（可选）
    # ============================================================================
    if args.pretrained_path is not None and os.path.exists(args.pretrained_path):
        print(f"加载预训练模型参数: {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
            epoch_info = checkpoint.get('epoch', 'unknown')
            print(f"  来自 epoch: {epoch_info}")
        else:
            pretrained_dict = checkpoint

        model_dict = model.state_dict()

        # 构建需要排除的模块列表
        # 如果没有指定 --load_* 标志，则排除对应模块
        # 始终排除 edge_extractor
        exclude_modules = ['edge_extractor']
        if not args.load_dvs_input:
            exclude_modules.append('dvs_input')
        if not args.load_features:
            exclude_modules.append('features')
        if not args.load_bottleneck:
            exclude_modules.append('bottleneck')
        if not args.load_classifier:
            exclude_modules.append('classifier')

        # 过滤掉不匹配的键和需要排除的模块
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict
                           and v.shape == model_dict[k].shape
                           and not any(exclude_module in k for exclude_module in exclude_modules)}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"  成功加载: {len(pretrained_dict)}/{len(model_dict)} 个参数")
        skipped = set(model_dict.keys()) - set(pretrained_dict.keys())
        if skipped:
            print(f"  跳过参数数量: {len(skipped)}")
            for mod in ['dvs_input', 'features', 'bottleneck', 'classifier', 'edge_extractor']:
                mod_params = [k for k in skipped if mod in k]
                if mod_params:
                    print(f"    - {mod}相关参数（未加载）: {len(mod_params)} 个")
        print()
    else:
        print("从头开始训练（未使用预训练模型）\n")

    # ============================================================================
    # 准备训练组件
    # ============================================================================
    print("配置训练组件...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epoch)
    criterion = TET_loss

    print(f"  优化器: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  学习率调度: CosineAnnealingLR (T_max={args.epoch})")
    print(f"  损失函数: TET_loss")
    print()

    # ============================================================================
    # 开始训练
    # ============================================================================
    print("开始训练...\n")
    trainer = BaselineTrainer(args, device, writer, model, optimizer, criterion, scheduler, model_path)

    if args.val_strategy == 'test_as_val':
        # 用测试集当验证集，基于 val_acc 保存最佳模型
        best_val_acc = trainer.train(train_loader, val_loader)
    else:
        # 不用验证集，基于 train_acc 保存最佳模型
        trainer.train(train_loader)

    # ============================================================================
    # 最终测试
    # ============================================================================
    print("\n" + "=" * 60)
    print("最终测试评估")
    print("=" * 60)

    # 如果使用了验证集策略，加载最佳模型再测
    if args.val_strategy == 'test_as_val' and os.path.exists(model_path):
        print(f"加载最佳验证集模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)

    test_loss, test_acc = trainer.test(test_loader)
    print(f'测试损失: {test_loss:.5f}')
    print(f'测试精度: {test_acc:.3f} ({test_acc * 100:.2f}%)')
    print(f'最佳训练准确率: {trainer.best_train_acc:.3f}')
    print("=" * 60)

    # 记录最终测试结果
    writer.add_scalar(tag="final_test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    if args.val_strategy == 'test_as_val':
        writer.add_scalar(tag="final_test/val_accuracy", scalar_value=best_val_acc, global_step=0)
        writer.add_scalar(tag="final_test/train_accuracy", scalar_value=trainer.best_train_acc, global_step=0)

    writer.close()
    print(f"\n训练完成！模型已保存至: {model_path}")
