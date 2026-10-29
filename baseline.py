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
import json
import random
from collections import defaultdict, Counter
from tqdm import tqdm
from tl_utils.common_utils import seed_all
from tl_utils.trainer import BaselineTrainer
from tl_utils.loss_function import TET_loss
from dataloader.mnist import get_n_mnist
from dataloader.cifar import get_cifar10_DVS
from dataloader.caltech101 import get_n_caltech101, create_caltech101_dataloaders
from dataloader.cepdvs import get_cepdvs_dvs, get_cepdvs_gray_attention, get_cepdvs_rgb_attention
from models.snn_models.VGG import VGGSNN, VGGSNNwoAP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from dataloader.dataloader_utils import DataLoaderX


def make_cep_train_holdout(train_loader, test_loader, val_ratio, seed, num_workers, manifest_path):
    """只在 CEP-DVS 原训练编号内按类别划分验证集，并记录实际编号。"""
    train_subset = train_loader.dataset
    test_subset = test_loader.dataset
    full_dataset = train_subset.dataset
    if not hasattr(full_dataset, 'samples'):
        raise ValueError('CEP-DVS 数据集缺少 samples，无法按编号记录固定划分')

    by_class = defaultdict(list)
    for index in train_subset.indices:
        by_class[full_dataset.samples[index][1]].append(index)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for label, indices in sorted(by_class.items()):
        rng.shuffle(indices)
        count = min(max(1, round(len(indices) * val_ratio)), len(indices) - 1) if len(indices) > 1 else 0
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    if not val_indices:
        raise ValueError('训练编号不足，无法划分验证集')

    def ids_and_counts(indices):
        rows = [(int(os.path.splitext(os.path.basename(full_dataset.samples[i][0]))[0]),
                 int(full_dataset.samples[i][1])) for i in indices]
        return sorted(rows), dict(sorted(Counter(label for _, label in rows).items()))

    train_rows, train_counts = ids_and_counts(train_indices)
    val_rows, val_counts = ids_and_counts(val_indices)
    test_rows, test_counts = ids_and_counts(test_subset.indices)
    train_ids = {sample_id for sample_id, _ in train_rows}
    val_ids = {sample_id for sample_id, _ in val_rows}
    test_ids = {sample_id for sample_id, _ in test_rows}
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError('CEP-DVS 训练、验证、测试编号重叠')
    with open(manifest_path, 'w', encoding='utf-8') as file:
        json.dump({'seed': seed, 'val_ratio': val_ratio,
                   'train': train_rows, 'val': val_rows, 'test': test_rows,
                   'class_counts': {'train': train_counts, 'val': val_counts, 'test': test_counts}},
                  file, ensure_ascii=False, indent=2)
    print(f"固定划分: train={len(train_indices)}, val={len(val_indices)}, test={len(test_subset)}")
    print(f"每类数量: train={train_counts}, val={val_counts}, test={test_counts}")
    print(f"划分清单: {manifest_path}")
    batch_size = train_loader.batch_size
    train_loader = DataLoaderX(Subset(full_dataset, train_indices), batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoaderX(Subset(full_dataset, val_indices), batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)
    return train_loader, val_loader

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
                    help='图像尺寸（默认: CIFAR10=48, MNIST=32, Caltech101=48, CEP-DVS=48）')
parser.add_argument('--num_workers', type=int, default=8,
                    help='DataLoader 工作进程数 (Windows 遇到问题可设 0)')

# ======================== 验证集策略（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--val_strategy', type=str, default='none',
                    choices=['none', 'test_as_val', 'train_holdout'],
                    help='验证集策略: none=按训练准确率选模 | train_holdout=从 CEP-DVS 训练编号划验证集 | test_as_val=测试集作验证集')
parser.add_argument('--val_ratio', type=float, default=0.1,
                    help='train_holdout 模式下从训练编号划出的验证比例')

# ======================== Caltech101 专用参数（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--caltech101_dvs_path', type=str, default=None,
                    help='N-Caltech101 DVS 数据集路径 (.pt/.bin 格式); 不提供则使用 get_n_caltech101')
parser.add_argument('--fine_tuning', default='no', type=str, help='Fine-tuning 模式标识')

# ======================== 预训练模型加载（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--pretrained_path', type=str, default=None,
                    help='预训练模型参数路径')
parser.add_argument('--skip_dvs_input', action='store_true', default=False,
                    help='跳过加载 dvs_input 相关参数 (默认加载，加此参数则跳过)')
parser.add_argument('--skip_features', action='store_true', default=False,
                    help='跳过加载 features 相关参数 (默认加载，加此参数则跳过)')
parser.add_argument('--skip_bottleneck', action='store_true', default=False,
                    help='跳过加载 bottleneck 相关参数 (默认加载，加此参数则跳过)')
parser.add_argument('--skip_classifier', action='store_true', default=False,
                    help='跳过加载 classifier 相关参数 (默认加载，加此参数则跳过)')

# ======================== 事件注意力（合并自 train_caltech101_baseline.py）========================
parser.add_argument('--use_event_attention', action='store_true', default=False,
                    help='启用事件中间帧引导注意力 (Event Mid-Frame Guided Attention)')
parser.add_argument('--event_attention_reduction', type=int, default=8,
                    help='事件注意力通道压缩比 (默认 8)')
parser.add_argument('--event_attention_layers', type=int, choices=[2, 8], default=8,
                    help='启用注意力的位置数：前两处或全部八处（默认 8）')
parser.add_argument('--attention_source', choices=['mid_event', 'gray_rgb', 'rgb'], default='mid_event',
                    help='注意力 Query 来源：事件中间两帧，或按编号配对的 CEP-DVS 灰度/RGB 图')
parser.add_argument('--attention_loss_weight', type=float, default=0.0,
                    help='注意力 Query 空间先验 KL 损失权重；0 表示不使用')

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
parser.add_argument('--dvs_data_root', type=str, default=None,
                    help='DVS数据根目录(覆盖dataloader默认硬编码路径); 指向合成DVS如 cifar10_rgb2dvs')

args = parser.parse_args()
if args.attention_loss_weight < 0:
    parser.error('--attention_loss_weight 不能小于 0')
if args.attention_loss_weight > 0 and not args.use_event_attention:
    parser.error('--attention_loss_weight > 0 时必须启用 --use_event_attention')
if args.attention_source in ('gray_rgb', 'rgb') and (args.data_set != 'CEP-DVS' or not args.use_event_attention):
    parser.error('--attention_source gray_rgb/rgb 仅支持 CEP-DVS 且必须启用 --use_event_attention')
if args.val_strategy == 'train_holdout' and args.data_set != 'CEP-DVS':
    parser.error('--val_strategy train_holdout 当前仅支持 CEP-DVS')
if args.val_strategy == 'train_holdout' and not 0 < args.val_ratio < 1:
    parser.error('--val_ratio 必须在 0 与 1 之间')

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
    print(f"事件注意力: {'启用 (' + str(args.event_attention_layers) + '处, reduction=' + str(args.event_attention_reduction) + ')' if args.use_event_attention else '未启用'}")
    print(f"注意力 Query 来源: {args.attention_source}")
    print(f"注意力引导损失权重: {args.attention_loss_weight}")
    if args.pretrained_path:
        print(f"预训练模型: {args.pretrained_path}")
        print(f"  跳过 dvs_input: {args.skip_dvs_input}, features: {args.skip_features}, "
              f"bottleneck: {args.skip_bottleneck}, classifier: {args.skip_classifier} (默认全加载)")
    print('=' * 60 + '\n')

    # 生成日志名称
    eventrpg_tag = f"_EventRPG-mix{args.eventrpg_mix_prob}" if args.use_eventrpg else ""
    attn_tag = f"_EventAttn{args.event_attention_layers}" if args.use_event_attention else ""
    if args.attention_source == 'gray_rgb':
        attn_tag += '_GrayRGBQuery'
    elif args.attention_source == 'rgb':
        attn_tag += '_RGBQuery'
    if args.attention_loss_weight > 0:
        loss_tag = {'mid_event': 'MidKL', 'gray_rgb': 'GrayPriorKL',
                    'rgb': 'RGBPriorKL'}[args.attention_source]
        attn_tag += f"_{loss_tag}{args.attention_loss_weight:g}"
    val_tag = f"_TrainVal{args.val_ratio:g}" if args.val_strategy == 'train_holdout' else ""
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
                f"{val_tag}"
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
            eventrpg_mix_prob=args.eventrpg_mix_prob,
            data_root=args.dvs_data_root
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
                eventrpg_mix_prob=args.eventrpg_mix_prob,
                data_root=args.dvs_data_root
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
        cep_loader = {'mid_event': get_cepdvs_dvs,
                      'gray_rgb': get_cepdvs_gray_attention,
                      'rgb': get_cepdvs_rgb_attention}[args.attention_source]
        train_loader, test_loader = cep_loader(
            args.batch_size,
            train_set_ratio=args.dvs_sample_ratio,
            img_size=args.img_shape,
            time_bins=args.T,
            split_ratio=0.9
        )
    else:
        raise ValueError(f"不支持的数据集: {args.data_set}")

    # 根据验证集策略设置 val_loader
    if args.val_strategy == 'train_holdout':
        train_loader, val_loader = make_cep_train_holdout(
            train_loader, test_loader, args.val_ratio, args.seed, args.num_workers,
            os.path.join(checkpoint_dir, f'{log_name}_split.json'))
    elif args.val_strategy == 'test_as_val':
        val_loader = test_loader
        print(f"  验证集策略: 使用测试集作为每轮验证集选择最佳模型")
    else:
        print(f"  验证集策略: 不使用验证集，基于训练集准确率保存最佳模型")

    print(f"✓ 训练集样本数: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    if val_loader is not None:
        print(f"✓ 验证集样本数: {len(val_loader.dataset)} ({len(val_loader)} batches)")
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
        event_attention_reduction=args.event_attention_reduction,
        event_attention_layers=args.event_attention_layers,
        attention_source=args.attention_source
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
        # 默认加载全部模块（edge_extractor 始终排除）
        # 通过 --skip_* 标志排除对应模块
        exclude_modules = ['edge_extractor']
        if args.skip_dvs_input:
            exclude_modules.append('dvs_input')
        if args.skip_features:
            exclude_modules.append('features')
        if args.skip_bottleneck:
            exclude_modules.append('bottleneck')
        if args.skip_classifier:
            exclude_modules.append('classifier')

        # 过滤掉不匹配的键和需要排除的模块
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict
                           and v.shape == model_dict[k].shape
                           and not any(exclude_module in k for exclude_module in exclude_modules)}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        print(f"  成功加载: {len(pretrained_dict)}/{len(model_dict)} 个参数")
        if args.use_event_attention:
            attention_keys = [k for k in model_dict if k.startswith('event_attn_')]
            loaded_attention = [k for k in attention_keys if k in pretrained_dict]
            print(f"  注意力参数加载: {len(loaded_attention)}/{len(attention_keys)}")
            if len(loaded_attention) != len(attention_keys):
                print("  注意: 部分注意力参数未从预训练权重加载，将随机初始化后训练")
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

    if args.val_strategy in ('test_as_val', 'train_holdout'):
        # 基于所选验证集准确率保存最佳模型
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

    # 测试前恢复本次选出的模型，避免将历史最佳训练值与末轮测试值混用。
    if os.path.exists(model_path):
        print(f"加载本次选出的模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'epoch' in checkpoint:
            print(f"选中轮次: {checkpoint['epoch'] + 1}")
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)
    elif args.val_strategy == 'train_holdout':
        raise FileNotFoundError(f"未找到验证集选出的模型: {model_path}")

    test_loss, test_acc = trainer.test(test_loader)
    print(f'测试损失: {test_loss:.5f}')
    print(f'测试精度: {test_acc:.3f} ({test_acc * 100:.2f}%)')
    print(f'最佳训练准确率: {trainer.best_train_acc:.3f}')
    print("=" * 60)

    # 记录最终测试结果
    writer.add_scalar(tag="final_test/accuracy", scalar_value=test_acc, global_step=0)
    writer.add_scalar(tag="final_test/loss", scalar_value=test_loss, global_step=0)
    if args.val_strategy in ('test_as_val', 'train_holdout'):
        writer.add_scalar(tag="final_test/val_accuracy", scalar_value=best_val_acc, global_step=0)
        writer.add_scalar(tag="final_test/train_accuracy", scalar_value=trainer.best_train_acc, global_step=0)

    writer.close()
    print(f"\n训练完成！模型已保存至: {model_path}")
