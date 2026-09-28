# -*- coding: utf-8 -*-
"""
CIFAR10 RGB 按类别预处理脚本

将 torchvision CIFAR10 的 RGB 图像按类别组织成 .pt 张量，
目录形式与 DVS 数据（{train,test}/{类别}/*.pt）完全对称，
便于按类别管理、per-class 采样，以及与 DVS 读取逻辑统一。

输出结构：
  output_dir/
    train/{class_name}/{idx:05d}.pt   (50000 张, (3, img_size, img_size) float32 [0,1])
    test/{class_name}/{idx:05d}.pt    (10000 张)

存储格式：纯张量（不存 (data, label) 元组），标签由类别文件夹名按 sorted() 字母序推断，
         与 dataloader/cifar.py 的 DVSCifar10v1 存储/读取方式对称。
类别名：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

使用方法：
  python data_process/preprocess_cifar10_rgb2cls.py
  python data_process/preprocess_cifar10_rgb2cls.py --output_dir /path/to/cifar10_rgb --img_size 48
"""

import argparse
import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms


def preprocess_split(cifar10_root, output_dir, train, img_size):
    """处理单个 split（train 或 test），按类别保存为 .pt 纯张量。"""
    split_name = 'train' if train else 'test'
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # (3, H, W) float32, 范围 [0, 1]
    ])

    dataset = datasets.CIFAR10(cifar10_root, train=train, transform=transform, download=False)
    classes = dataset.classes  # ['airplane', 'automobile', ...] (小写, 与 DVS 类别名一致)

    # 为每个类别创建输出目录
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split_name, cls), exist_ok=True)

    counts = {cls: 0 for cls in classes}
    for idx in tqdm(range(len(dataset)), desc=f"{split_name}", ncols=100):
        img, label = dataset[idx]
        cls_name = classes[label]
        save_idx = counts[cls_name]
        save_path = os.path.join(output_dir, split_name, cls_name, f"{save_idx:05d}.pt")
        torch.save(img, save_path)  # 纯张量，标签由文件夹名推断
        counts[cls_name] += 1

    total = sum(counts.values())
    print(f"\n{split_name} 完成: 共 {total} 张")
    for cls in sorted(classes):
        print(f"  {cls}: {counts[cls]}")

    # 抽样验证
    first_cls = classes[0]
    sample_path = os.path.join(output_dir, split_name, first_cls, "00000.pt")
    if os.path.exists(sample_path):
        data = torch.load(sample_path, weights_only=True)
        print(f"  验证 {first_cls}/00000.pt: shape={tuple(data.shape)}, "
              f"dtype={data.dtype}, range=[{data.min():.3f}, {data.max():.3f}]")
    return total


def main():
    parser = argparse.ArgumentParser(description='CIFAR10 RGB 按类别预处理（与 DVS 目录对称）')
    parser.add_argument('--cifar10_root', type=str,
                        default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10',
                        help='torchvision CIFAR10 根目录（含 data_batch_*.bin 等）')
    parser.add_argument('--output_dir', type=str,
                        default='/home/user/kpm/kpm/Dataset/CIFAR10/cifar10_rgb',
                        help='输出目录（生成 train/ 和 test/ 两个子目录）')
    parser.add_argument('--img_size', type=int, default=48, help='图像尺寸（边长）')
    args = parser.parse_args()

    print(f"CIFAR10 RGB 按类别预处理")
    print(f"  输入: {args.cifar10_root}")
    print(f"  输出: {args.output_dir}")
    print(f"  尺寸: {args.img_size}x{args.img_size}")
    print(f"  存储: 纯张量 (3,{args.img_size},{args.img_size}) float32 [0,1]，标签由文件夹名推断\n")

    n_train = preprocess_split(args.cifar10_root, args.output_dir, train=True, img_size=args.img_size)
    n_test = preprocess_split(args.cifar10_root, args.output_dir, train=False, img_size=args.img_size)

    print(f"\n{'=' * 60}")
    print(f"全部完成！train={n_train}, test={n_test}")
    print(f"输出目录: {args.output_dir}")
    print(f"结构: {{train,test}}/{{类别}}/*.pt")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
