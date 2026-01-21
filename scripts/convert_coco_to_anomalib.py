#!/usr/bin/env python3
"""
将 COCO 格式的数据集转换为 anomalib 需要的文件夹结构

用法:
    python scripts/convert_coco_to_anomalib.py \
        --coco_json /root/autodl-tmp/YOSUN/train/_annotations.coco.json \
        --source_dir /root/autodl-tmp/YOSUN/train \
        --output_dir /root/autodl-tmp/YOSUN/anomalib_format \
        --train_ratio 0.9
"""

import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def convert_coco_to_anomalib(
    coco_json_path: str,
    source_image_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    additional_coco_files: List[Tuple[str, str]] = None
):
    """
    将 COCO 格式的数据集转换为 anomalib 需要的文件夹结构
    
    Args:
        coco_json_path: COCO 标注文件路径（主文件，用于训练集）
        source_image_dir: 原始图像所在目录
        output_dir: 输出目录
        train_ratio: 训练集比例（默认 0.8）
        seed: 随机种子
        additional_coco_files: 额外的 (coco_json_path, image_dir) 元组列表，用于合并其他目录的数据
    """
    print(f"正在读取 COCO 标注文件: {coco_json_path}")
    # 读取 COCO JSON 文件
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 合并额外的 COCO 文件（如果有）
    if additional_coco_files:
        print(f"\n正在合并 {len(additional_coco_files)} 个额外的标注文件...")
        for add_json, add_dir in additional_coco_files:
            print(f"  读取: {add_json}")
            with open(add_json, 'r', encoding='utf-8') as f:
                add_data = json.load(f)
            
            # 合并图像（需要调整 image_id 以避免冲突）
            max_image_id = max([img['id'] for img in coco_data['images']]) if coco_data['images'] else -1
            max_ann_id = max([ann['id'] for ann in coco_data['annotations']]) if coco_data['annotations'] else -1
            
            # 调整 image_id 并合并
            for img in add_data['images']:
                old_id = img['id']
                new_id = max_image_id + 1 + old_id
                img['id'] = new_id
                # 更新对应的 annotations
                for ann in add_data['annotations']:
                    if ann['image_id'] == old_id:
                        ann['image_id'] = new_id
                        ann['id'] = max_ann_id + 1 + ann['id']
                coco_data['images'].append(img)
            
            # 合并 annotations
            coco_data['annotations'].extend(add_data['annotations'])
            
            # 合并图像文件路径（标记来源目录）
            for img in add_data['images']:
                img['_source_dir'] = add_dir
    
    print(f"找到 {len(coco_data['images'])} 张图像")
    print(f"找到 {len(coco_data['annotations'])} 个标注")
    print(f"类别: {[cat['name'] for cat in coco_data['categories']]}")
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    train_good_dir = output_path / "train" / "good"
    test_good_dir = output_path / "test" / "good"
    test_abnormal_dir = output_path / "test" / "abnormal"
    
    for dir_path in [train_good_dir, test_good_dir, test_abnormal_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {dir_path}")
    
    # 建立 image_id 到 category_id 的映射
    # 一个图像可能有多个标注，我们取第一个（或多数类别）
    image_to_categories = defaultdict(list)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_to_categories[image_id].append(category_id)
    
    # 对于有多个标注的图像，取多数类别
    image_to_category = {}
    for image_id, categories in image_to_categories.items():
        # 取最常见的类别
        category_id = max(set(categories), key=categories.count)
        image_to_category[image_id] = category_id
    
    # 建立 image_id 到 file_name 和 source_dir 的映射
    image_id_to_filename = {}
    image_id_to_source_dir = {}
    for img in coco_data['images']:
        image_id_to_filename[img['id']] = img['file_name']
        # 如果图像来自额外目录，使用该目录，否则使用主目录
        image_id_to_source_dir[img['id']] = img.get('_source_dir', source_image_dir)
    
    # 建立 category_id 到 category_name 的映射
    category_id_to_name = {}
    for cat in coco_data['categories']:
        category_id_to_name[cat['id']] = cat['name']
    
    print(f"\n类别映射: {category_id_to_name}")
    
    # 分类图像：category_id=1 是正常(ok)，category_id=0 是异常(ng)
    normal_images: List[Tuple[int, Path]] = []
    abnormal_images: List[Tuple[int, Path]] = []
    
    source_path = Path(source_image_dir)
    missing_files = []
    
    for image_id, category_id in image_to_category.items():
        filename = image_id_to_filename.get(image_id)
        if filename is None:
            print(f"警告: 图像 ID {image_id} 没有对应的文件名")
            continue
        
        # 使用对应的源目录
        img_source_dir = image_id_to_source_dir.get(image_id, source_image_dir)
        source_file = Path(img_source_dir) / filename
        if not source_file.exists():
            missing_files.append(str(source_file))
            continue
        
        category_name = category_id_to_name.get(category_id, f"unknown_{category_id}")
        
        if category_id == 1 or category_name == "ok":  # ok (正常)
            normal_images.append((image_id, source_file))
        elif category_id == 0 or category_name == "ng":  # ng (异常)
            abnormal_images.append((image_id, source_file))
        else:
            print(f"警告: 未知类别 ID {category_id} ({category_name})")
    
    if missing_files:
        print(f"\n警告: 有 {len(missing_files)} 个文件未找到（前5个）:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    # 划分训练集和测试集（只对正常样本划分，异常样本全部作为测试集）
    random.seed(seed)
    random.shuffle(normal_images)
    
    split_idx = int(len(normal_images) * train_ratio)
    train_normal = normal_images[:split_idx]
    test_normal = normal_images[split_idx:]
    
    print(f"\n开始复制文件...")
    print(f"训练集正常样本: {len(train_normal)} 张")
    print(f"测试集正常样本: {len(test_normal)} 张")
    print(f"测试集异常样本: {len(abnormal_images)} 张")
    
    # 复制文件到相应目录
    copied_count = 0
    
    print(f"\n正在复制训练集正常样本...")
    for _, src_file in train_normal:
        dst_file = train_good_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"正在复制测试集正常样本...")
    for _, src_file in test_normal:
        dst_file = test_good_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"正在复制测试集异常样本...")
    for _, src_file in abnormal_images:
        dst_file = test_abnormal_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"\n{'='*60}")
    print(f"转换完成！")
    print(f"{'='*60}")
    print(f"训练集正常样本: {len(train_normal)} 张 -> {train_good_dir}")
    print(f"测试集正常样本: {len(test_normal)} 张 -> {test_good_dir}")
    print(f"测试集异常样本: {len(abnormal_images)} 张 -> {test_abnormal_dir}")
    print(f"总计: {copied_count} 张图像")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="将 COCO 格式的数据集转换为 anomalib 需要的文件夹结构"
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="COCO 标注文件路径"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="原始图像所在目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认 0.8）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）"
    )
    parser.add_argument(
        "--additional_json",
        type=str,
        nargs="+",
        help="额外的 COCO 标注文件路径（可以多个）"
    )
    parser.add_argument(
        "--additional_dir",
        type=str,
        nargs="+",
        help="额外的图像目录（与 --additional_json 一一对应）"
    )
    
    args = parser.parse_args()
    
    # 处理额外的 COCO 文件
    additional_files = None
    if args.additional_json:
        if args.additional_dir and len(args.additional_json) == len(args.additional_dir):
            additional_files = list(zip(args.additional_json, args.additional_dir))
        else:
            print("警告: --additional_json 和 --additional_dir 数量不匹配，忽略额外文件")
    
    convert_coco_to_anomalib(
        coco_json_path=args.coco_json,
        source_image_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        additional_coco_files=additional_files
    )


if __name__ == "__main__":
    main()

