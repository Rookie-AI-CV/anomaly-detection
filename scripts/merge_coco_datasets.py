#!/usr/bin/env python3
"""
合并多个 COCO 格式数据集为一个

用法:
    python scripts/merge_coco_datasets.py \
        --datasets /path/to/dataset1 /path/to/dataset2 /path/to/dataset3 \
        --output_dir /path/to/merged_output \
        --train_ratio 0.9

或使用配置文件:
    python scripts/merge_coco_datasets.py \
        --config datasets.txt \
        --output_dir /path/to/merged_output
"""

import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def find_coco_json(dataset_dir: Path) -> Optional[Path]:
    """在目录中查找 COCO JSON 文件"""
    common_names = [
        "_annotations.coco.json",
        "annotations.coco.json",
        "_annotations.json",
        "annotations.json",
        "coco.json"
    ]
    
    # 先检查当前目录
    for name in common_names:
        json_path = dataset_dir / name
        if json_path.exists():
            return json_path
    
    # 尝试查找任何包含 "annotation" 的 JSON 文件
    for json_path in dataset_dir.glob("*.json"):
        if "annotation" in json_path.name.lower():
            return json_path
    
    # 检查是否有 train/test/valid 子目录
    split_dirs = ['train', 'test', 'valid', 'val']
    for split_dir in split_dirs:
        split_path = dataset_dir / split_dir
        if split_path.exists() and split_path.is_dir():
            for name in common_names:
                json_path = split_path / name
                if json_path.exists():
                    return json_path
            for json_path in split_path.glob("*.json"):
                if "annotation" in json_path.name.lower():
                    return json_path
    
    return None


def find_image_dirs(dataset_dir: Path) -> List[Path]:
    """查找图像目录（支持train/test/valid子目录结构）"""
    split_dirs = ['train', 'test', 'valid', 'val']
    image_dirs = []
    
    # 检查是否有 train/test/valid 子目录
    has_split_dirs = any((dataset_dir / d).exists() and (dataset_dir / d).is_dir() for d in split_dirs)
    
    if has_split_dirs:
        # 如果有子目录，遍历所有子目录
        for split_dir in split_dirs:
            split_path = dataset_dir / split_dir
            if split_path.exists() and split_path.is_dir():
                # 检查子目录下是否有图像文件
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                has_images = any(f.suffix.lower() in image_extensions for f in split_path.rglob('*') if f.is_file())
                if has_images:
                    image_dirs.append(split_path)
    else:
        # 如果没有子目录，使用当前目录
        image_dirs.append(dataset_dir)
    
    return image_dirs if image_dirs else [dataset_dir]


def merge_coco_datasets(
    dataset_dirs: List[str],
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    coco_json_names: Optional[List[str]] = None,
    image_dirs: Optional[List[str]] = None,
    use_indexed_names: bool = False
):
    """
    合并多个 COCO 格式数据集
    
    Args:
        dataset_dirs: 数据集目录列表（每个目录包含 COCO JSON 和图像）
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
        coco_json_names: 可选的 COCO JSON 文件名列表（与 dataset_dirs 对应）
        image_dirs: 可选的图像目录列表（与 dataset_dirs 对应，如果与数据集目录不同）
    """
    output_path = Path(output_dir)
    train_good_dir = output_path / "train" / "good"
    test_good_dir = output_path / "test" / "good"
    test_abnormal_dir = output_path / "test" / "abnormal"
    
    for dir_path in [train_good_dir, test_good_dir, test_abnormal_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    all_coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    max_image_id = -1
    max_ann_id = -1
    max_cat_id = -1
    category_name_to_id = {}
    dataset_info = []
    
    print(f"正在合并 {len(dataset_dirs)} 个数据集...\n")
    
    for idx, dataset_dir in enumerate(dataset_dirs):
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            print(f"警告: 数据集目录不存在，跳过: {dataset_dir}")
            continue
        
        # 查找 COCO JSON 文件
        if coco_json_names and idx < len(coco_json_names):
            coco_json_path = dataset_path / coco_json_names[idx]
        else:
            coco_json_path = find_coco_json(dataset_path)
        
        if not coco_json_path or not coco_json_path.exists():
            print(f"警告: 在 {dataset_dir} 中未找到 COCO JSON 文件，跳过")
            continue
        
        # 确定图像目录
        if image_dirs and idx < len(image_dirs):
            image_dir = Path(image_dirs[idx])
        else:
            # 如果JSON在子目录中，图像目录应该是JSON所在目录
            # 如果JSON在根目录，检查是否有train/test/valid子目录
            if coco_json_path.parent != dataset_path:
                # JSON在子目录中，使用子目录作为图像目录
                image_dir = coco_json_path.parent
            else:
                # JSON在根目录，检查是否有子目录结构
                found_image_dirs = find_image_dirs(dataset_path)
                if len(found_image_dirs) == 1:
                    image_dir = found_image_dirs[0]
                else:
                    # 有多个子目录，使用JSON所在目录或根目录
                    image_dir = dataset_path
        
        print(f"数据集 {idx+1}/{len(dataset_dirs)}: {dataset_dir}")
        print(f"  COCO JSON: {coco_json_path}")
        print(f"  图像目录: {image_dir}")
        
        with open(coco_json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 更新类别映射（合并相同名称的类别）
        cat_id_mapping = {}
        for cat in coco_data.get('categories', []):
            cat_name = cat['name']
            if cat_name not in category_name_to_id:
                max_cat_id += 1
                category_name_to_id[cat_name] = max_cat_id
                all_coco_data['categories'].append({
                    'id': max_cat_id,
                    'name': cat_name,
                    'supercategory': cat.get('supercategory', '')
                })
            cat_id_mapping[cat['id']] = category_name_to_id[cat_name]
        
        # 更新 image_id 和 image 路径
        image_id_mapping = {}
        for img in coco_data.get('images', []):
            old_id = img['id']
            max_image_id += 1
            new_id = max_image_id
            image_id_mapping[old_id] = new_id
            
            img['id'] = new_id
            img['_source_dir'] = str(image_dir)
            img['_dataset_name'] = dataset_path.name
            all_coco_data['images'].append(img)
        
        # 更新 annotation 的 image_id 和 category_id
        for ann in coco_data.get('annotations', []):
            old_ann_id = ann['id']
            max_ann_id += 1
            
            ann['id'] = max_ann_id
            ann['image_id'] = image_id_mapping.get(ann['image_id'], ann['image_id'])
            ann['category_id'] = cat_id_mapping.get(ann['category_id'], ann['category_id'])
            all_coco_data['annotations'].append(ann)
        
        dataset_info.append({
            'name': dataset_path.name,
            'images': len(coco_data.get('images', [])),
            'annotations': len(coco_data.get('annotations', []))
        })
        print(f"  图像: {len(coco_data.get('images', []))}, 标注: {len(coco_data.get('annotations', []))}\n")
    
    print(f"合并完成！")
    print(f"总图像数: {len(all_coco_data['images'])}")
    print(f"总标注数: {len(all_coco_data['annotations'])}")
    print(f"类别: {[cat['name'] for cat in all_coco_data['categories']]}\n")
    
    # 建立映射
    image_to_categories = defaultdict(list)
    for ann in all_coco_data['annotations']:
        image_to_categories[ann['image_id']].append(ann['category_id'])
    
    image_to_category = {}
    for image_id, categories in image_to_categories.items():
        category_id = max(set(categories), key=categories.count)
        image_to_category[image_id] = category_id
    
    image_id_to_info = {}
    for img in all_coco_data['images']:
        image_id_to_info[img['id']] = {
            'file_name': img['file_name'],
            'source_dir': img.get('_source_dir'),
            'dataset_name': img.get('_dataset_name', 'unknown')
        }
    
    category_id_to_name = {cat['id']: cat['name'] for cat in all_coco_data['categories']}
    
    # 分类图像
    normal_images: List[Tuple[int, Path]] = []
    abnormal_images: List[Tuple[int, Path]] = []
    missing_files = []
    
    for image_id, category_id in image_to_category.items():
        info = image_id_to_info[image_id]
        file_name = info['file_name']
        source_dir = Path(info['source_dir'])
        
        # 尝试多个可能的路径
        possible_paths = [
            source_dir / file_name,  # 直接路径
            source_dir.parent / file_name,  # 上一级目录
        ]
        
        # 如果source_dir是子目录，也尝试在根目录查找
        if source_dir.parent.name in ['train', 'test', 'valid', 'val']:
            possible_paths.append(source_dir.parent.parent / file_name)
        
        # 如果file_name包含子目录路径，尝试直接使用
        if '/' in file_name or '\\' in file_name:
            possible_paths.append(source_dir.parent / file_name)
        
        source_file = None
        for path in possible_paths:
            if path.exists():
                source_file = path
                break
        
        # 如果还是找不到，尝试在source_dir及其子目录中递归查找
        if not source_file:
            file_name_only = Path(file_name).name
            for img_file in source_dir.rglob(file_name_only):
                if img_file.name == file_name_only:
                    source_file = img_file
                    break
            # 也尝试在父目录中查找
            if not source_file and source_dir.parent.exists():
                for img_file in source_dir.parent.rglob(file_name_only):
                    if img_file.name == file_name_only:
                        source_file = img_file
                        break
        
        if not source_file:
            missing_files.append(str(source_file))
            continue
        
        # 类别ID 0为正常，其他都是异常
        if category_id == 0:
            normal_images.append((image_id, source_file))
        else:
            abnormal_images.append((image_id, source_file))
    
    if missing_files:
        print(f"\n警告: 有 {len(missing_files)} 个文件未找到（前5个）:")
        for f in missing_files[:5]:
            print(f"  - {f}")
    
    # 划分训练集和测试集
    random.seed(seed)
    random.shuffle(normal_images)
    
    split_idx = int(len(normal_images) * train_ratio)
    train_normal = normal_images[:split_idx]
    test_normal = normal_images[split_idx:]
    
    print(f"\n开始复制文件...")
    print(f"训练集正常样本: {len(train_normal)} 张")
    print(f"测试集正常样本: {len(test_normal)} 张")
    print(f"测试集异常样本: {len(abnormal_images)} 张\n")
    
    copied_count = 0
    
    print(f"正在复制训练集正常样本...")
    train_idx = 1
    for _, src_file in train_normal:
        if use_indexed_names:
            ext = src_file.suffix
            dst_file = train_good_dir / f"{train_idx:06d}{ext}"
            train_idx += 1
        else:
            dst_file = train_good_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"正在复制测试集正常样本...")
    test_normal_idx = 1
    for _, src_file in test_normal:
        if use_indexed_names:
            ext = src_file.suffix
            dst_file = test_good_dir / f"{test_normal_idx:06d}{ext}"
            test_normal_idx += 1
        else:
            dst_file = test_good_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"正在复制测试集异常样本...")
    abnormal_idx = 1
    for _, src_file in abnormal_images:
        if use_indexed_names:
            ext = src_file.suffix
            dst_file = test_abnormal_dir / f"{abnormal_idx:06d}{ext}"
            abnormal_idx += 1
        else:
            dst_file = test_abnormal_dir / src_file.name
        shutil.copy2(src_file, dst_file)
        copied_count += 1
        if copied_count % 100 == 0:
            print(f"  已复制 {copied_count} 张...")
    
    print(f"\n{'='*60}")
    print(f"合并完成！")
    print(f"{'='*60}")
    print(f"数据集统计:")
    for info in dataset_info:
        print(f"  {info['name']}: {info['images']} 图像, {info['annotations']} 标注")
    print(f"\n输出统计:")
    print(f"  训练集正常样本: {len(train_normal)} 张 -> {train_good_dir}")
    print(f"  测试集正常样本: {len(test_normal)} 张 -> {test_good_dir}")
    print(f"  测试集异常样本: {len(abnormal_images)} 张 -> {test_abnormal_dir}")
    print(f"  总计: {copied_count} 张图像")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")


def load_datasets_from_config(config_path: str) -> Tuple[List[str], Optional[List[str]], Optional[List[str]]]:
    """从配置文件加载数据集列表"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    datasets = []
    coco_jsons = []
    image_dirs = []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(',')
            if len(parts) == 1:
                datasets.append(parts[0].strip())
            elif len(parts) == 2:
                datasets.append(parts[0].strip())
                coco_jsons.append(parts[1].strip())
            elif len(parts) >= 3:
                datasets.append(parts[0].strip())
                coco_jsons.append(parts[1].strip())
                image_dirs.append(parts[2].strip())
    
    coco_jsons = coco_jsons if coco_jsons else None
    image_dirs = image_dirs if image_dirs else None
    
    return datasets, coco_jsons, image_dirs


def main():
    parser = argparse.ArgumentParser(
        description="合并多个 COCO 格式数据集为一个"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="数据集目录列表（每个目录包含 COCO JSON 和图像）"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径（每行一个数据集目录，可选: dataset_dir,coco_json,image_dir）"
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        nargs="+",
        help="COCO JSON 文件路径列表（与 --datasets 对应）"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        nargs="+",
        help="图像目录列表（与 --datasets 对应，如果与数据集目录不同）"
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
        "--use-indexed-names",
        action="store_true",
        help="使用序号命名文件（000001.jpg, 000002.jpg 等）"
    )
    
    args = parser.parse_args()
    
    if args.config:
        datasets, coco_jsons, image_dirs = load_datasets_from_config(args.config)
    elif args.datasets:
        datasets = args.datasets
        coco_jsons = args.coco_json if args.coco_json else None
        image_dirs = args.image_dir if args.image_dir else None
    else:
        parser.error("需要提供 --datasets 或 --config")
    
    merge_coco_datasets(
        dataset_dirs=datasets,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        coco_json_names=coco_jsons,
        image_dirs=image_dirs,
        use_indexed_names=args.use_indexed_names
    )


if __name__ == "__main__":
    main()
