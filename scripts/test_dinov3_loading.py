#!/usr/bin/env python3
"""
测试 DINOv3 模型参数是否正确加载

用法:
    python scripts/test_dinov3_loading.py \
        --model-name dinov3_huge \
        --model-path /path/to/model.safetensors
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from env_config import load_env
load_env()

from hq_anomaly_detection.models.dinov3.feature_extractor import DINOv3FeatureExtractor


def check_model_weights(model, model_name: str):
    """检查模型权重是否正确加载"""
    print(f"\n{'='*60}")
    print(f"检查模型权重: {model_name}")
    print(f"{'='*60}")
    
    total_params = 0
    loaded_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    
    param_stats = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 检查参数是否已初始化（非零）
        non_zero = torch.count_nonzero(param).item()
        loaded_params += non_zero
        zero_params += (param.numel() - non_zero)
        
        # 检查NaN和Inf
        nan_count = torch.isnan(param).sum().item()
        inf_count = torch.isinf(param).sum().item()
        nan_params += nan_count
        inf_params += inf_count
        
        param_stats.append({
            'name': name,
            'shape': tuple(param.shape),
            'numel': param.numel(),
            'non_zero': non_zero,
            'zero': param.numel() - non_zero,
            'nan': nan_count,
            'inf': inf_count,
            'mean': param.mean().item() if param.numel() > 0 else 0.0,
            'std': param.std().item() if param.numel() > 0 else 0.0,
            'min': param.min().item() if param.numel() > 0 else 0.0,
            'max': param.max().item() if param.numel() > 0 else 0.0,
        })
    
    print(f"\n总体统计:")
    print(f"  总参数数量: {total_params:,}")
    print(f"  非零参数: {loaded_params:,} ({loaded_params/total_params*100:.2f}%)")
    print(f"  零参数: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    print(f"  NaN参数: {nan_params:,}")
    print(f"  Inf参数: {inf_params:,}")
    
    # 检查关键层
    print(f"\n关键层检查:")
    key_layers = ['patch_embed', 'blocks.0', 'blocks.1', 'norm', 'head']
    for layer_name in key_layers:
        found = False
        for stat in param_stats:
            if layer_name in stat['name']:
                found = True
                print(f"  {stat['name']}:")
                print(f"    形状: {stat['shape']}, 非零: {stat['non_zero']}/{stat['numel']} "
                      f"({stat['non_zero']/stat['numel']*100:.2f}%)")
                print(f"    均值: {stat['mean']:.6f}, 标准差: {stat['std']:.6f}")
                print(f"    范围: [{stat['min']:.6f}, {stat['max']:.6f}]")
                if stat['nan'] > 0:
                    print(f"    ⚠️  警告: 发现 {stat['nan']} 个NaN值")
                if stat['inf'] > 0:
                    print(f"    ⚠️  警告: 发现 {stat['inf']} 个Inf值")
                break
        if not found:
            print(f"  {layer_name}: 未找到")
    
    # 判断模型是否正确加载
    is_valid = True
    issues = []
    
    if loaded_params == 0:
        is_valid = False
        issues.append("所有参数都是零，模型可能未正确加载")
    
    if loaded_params / total_params < 0.01:
        is_valid = False
        issues.append(f"非零参数比例过低 ({loaded_params/total_params*100:.2f}%)")
    
    if nan_params > 0:
        is_valid = False
        issues.append(f"发现 {nan_params} 个NaN参数")
    
    if inf_params > 0:
        is_valid = False
        issues.append(f"发现 {inf_params} 个Inf参数")
    
    print(f"\n{'='*60}")
    if is_valid:
        print("✓ 模型权重检查通过")
    else:
        print("✗ 模型权重检查失败:")
        for issue in issues:
            print(f"  - {issue}")
    print(f"{'='*60}\n")
    
    return is_valid, param_stats


def test_forward_pass(model, device):
    """测试前向传播"""
    print(f"\n{'='*60}")
    print("测试前向传播")
    print(f"{'='*60}")
    
    # 创建测试图像（尺寸需要能被patch size 16整除）
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建随机测试图像
    test_image = Image.new('RGB', (224, 224), color='red')
    image_tensor = transform(test_image).unsqueeze(0).to(device)
    
    print(f"输入形状: {image_tensor.shape}")
    print(f"设备: {device}")
    
    try:
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
        
        print(f"输出形状: {output.shape}")
        print(f"输出统计:")
        print(f"  均值: {output.mean().item():.6f}")
        print(f"  标准差: {output.std().item():.6f}")
        print(f"  最小值: {output.min().item():.6f}")
        print(f"  最大值: {output.max().item():.6f}")
        
        # 检查输出是否有效
        if torch.isnan(output).any():
            print("  ✗ 输出包含NaN值")
            return False
        if torch.isinf(output).any():
            print("  ✗ 输出包含Inf值")
            return False
        if output.abs().max() > 1e6:
            print(f"  ⚠️  警告: 输出值过大 (max={output.abs().max().item():.2f})")
        
        print("  ✓ 前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_forward(model, device, batch_size=4):
    """测试批量前向传播"""
    print(f"\n{'='*60}")
    print(f"测试批量前向传播 (batch_size={batch_size})")
    print(f"{'='*60}")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建批量测试图像
    test_images = [Image.new('RGB', (224, 224), color='blue') for _ in range(batch_size)]
    image_tensors = torch.stack([transform(img) for img in test_images]).to(device)
    
    print(f"输入形状: {image_tensors.shape}")
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensors)
        
        print(f"输出形状: {outputs.shape}")
        print(f"输出统计:")
        print(f"  均值: {outputs.mean().item():.6f}")
        print(f"  标准差: {outputs.std().item():.6f}")
        print(f"  最小值: {outputs.min().item():.6f}")
        print(f"  最大值: {outputs.max().item():.6f}")
        
        if torch.isnan(outputs).any():
            print("  ✗ 输出包含NaN值")
            return False
        if torch.isinf(outputs).any():
            print("  ✗ 输出包含Inf值")
            return False
        
        print("  ✓ 批量前向传播测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 批量前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 DINOv3 模型参数加载")
    parser.add_argument("--model-name", type=str, default="dinov3_huge",
                       help="模型名称 (dinov3_base, dinov3_large, dinov3_huge)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型权重路径 (可选)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--device-id", type=int, default=0,
                       help="GPU设备ID")
    args = parser.parse_args()
    
    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    
    print(f"{'='*60}")
    print("DINOv3 模型加载测试")
    print(f"{'='*60}")
    print(f"模型名称: {args.model_name}")
    print(f"模型路径: {args.model_path}")
    print(f"设备: {device}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("正在加载模型...")
    try:
        extractor = DINOv3FeatureExtractor(
            model_name=args.model_name,
            model_path=args.model_path
        )
        extractor = extractor.to(device)
        print("✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 检查模型权重
    is_valid, param_stats = check_model_weights(extractor.backbone_model, args.model_name)
    
    # 测试前向传播
    forward_ok = test_forward_pass(extractor, device)
    
    # 测试批量前向传播
    batch_ok = test_batch_forward(extractor, device, batch_size=4)
    
    # 总结
    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"权重检查: {'✓ 通过' if is_valid else '✗ 失败'}")
    print(f"前向传播: {'✓ 通过' if forward_ok else '✗ 失败'}")
    print(f"批量前向传播: {'✓ 通过' if batch_ok else '✗ 失败'}")
    
    if is_valid and forward_ok and batch_ok:
        print(f"\n✓ 所有测试通过！模型参数已正确加载。")
        return 0
    else:
        print(f"\n✗ 部分测试失败，请检查模型加载。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
