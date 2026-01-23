"""
DINOv3 异常检测示例

展示三种检测模式：cls / patch / combined
"""

from pathlib import Path
from hq_anomaly_detection import AnomalyDetector


def main():
    # 1. 创建检测器
    print("初始化检测器...")
    detector = AnomalyDetector(config_path="configs/dinov3_patchcore_style.yaml")
    
    # 2. 训练（根据配置文件中的 detection_mode 自动选择模式）
    print("\n训练模型...")
    detector.train()
    
    # 3. 保存
    checkpoint_path = "./outputs/checkpoint.pth"
    detector.save_model(checkpoint_path)
    print(f"模型已保存: {checkpoint_path}")
    
    # 4. 加载并推理
    detector.load_model(checkpoint_path)
    
    result = detector.detect(
        image_path="./data/test/image.png",
        threshold=0.5,
    )
    
    print(f"\n检测结果:")
    print(f"  异常分数: {result['anomaly_score']:.4f}")
    if 'cls_score' in result:
        print(f"  CLS 分数: {result['cls_score']:.4f}")
    if 'patch_score' in result:
        print(f"  Patch 分数: {result['patch_score']:.4f}")
    if 'prediction' in result:
        print(f"  预测: {'异常' if result['prediction'] else '正常'}")


if __name__ == "__main__":
    main()
