"""
DINOv3 异常检测示例（CLS 模式）
"""

from hq_anomaly_detection import AnomalyDetector


def main():
    # 1. 创建检测器
    print("初始化检测器...")
    detector = AnomalyDetector(config_path="configs/dinov3_image_level.yaml")
    
    # 2. 训练
    print("\n训练模型...")
    detector.train()
    
    # 3. 保存
    checkpoint = "./outputs/checkpoint.pth"
    detector.save_model(checkpoint)
    print(f"模型已保存: {checkpoint}")
    
    # 4. 加载并推理
    detector.load_model(checkpoint)
    
    result = detector.detect("./data/test/image.png", threshold=0.5)
    
    print(f"\n检测结果:")
    print(f"  异常分数: {result['anomaly_score']:.4f}")
    if 'prediction' in result:
        print(f"  预测: {'异常' if result['prediction'] else '正常'}")
    
    # 批量推理
    print("\n批量推理...")
    results = detector.predict_batch(["./data/test/image.png"] * 2, threshold=0.5)
    for i, r in enumerate(results):
        print(f"  图片 {i+1}: 分数={r['anomaly_score']:.4f}")


if __name__ == "__main__":
    main()
