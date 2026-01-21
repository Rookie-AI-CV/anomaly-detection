"""
使用示例

演示如何使用 hq_anomaly_detection 进行异常检测。
使用配置文件方式。
"""

from hq_anomaly_detection import AnomalyDetector


def main():
    """主函数示例"""
    
    # 1. 从配置文件创建检测器
    print("初始化异常检测器...")
    detector = AnomalyDetector(config_path="configs/dinov3_image_level.yaml")
    
    # 2. 训练模型（从配置文件读取参数）
    # print("训练模型...")
    # detector.train()
    
    # 3. 保存模型
    # detector.save_model("./checkpoints/model_checkpoint.pth")
    
    # 4. 加载已训练的模型
    # detector.load_model("./checkpoints/model_checkpoint.pth")
    
    # 5. 进行异常检测
    print("进行异常检测...")
    # image_path = "./data/test/image.png"
    # result = detector.detect(image_path, threshold=0.5)
    
    # 6. 可视化结果（如果有异常图）
    # if result["anomaly_map"] is not None:
    #     from hq_anomaly_detection.utils.visualization import visualize_detection_result
    #     from hq_anomaly_detection.utils.data_loader import load_image
    #     image = load_image(image_path)
    #     visualize_detection_result(
    #         image=image,
    #         anomaly_map=result["anomaly_map"],
    #         anomaly_score=result["anomaly_score"],
    #         threshold=0.5,
    #         save_path="./results/visualization.png",
    #         show=True
    #     )
    
    print("示例完成！")


if __name__ == "__main__":
    main()
