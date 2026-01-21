"""
DINOv3 image level anomaly detection example

show how to use DINOv3 image level detector.
use config file to manage all parameters.
"""

from pathlib import Path
from hq_anomaly_detection import AnomalyDetector


def main():
    """main function example"""
    
    # 1. create detector from config file
    print("init DINOv3 image level detector...")
    detector = AnomalyDetector(config_path="configs/dinov3_image_level.yaml")
    print("DINOv3 image level detector initialized.")
    # 2. train model (batch loading data, avoid memory overflow)
    print("\ntrain model...")
    print("use batch loading data strategy, avoid memory overflow...")
    
    detector.train()
    print("model trained.")
    
    # 3. save model
    detector.save_model("./checkpoints/dinov3_image_level_checkpoint.pth")
    print("model saved.")
    
    # 4. load trained model
    detector.load_model("./checkpoints/dinov3_image_level_checkpoint.pth")
    print("trained model loaded.")
    
    # 5. detect image (single image)
    print("\ndetect image...")
    
    # detect single image
    result = detector.detect(
        image_path="./data/test/image.png",
        threshold=0.5,  # 异常阈值（可选）
    )
    
    print(f"\ndetect result:")
    print(f"  anomaly score: {result['anomaly_score']:.4f}")
    print(f"  prediction: {'anomaly' if result['prediction'] else 'normal'}")
    
    # 6. batch predict
    print("\nbatch predict...")
    image_paths = ["./data/test/image.png", "./data/test/image.png"]
    results = detector.predict_batch(image_paths, threshold=0.5)
    for i, result in enumerate(results):
        print(f"  image {i+1}: anomaly score={result['anomaly_score']:.4f}, "
              f"prediction={'anomaly' if result['prediction'] else 'normal'}")
    
    print("\nexample completed!")
    print("\nnotes:")
    print("1. use config file to manage all parameters")
    print("2. use batch loading data strategy, control memory usage through buffer_size")
    print("3. when buffer is full, automatically sample new embeddings")
    print("4. image level detection only returns anomaly score, no spatial anomaly map")


if __name__ == "__main__":
    main()
