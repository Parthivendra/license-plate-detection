"""
test_model.py

Unified testing interface for:
- Vanilla YOLOv8 (COCO pretrained)
- Custom YOLOv8 (license plate detection)

Usage:
------
Vanilla:
    python test_model.py -v

Custom:
    python test_model.py -c
"""

import cv2
import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO


class PlateTester:
    def __init__(self, model_type="custom", model_path=None):
        if model_type == "vanilla":
            print("[INFO] Using pretrained YOLOv8 (general object detection)")
            self.model = YOLO("yolov8n.pt")

        elif model_type == "custom":
            print(f"[INFO] Using trained model: {model_path}")
            self.model = YOLO(model_path)

        else:
            raise ValueError("Invalid model type")

    def run(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        results = self.model(image)
        annotated = results[0].plot()

        return annotated, results


def show_image(img, title="Output"):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test YOLO models")

    # 🔥 Short flags
    parser.add_argument("-v", "--vanilla", action="store_true",
                        help="Use pretrained YOLOv8 model")

    parser.add_argument("-c", "--custom", action="store_true",
                        help="Use custom trained model")

    parser.add_argument("--image", type=str, default="test.png",
                        help="Path to test image")

    parser.add_argument("--model_path", type=str,
                        default="models/plate_detector/best.pt",
                        help="Path to trained model")

    args = parser.parse_args()

    # 🔥 Logic to decide model
    if args.vanilla:
        model_type = "vanilla"
    elif args.custom:
        model_type = "custom"
    else:
        print("[WARNING] No model specified, defaulting to custom")
        model_type = "custom"

    tester = PlateTester(
        model_type=model_type,
        model_path=args.model_path
    )

    output, results = tester.run(args.image)

    print("\n[INFO] Detection Results:")
    print(results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else "No detections")

    show_image(output, title=f"{model_type.upper()} MODEL OUTPUT")

