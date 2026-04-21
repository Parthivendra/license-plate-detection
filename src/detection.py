from ultralytics import YOLO
import torch

class PlateDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image):
        results = self.model(image, device=self.device)

        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append([int(x1), int(y1), int(x2), int(y2)])

        return boxes

    def draw_boxes(self, image, boxes):
        """
        Draw bounding boxes on image
        """
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image
