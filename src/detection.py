from ultralytics import YOLO
import cv2

class PlateDetector:
    def __init__(self, model_path):
        """
        Initialize the YOLO model
        """
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect license plates in an image

        Args:
            image: input image (numpy array)

        Returns:
            boxes: list of bounding boxes [x1, y1, x2, y2]
        """
        results = self.model(image)

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
