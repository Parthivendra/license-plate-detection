import easyocr
import cv2
import torch

class PlateOCR:
    def __init__(self):

        use_gput = torch.cuda.isavailable()
        print(f"OCR GPU: {use_gpu}")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image):
        """
        Extract text from plate image

        Args:
            image: cropped plate image

        Returns:
            text: detected string
        """
        results = self.reader.readtext(image)

        extracted_text = ""

        for (bbox, text, confidence) in results:
            extracted_text += text + " "

        return extracted_text.strip()
