import easyocr
import cv2
import torch

class PlateOCR:
    def __init__(self):

        use_gpu = torch.cuda.is_available()
        print(f"OCR GPU: {use_gpu}")
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def extract_text(self, image):
        """
        Extract text from plate image.

        Uses dual-pass: tries the preprocessed image first,
        and if confidence is low, retries with raw color crop.

        Args:
            image: preprocessed plate image (grayscale)

        Returns:
            text: detected string
        """
        text, conf = self._read(image)
        return text

    def _read(self, image):
        """Run OCR and return (text, avg_confidence)."""
        results = self.reader.readtext(
            image,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        )

        if not results:
            return "", 0.0

        texts = []
        confs = []
        for (bbox, text, confidence) in results:
            texts.append(text)
            confs.append(confidence)

        return " ".join(texts), sum(confs) / len(confs)
