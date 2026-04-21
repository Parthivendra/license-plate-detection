import cv2
import numpy as np

# Minimum height (px) the plate crop should be before OCR sees it.
# EasyOCR works best when character height is ~32-64 px.
MIN_HEIGHT = 64


def preprocess_plate(plate_img):
    """
    OCR-optimised preprocessing with adaptive upscaling.
    Scales the crop so its height is at least MIN_HEIGHT,
    then applies gentle contrast + sharpening.
    """

    h, w = plate_img.shape[:2]

    # 1. Adaptive upscale — guarantee a minimum height for OCR
    scale = max(MIN_HEIGHT / h, 1.0)
    if scale > 1.0:
        plate_img = cv2.resize(
            plate_img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_LANCZOS4,
        )

    # 2. Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE – gentle local contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Light sharpen via unsharp mask
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return gray
