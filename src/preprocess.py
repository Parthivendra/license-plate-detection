import cv2

def preprocess_plate(plate_img):
    """
    Improve plate image for better OCR
    """

    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding (important!)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    return thresh
