"""
main.py

Batch License Plate Detection + OCR Pipeline with Postprocessing.

Pipeline:
---------
Image → Detection → Crop → Preprocess → OCR → Postprocess → Save Results

Input:
    ./data/input/

Output:
    ./data/output/
        - annotated images
        - results.csv
"""

import os
import cv2
import csv

from src.detection import PlateDetector
from src.ocr import PlateOCR
from src.preprocess import preprocess_plate
from src.postprocess import process_plate_text


INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")


def main():
    # -----------------------------
    # 📁 Setup
    # -----------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_files:
        print("[ERROR] No images found in data/input/")
        return

    print(f"[INFO] Found {len(image_files)} image(s)")

    # -----------------------------
    # 🚀 Initialize Modules
    # -----------------------------
    detector = PlateDetector("models/plate_detector/best.pt")
    ocr = PlateOCR()

    # -----------------------------
    # 📝 Prepare CSV
    # -----------------------------
    with open(CSV_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "raw_text", "final_text"])

        # -----------------------------
        # 🔁 Process Each Image
        # -----------------------------
        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"[WARNING] Skipping invalid image: {img_name}")
                continue

            print(f"\n[INFO] Processing: {img_name}")

            boxes = detector.detect(image)

            h, w = image.shape[:2]
            raw_texts = []
            final_texts = []

            for box in boxes:
                x1, y1, x2, y2 = box

                # 🔧 Clamp bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                plate = image[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                # 🔧 Preprocess
                processed_plate = preprocess_plate(plate)

                # 🔡 OCR
                raw_text = ocr.extract_text(processed_plate)
                final_text = process_plate_text(raw_text)

                print("RAW:", raw_text)
                print("FINAL:", final_text)

                if raw_text:
                    raw_texts.append(raw_text)
                if final_text:
                    final_texts.append(final_text)

                # -----------------------------
                # 🖍️ Draw Results
                # -----------------------------
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    image,
                    final_text if final_text else raw_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

            # Handle no detections
            raw_joined = " | ".join(raw_texts) if raw_texts else "No Plate"
            final_joined = " | ".join(final_texts) if final_texts else "No Plate"

            print(f"[RESULT] {img_name} → {final_joined}")

            # Write CSV
            writer.writerow([img_name, raw_joined, final_joined])

            # Save annotated image
            output_img_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(output_img_path, image)

    print(f"\n[INFO] Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
