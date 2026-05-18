# Automatic Number Plate Recognition (ANPR)

A modular **Automatic Number Plate Recognition (ANPR)** system built using **YOLOv8**, **EasyOCR**, and **OpenCV** for detecting and recognising Indian vehicle registration plates from images.

This project follows a production-style computer vision pipeline:

> **Image → Detection → Cropping → Preprocessing → OCR → Postprocessing**

---

## Features

* YOLOv8-based license plate detection
* OCR using EasyOCR
* Image preprocessing optimized for OCR readability
* Rule-based postprocessing for Indian plate formats
* Batch image processing support
* GPU acceleration support (CUDA)
* CSV export of results
* Optional annotated image saving
* Modular and extensible architecture

---

## Tech Stack

* **Python**
* **YOLOv8 (Ultralytics)**
* **EasyOCR**
* **OpenCV**
* **NumPy**
* **Matplotlib**

---

## Project Architecture

```text
license-plate-detection/
│
├── data/
│   ├── input/              # Input vehicle images
│   └── output/             # Results CSV + annotated images
│
├── models/
│   └── best.pt             # YOLOv8 trained model
│
├── src/
│   ├── detection.py        # Plate detection
│   ├── preprocess.py       # Image preprocessing
│   ├── ocr.py              # OCR pipeline
│   └── postprocess.py      # Validation & correction
│
├── main.py                 # Main pipeline
├── requirements.txt
└── README.md
```

---

## ANPR Pipeline

### 1. Input Handling

* Supports `.jpg`, `.jpeg`, and `.png`
* Batch processing via CLI

### 2. License Plate Detection

* YOLOv8 detects plate regions
* Returns bounding boxes in pixel coordinates

### 3. Cropping with Padding

* Dynamic contextual padding added
* Prevents truncation of edge characters

### 4. Image Preprocessing

The preprocessing pipeline includes:

* Adaptive upscaling
* Grayscale conversion
* CLAHE contrast enhancement
* Unsharp masking (sharpening)

Aggressive thresholding was intentionally avoided because it degraded OCR quality on unevenly illuminated plates.

### 5. OCR

* EasyOCR extracts alphanumeric text
* Restricted allowlist:

  * `A-Z`
  * `0-9`

### 6. Postprocessing & Validation

The extracted text is:

* Cleaned
* Corrected using positional heuristics
* Matched against Indian number plate formats
* Validated against official state/UT codes

Invalid outputs are labelled:

```text
NOT LEGIBLE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Parthivendra/license-plate-detection.git
cd license-plate-detection
```

Create virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Process Random Sample (Default)

```bash
python main.py
```

### Process Limited Images

```bash
python main.py --limit 20
```

### Process Entire Dataset

```bash
python main.py --no-limit
```

### Save Annotated Images

```bash
python main.py --save-images
```

### Debug Preprocessed Plates

```bash
python main.py --debug
```

---

## Example Output

```text
[INFO] Processing: UP10.jpg

RAW: UP7DB3730
FINAL: UP7DB3730

[RESULT] UP10.jpg → UP7DB3730
```

Results are stored in:

```text
data/output/results.csv
```

CSV format:

| filename | raw_text | final_text | filepath |
| -------- | -------- | ---------- | -------- |

---

## Performance Observations

| Condition              | Outcome                                 |
| ---------------------- | --------------------------------------- |
| Clear, well-lit plates | High accuracy                           |
| Slight blur            | Partial recovery through postprocessing |
| Small crops            | Upscaling improves readability          |
| Heavy glare/shadows    | Often marked NOT LEGIBLE                |
| Non-standard plates    | Rejected by validator                   |

---

## Key Engineering Decisions

* CLAHE over global thresholding
* Adaptive upscaling for tiny crops
* OCR allowlist restriction
* Positional OCR correction (`0 ↔ O`, `1 ↔ I`, etc.)
* Modular architecture for maintainability

---

## Future Improvements

* Real-time video ANPR
* Fine-tuned OCR models (CRNN/TrOCR)
* Custom-trained YOLOv8 detector
* Multi-language plate support
* Edge deployment (Jetson/Raspberry Pi)
* Confidence-based OCR filtering
* Database integration

---

## Sample Pipeline

```text
Input Image
    ↓
YOLOv8 Detection
    ↓
Plate Cropping
    ↓
Image Preprocessing
    ↓
EasyOCR
    ↓
Postprocessing & Validation
    ↓
CSV Output
```

---

## References

* Ultralytics YOLOv8
* EasyOCR
* OpenCV
* Indian Vehicle Dataset (Kaggle)

---

## Author

**Parthivendra Singh**

GitHub: https://github.com/Parthivendra

---

## Repository

https://github.com/Parthivendra/license-plate-detection
