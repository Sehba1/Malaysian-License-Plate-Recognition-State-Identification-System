# Malaysian License Plate Recognition & State Identification System

A computer vision application that detects Malaysian vehicle license plates, recognizes plate characters using Optical Character Recognition (OCR), and identifies the corresponding Malaysian state based on license plate prefixes.

This project was developed as part of a group assignment, with my primary contribution focused on the truck and pickup vehicle detection modules.

---

## 📌 Project Overview

The system processes vehicle images using classical computer vision techniques to localize license plates before extracting plate characters with PaddleOCR. The recognized license plate is then used to determine the corresponding Malaysian state based on official registration prefixes.

The project supports multiple Malaysian vehicle categories and includes an interactive graphical user interface for image processing and visualization.

---

## ✨ Features

- Automatic Malaysian license plate detection
- Optical Character Recognition (OCR) using PaddleOCR
- Malaysian state identification from license plate prefixes
- Image preprocessing pipeline for improved detection accuracy
- Vehicle-specific detection modules
- Interactive desktop graphical user interface
- Step-by-step visualization of the image processing pipeline

---

## 👨‍💻 My Contribution

As part of this group project, I was responsible for:

- Developing the truck and pickup vehicle detection modules
- Designing vehicle-specific image preprocessing pipelines
- Implementing adaptive thresholding and contour-based plate localization
- Integrating PaddleOCR for license plate character recognition
- Contributing to Malaysian state identification based on recognized plate prefixes

---

## 🛠 Technologies Used

- Python
- OpenCV
- PaddleOCR
- NumPy
- PyWavelets
- Pillow (PIL)
- Tkinter
- Regular Expressions (Regex)

---

## 🧠 Computer Vision Pipeline

1. Load vehicle image
2. Image preprocessing
3. Contrast enhancement
4. Adaptive thresholding
5. Canny edge detection
6. Morphological operations
7. Contour analysis
8. License plate localization
9. Character recognition using PaddleOCR
10. Malaysian state identification

---

## 🚀 Getting Started

### Requirements

- Python 3.13+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app_gui.py
```

---

## 📂 Project Structure

```
Malaysian-License-Plate-Recognition-State-Identification-System/
│
├── core/
├── processors/
├── test_images/
├── processing_results/
├── app_gui.py
├── main_processor.py
├── requirements.txt
└── README.md
```

---

## 📸 Sample Results

The project generates intermediate processing results including:

- Original image
- Grayscale conversion
- Contrast enhancement
- Wavelet transformation
- Edge detection
- Morphological processing
- License plate localization
- OCR recognition
- Malaysian state identification



---

## 📄 License

This project is intended for educational and portfolio purposes.

## Team Contribution

This project was completed as a group assignment. My primary responsibilities included:

- Truck vehicle detection module
- Pickup vehicle detection module
- License plate localization for assigned vehicle categories
- OCR integration using PaddleOCR
- Testing and validation of assigned modules
