# Breast Cancer Histopathology Classification using Hybrid Feature Fusion

## Overview

A hybrid approach for classifying breast cancer histopathology images as **Benign** or **Malignant**. The pipeline combines classical image processing features with deep learning features (ResNet18) and uses AutoML (LazyPredict) to automatically find the best classifier — achieving **90.62% accuracy**.

---

## How It Works

```
Input Image
    │
    ▼
HSV Thresholding + Morphological Cleaning + Watershed Segmentation
    │
    ▼
Feature Extraction
    ├── Classical Features (7)  →  Nuclei count, area, eccentricity, solidity, intensity, texture contrast, texture energy
    └── Deep Features (512)     →  ResNet18 (pretrained, last layer removed)
    │
    ▼
Concatenate → 519-dimensional feature vector
    │
    ▼
LazyPredict AutoML (30+ classifiers tested automatically)
    │
    ▼
Best Model: CalibratedClassifierCV → 90.62% Accuracy
```

---

## Dataset

**BreakHis** — Breast Cancer Histopathological Database
- 7,909 images from 82 patients
- 4 magnification levels: 40X, 100X, 200X, 400X
- 2 classes: Benign / Malignant
- We sampled 100 images per class per magnification → **800 images total**
- 80-20 train-test split (stratified)

---

## Results

### Overall

| Metric | Value |
|--------|-------|
| Best Model | CalibratedClassifierCV |
| Accuracy | **90.62%** |
| ROC AUC | 93.83% |
| F1 Score | 90.62% |

### Accuracy by Magnification

| Magnification | Accuracy |
|---------------|----------|
| 40X | 89.19% |
| 100X | 87.18% |
| 200X | 79.07% |
| 400X | 82.93% |

### Hybrid vs Standard CNN (ResNet18 fine-tuned)

| Approach | Accuracy |
|----------|----------|
| Standard CNN (ResNet18) | 75.00% |
| **Hybrid AutoML (Ours)** | **90.62%** |

---

## Top Models from AutoML

| Model | Accuracy |
|-------|----------|
| CalibratedClassifierCV | 90.62% |
| LogisticRegression | 90.62% |
| SVC | 88.75% |
| ExtraTreesClassifier | 87.50% |
| RandomForestClassifier | 86.25% |

---

## Tech Stack

- **OpenCV** — image preprocessing and segmentation
- **scikit-image** — region properties and GLCM texture features
- **PyTorch + torchvision** — ResNet18 deep feature extraction
- **LazyPredict** — automated model selection
- **scikit-learn** — train/test split, evaluation metrics
- **pandas / numpy / matplotlib / seaborn** — data handling and visualization

---

## Installation

```bash
git clone https://github.com/yourusername/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-python
scikit-image
pandas
numpy
torch
torchvision
scikit-learn
lazypredict
matplotlib
seaborn
kagglehub
```

---

## Project Structure

```
├── main.ipynb               # Full pipeline notebook
├── requirements.txt
├── README.md
├── models/
│   └── best_model.pkl
└── results/
    ├── confusion_matrix.png
    └── magnification_accuracy.png
```

---

## Key Takeaways

- Hybrid features (classical + deep) outperform either alone
- Lower magnifications (40X, 100X) give better classification accuracy
- AutoML removes the need for manual model tuning
- Classical features (nuclei count, area) provide interpretability alongside deep features
