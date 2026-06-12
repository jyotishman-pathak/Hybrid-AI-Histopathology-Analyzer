# 🩺 Hybrid AI Histopathology Analyzer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Bridging the gap between Explainable Classical Computer Vision and Deep Learning via AutoML.**  
> A research-grade, hybrid machine learning pipeline for breast cancer histopathology classification that eliminates the "black box" nature of traditional deep learning while achieving state-of8-the-art accuracy.

---

## ⚠️ The Problem
In medical AI, there is a fundamental trade-off:
1. **Deep Learning (CNNs)** achieves high accuracy but acts as a "black box." Doctors cannot trust a model if it cannot explain *why* it made a diagnosis.
2. **Classical Machine Learning** is highly explainable (using hand-crafted features like area, roundness, and texture), but it mathematically caps out at ~60-65% accuracy because it fails to capture the chaotic, high-dimensional texture of malignant cells.

---

## 💡 The Solution: Hybrid Feature Fusion
This project introduces a **Hybrid Explainable Pipeline** that gets the best of both worlds. Instead of feeding raw pixels into a black-box CNN, we:
1. Use **Classical Computer Vision** (Watershed Segmentation) to transparently isolate cell nuclei.
2. Extract **7 Clinically Relevant Features** (morphological and textural).
3. Pass the same image through a pre-trained **ResNet18** to extract **512 Deep Semantic Features**.
4. Fuse them into a **519-Dimensional Feature Space** and benchmark 30+ algorithms using **LazyPredict (AutoML)**.

This approach provides the transparency of classical ML with the predictive power of Deep Learning.

---

## 📈 Key Results & Clinical Insights

Beyond just building a classifier, this project conducted a **Magnification-Dependent Analysis** to understand how the model behaves under different microscope settings:

| Magnification | Accuracy | Clinical Insight |
| :--- | :---: | :--- |
| **40X** | **~89%** | 🏆 **Peak Performance.** Captures global tissue architecture and cell-stroma interaction, primary indicators of malignancy. |
| **100X** | **~87%** | Strong performance, balancing local nuclear detail with some tissue context. |
| **400X** | **~83%** | Accuracy drops. At extreme zoom, the model loses global architectural context, mimicking a known challenge in digital pathology. |

---

## 🛠️ Tech Stack
- **Image Processing:** `OpenCV`, `scikit-image` (Watershed, GLCM, RegionProps)
- **Deep Learning:** `PyTorch`, `torchvision` (ResNet18 Feature Extraction)
- **Machine Learning:** `scikit-learn`, `LazyPredict` (AutoML Benchmarking)
- **Deployment / UI:** `Gradio` (1-Click Web Interface)
- **Data Handling:** `Pandas`, `NumPy`, `KaggleHub`

---

## 🚀 Quick Start (Run in 5 Minutes)
The entire pipeline is designed to run end-to-end in a single Google Colab cell.

1. Open a new [Google Colab Notebook](https://colab.research.google.com/).
2. Copy the **Master Single-File Script** from the repository.
3. Paste it into the first cell and hit **Run**.
4. The script will automatically download the data, extract features, train the AutoML models, generate the research graph, and launch a live Gradio Web UI with a public shareable link.

---

## 📚 Dataset & Acknowledgements
This project utilizes the **BreakHis (Breast Cancer Histopathological Database)** dataset, containing H&E stained breast tumor images at 40X, 100X, 200X, and 400X magnifications.  
🔗 *Dataset Source: [Kaggle - BreakHis Breast Cancer Histopathological Dataset](https://www.kaggle.com/datasets/waseemalastal/breakhis-breast-cancer-histopathological-dataset)*

---

## 📜 License
This project is licensed under the MIT License.
