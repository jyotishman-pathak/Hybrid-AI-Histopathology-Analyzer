# 🩺 Hybrid AI Histopathology Analyzer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Bridging the gap between Explainable Classical Computer Vision and Deep Learning via AutoML.**  
> A research-grade, hybrid machine learning pipeline for breast cancer histopathology classification that eliminates the "black box" nature of traditional deep learning while achieving state-of-the-art accuracy.

---

## 📌 Table of Contents
- [The Problem](#-the-problem)
- [The Solution: Hybrid Feature Fusion](#-the-solution-hybrid-feature-fusion)
- [System Architecture](#-system-architecture)
- [Key Results & Clinical Insights](#-key-results--clinical-insights)
- [Tech Stack](#-tech-stack)
- [Quick Start (Run in 2 Minutes)](#-quick-start)
- [Project Structure](#-project-structure)
- [Dataset & Acknowledgements](#-dataset--acknowledgements)

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

## 🏗️ System Architecture

```mermaid
graph TD
    A[📷 H&E Histopathology Image] --> B(Stage 1: Classical Segmentation)
    B -->|HSV Thresholding + Watershed| C[Isolated Nuclei Masks]
    
    C --> D{Stage 2: Hybrid Feature Extraction}
    D -->|Morphology & GLCM Texture| E[7 Classical Features]
    A -->|Pre-trained ResNet18| F[512 Deep Semantic Features]
    
    E --> G[🧬 519-Dimensional Feature Vector]
    F --> G
    
    G --> H(Stage 3: AutoML Benchmarking)
    H -->|LazyPredict| I[🏆 Best Model Selection e.g., Random Forest]
    I --> J[📊 Explainable Prediction + Confidence Score]
