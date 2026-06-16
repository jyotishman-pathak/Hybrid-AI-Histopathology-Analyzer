# 🏥 Breast Cancer Histopathology Classification using Hybrid Feature Fusion

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

This project presents a **novel hybrid approach** for automated breast cancer classification from histopathological images. By combining classical computer vision techniques with deep learning feature extraction and AutoML, we achieve **90.62% accuracy** while maintaining clinical interpretability—a critical requirement for medical diagnosis systems.

### Why This Matters

Traditional deep learning models for medical imaging often act as "black boxes," providing predictions without explanations. Our hybrid approach addresses this critical gap by:
- ✅ Providing **interpretable features** (nuclei count, area, texture)
- ✅ Leveraging **deep semantic features** for complex patterns
- ✅ Achieving **superior accuracy** compared to standard CNNs
- ✅ Requiring **zero manual annotation** (unsupervised segmentation)

---

## 🔬 Problem Statement

Breast cancer diagnosis through histopathology is time-consuming and subject to inter-observer variability. Pathologists must:
1. Examine H&E stained tissue samples under microscopes
2. Identify malignant vs benign patterns based on nuclear morphology
3. Make critical treatment decisions

**Challenges:**
- Subjectivity in manual diagnosis
- Time-intensive process
- Need for explainable AI in clinical settings
- Limited labeled data availability

---

## 🏗️ Solution Architecture
