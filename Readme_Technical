# 🔧 Technical Implementation Guide

## Breast Cancer Histopathology Classification

This document provides in-depth technical details about the implementation, algorithms, and architecture decisions.

---

## 📐 System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ BreakHis DB │  │ Preprocessing│  │ Data Augmentation  │    │
│  └─────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Segmentation    │  │ Feature Ext. │  │ Feature Fusion │    │
│  │ Engine          │  │ (Dual Path)  │  │ (519-D)        │    │
│  └─────────────────┘  └──────────────┘  └────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     LEARNING LAYER                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ AutoML Engine   │  │ Model Eval.  │  │ Calibration    │    │
│  │ (30+ models)    │  │ (CV)         │  │ (Platt Scale)  │    │
│  └─────────────────┘  └──────────────┘  └────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     OUTPUT LAYER                                │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ Predictions     │  │ Explanations │  │ Visualizations │    │
│  └─────────────────┘  └──────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Algorithm Details

### 1. Nuclei Segmentation Pipeline

#### 1.1 Color Space Conversion

**Why HSV?**
- H&E staining produces purple (hematoxylin) nuclei
- HSV separates color (Hue) from intensity (Value)
- More robust to lighting variations than RGB

```python
def hsv_threshold(image):
    """
    Segment nuclei using HSV color thresholding

    Args:
        image: BGR image (OpenCV format)

    Returns:
        Binary mask of nuclei
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of purple colors
    # H: 120-170 (purple-blue range)
    # S: 40-255 (moderate to high saturation)
    # V: 20-200 (avoid very dark and very bright)
    lower_purple = np.array([120, 40, 20])
    upper_purple = np.array([170, 255, 200])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    return mask
```

#### 1.2 Morphological Operations

**Purpose:** Remove noise and smooth boundaries

```python
def morphological_cleanup(mask):
    """
    Clean binary mask using morphological operations

    Operations:
    1. Opening: Remove small noise
    2. Closing: Fill small holes
    """
    kernel = np.ones((3,3), np.uint8)

    # Opening: erosion followed by dilation
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask_opened
```

#### 1.3 Watershed Segmentation

**Problem:** Touching/overlapping nuclei appear as single objects

**Solution:** Watershed algorithm treats image as topographic surface

```python
def watershed_segmentation(mask):
    """
    Separate touching nuclei using watershed algorithm

    Algorithm:
    1. Compute distance transform
    2. Find sure foreground (peaks)
    3. Find sure background (valleys)
    4. Create markers
    5. Apply watershed
    """
    # Distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold to get sure foreground
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Dilate mask to get sure background
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)

    # Unknown region = sure_bg - sure_fg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Create markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Background = 1
    markers[unknown == 255] = 0

    # Apply watershed
    markers = watershed(-dist_transform, markers, mask=mask)

    return markers
```

---

### 2. Feature Extraction

#### 2.1 Classical Features (7 dimensions)

**A. Morphological Features (5)**

```python
def extract_morphological_features(markers, intensity_image):
    """
    Extract morphological features from segmented nuclei
    """
    properties = measure.regionprops(markers, intensity_image=intensity_image)

    features = {
        'nuclei_count': len(properties),
        'mean_area': np.mean([p.area for p in properties]) if properties else 0,
        'mean_eccentricity': np.mean([p.eccentricity for p in properties]) if properties else 0,
        'mean_solidity': np.mean([p.solidity for p in properties]) if properties else 0,
        'mean_intensity': np.mean([p.mean_intensity for p in properties]) if properties else 0,
    }

    return features
```

**Feature Descriptions:**

| Feature | Clinical Significance |
|---------|----------------------|
| **Nuclei Count** | Cell density indicator |
| **Mean Area** | Nuclear enlargement (malignancy marker) |
| **Eccentricity** | Shape irregularity |
| **Solidity** | Boundary irregularity |
| **Mean Intensity** | Staining intensity |

**B. Texture Features (2)**

```python
def extract_texture_features(gray_image):
    """
    Extract GLCM-based texture features
    """
    try:
        glcm = feature.graycomatrix(
            gray_image,
            distances=[5],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        energy = feature.graycoprops(glcm, 'energy')[0, 0]

        return {
            'texture_contrast': contrast,
            'texture_energy': energy
        }
    except:
        return {
            'texture_contrast': 0,
            'texture_energy': 0
        }
```

#### 2.2 Deep Features (512 dimensions)

**ResNet18 Feature Extractor**

```python
class DeepFeatureExtractor:
    """
    Extract deep semantic features using pre-trained ResNet18
    """
    def __init__(self):
        self.model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image):
        """
        Extract 512-dimensional feature vector

        Args:
            image: BGR image (OpenCV format)

        Returns:
            numpy array of shape (512,)
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        return features.numpy().flatten()
```

**Why ResNet18?**
- **Lightweight**: 18 layers, ~11M parameters
- **Pre-trained**: Learned rich features from ImageNet (1.2M images)
- **Transfer learning**: Features generalize well to medical images
- **Efficient**: Fast inference, suitable for real-time applications

#### 2.3 Feature Fusion

```python
def create_hybrid_feature_vector(classical_features, deep_features):
    """
    Concatenate classical and deep features

    Classical: 7 dimensions
    Deep: 512 dimensions
    Total: 519 dimensions
    """
    classical_array = np.array(list(classical_features.values()))
    hybrid_vector = np.concatenate([classical_array, deep_features])

    return hybrid_vector  # Shape: (519,)
```

---

### 3. AutoML Pipeline

#### 3.1 LazyPredict Implementation

```python
class AutoMLClassifier:
    """
    Automated model selection and evaluation using LazyPredict
    """
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train and evaluate 30+ algorithms
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True,
            custom_metric=None
        )

        self.models, self.predictions = clf.fit(
            X_train, X_test, y_train, y_test
        )

        self.best_model_name = self.models.index[0]
        return self.models

    def get_top_models(self, n=5):
        return self.models.head(n)
```

#### 3.2 Model Evaluation Metrics

```python
def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

    return metrics
```

---

## 🔬 Experimental Setup

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 or equivalent
- RAM: 8 GB
- Storage: 10 GB free space

**Recommended:**
- CPU: Intel i7 or better
- RAM: 16 GB
- GPU: NVIDIA GTX 1060+ (for faster deep feature extraction)
- Storage: 20 GB SSD

### Software Environment

```yaml
name: breast-cancer-env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pytorch=2.0.1
  - torchvision=0.15.2
  - cudatoolkit=11.7
  - opencv=4.8.0
  - scikit-image=0.21.0
  - pandas=2.0.3
  - numpy=1.24.3
  - scikit-learn=1.3.0
  - matplotlib=3.7.2
  - seaborn=0.12.2
  - jupyter=1.0.0
  - pip
  - pip:
      - lazypredict==0.2.12
      - kagglehub==0.1.0
```

### Training Configuration

**Data Split:**
- Training: 80% (640 images)
- Testing: 20% (160 images)
- Stratified sampling (maintain class balance)

**Hyperparameters:**
```python
# CNN Baseline
{
    'batch_size': 16,
    'epochs': 5,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss': 'CrossEntropyLoss',
    'image_size': (224, 224)
}

# AutoML
{
    'cv_folds': 5,
    'scoring': 'accuracy',
    'n_jobs': -1
}
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Memory Error during Feature Extraction**
```python
# Solution: Process images in batches
def batch_feature_extractor(image_paths, batch_size=32):
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        yield process_batch(batch)
```

**2. CUDA Out of Memory**
```python
import torch
torch.cuda.empty_cache()

# Or force CPU
device = torch.device('cpu')
```

**3. LazyPredict Installation Issues**
```bash
pip install scikit-learn pandas numpy
pip install lazypredict
```

---

## 📊 Performance Optimization

### Speed Improvements

**1. Parallel Processing:**
```python
from multiprocessing import Pool

def parallel_feature_extraction(image_paths, n_workers=4):
    with Pool(n_workers) as pool:
        features = pool.map(extract_features, image_paths)
    return features
```

**2. Feature Caching:**
```python
import joblib

# Save features
joblib.dump(features, 'features_cache.pkl')

# Load features
features = joblib.load('features_cache.pkl')
```

**3. Model Serialization:**
```python
joblib.dump(best_model, 'model.pkl')
model = joblib.load('model.pkl')
```

---

## 📚 References

1. Spanhol, F. A., et al. "A dataset for breast cancer histopathological image classification." IEEE TBME 63.7 (2015)
2. He, K., et al. "Deep residual learning for image recognition." CVPR 2016
3. Pedregosa, F., et al. "Scikit-learn: Machine learning in Python." JMLR 12 (2011)

---

<div align="center">

[⬆ Back to Top](#-technical-implementation-guide)

</div>
