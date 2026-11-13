# MedVision-AI — Model Card

## 1. Project Overview

**Project Name:** MedVision-AI  
**Project Type:** AI-Assisted Medical Imaging and Diagnosis System  
**Purpose:** Educational and research prototype for exploring deep learning techniques in medical image classification and segmentation.

MedVision-AI uses two deep learning models to demonstrate medical image analysis:

1. DenseNet121 for chest X-ray pneumonia classification.
2. 3D U-Net for spleen segmentation from CT scans.

The system also includes Grad-CAM visualization, a FastAPI backend, a React frontend, and basic scan review and audit logging functionality.

---

## 2. Model 1 — Pneumonia Classification

### Model Information

| Attribute | Details |
|---|---|
| Model | DenseNet121 |
| Framework | PyTorch |
| Task | Binary image classification |
| Input | Chest X-ray images |
| Output | Pneumonia / No Pneumonia |
| Number of classes | 2 |
| Pretrained weights | ImageNet |
| Training epochs | 5 |
| Best checkpoint | Epoch 2 |

### Dataset

**Dataset:** RSNA Pneumonia Detection Challenge

The dataset contains chest X-ray images with corresponding pneumonia annotations.

- Total labeled records: 30,227
- Unique patient groups/images used: 26,684
- Training samples: 21,347
- Validation samples: 5,337
- Training and validation split: approximately 80:20

Patient-level grouping was used to reduce the risk of patient overlap between training and validation sets.

### Model Architecture

DenseNet121 is a convolutional neural network that uses dense connections between layers.

The original classification layer was replaced with a new fully connected layer containing two output classes.

### Training Configuration

- Optimizer: Adam
- Learning rate: 0.0001
- Loss function: Weighted Cross Entropy Loss
- Input image size: 224 × 224
- Batch size: 16
- Training device: CUDA-enabled GPU

### Evaluation Results

| Metric | Result |
|---|---:|
| Accuracy | 76.47% |
| ROC-AUC | 0.8780 |
| Specificity | 74.39% |
| Pneumonia Precision | 0.49 |
| Pneumonia Recall | 0.84 |
| Pneumonia F1-score | 0.62 |

**Confusion Matrix**

| | Predicted No Pneumonia | Predicted Pneumonia |
|---|---:|---:|
| Actual No Pneumonia | 3076 | 1059 |
| Actual Pneumonia | 197 | 1005 |

### Explainability

Grad-CAM is used to generate heatmaps that highlight image regions contributing to the model's prediction.

These visualizations help demonstrate model interpretability but do not establish clinical reasoning or diagnostic accuracy.

---

## 3. Model 2 — Spleen Segmentation

### Model Information

| Attribute | Details |
|---|---|
| Model | 3D U-Net |
| Framework | PyTorch and MONAI |
| Task | Medical image segmentation |
| Input | 3D CT scans |
| Output | Spleen segmentation mask |
| Number of output classes | 2 |
| Training epochs | 5 |
| Best checkpoint | best_unet.pth |

### Dataset

**Dataset:** Medical Segmentation Decathlon — Task09_Spleen

The dataset contains abdominal CT scans and corresponding segmentation masks.

- Training image-mask pairs: 41
- Training samples: 32
- Validation samples: 9
- Separate test CT scans available: 20

### Model Architecture

The 3D U-Net architecture uses an encoder-decoder structure with skip connections.

It processes volumetric CT data and predicts a segmentation mask identifying the spleen region.

### Training Configuration

- Framework: MONAI
- Model: 3D U-Net
- Input patch size: 96 × 96 × 96
- Input channels: 1
- Output classes: 2
- Training device: CUDA-enabled GPU
- Training epochs: 5

### Evaluation Results

| Metric | Result |
|---|---:|
| Mean Validation Dice Score | 0.2602 |

The current segmentation model is an early prototype. Its validation performance is limited and requires further training, tuning, and evaluation.

---

## 4. Intended Use

MedVision-AI is intended for:

- Educational demonstrations of medical image classification.
- Learning deep learning workflows in medical imaging.
- Demonstrating 3D medical image segmentation.
- Exploring Grad-CAM explainability.
- Understanding API integration and medical imaging application development.

## 5. Limitations

- The models have not undergone clinical validation.
- Classification performance is not sufficient for clinical diagnosis.
- The segmentation model has limited validation performance.
- The training process uses a limited number of epochs.
- The system does not provide authenticated doctor access.
- The application does not currently provide full DICOM/Orthanc integration.
- Model confidence scores should not be interpreted as calibrated clinical probabilities.
- Grad-CAM heatmaps do not guarantee that predictions are medically correct.

## 6. Ethical and Safety Considerations

This project is an educational prototype and is not intended to replace qualified medical professionals.

Predictions and segmentation outputs must not be used as the sole basis for medical decisions, diagnosis, or treatment.

Patient privacy and responsible handling of medical imaging data should be maintained.

## 7. Future Improvements

- Improve classification performance through additional training and hyperparameter tuning.
- Improve segmentation performance with more training and validation.
- Add DICOM image support.
- Integrate Orthanc for medical image management.
- Implement secure authentication and role-based access.
- Expand automated testing and model evaluation.
- Explore model calibration and external validation.

## 8. Technology Stack

- Python
- PyTorch
- MONAI
- DenseNet121
- 3D U-Net
- Grad-CAM
- FastAPI
- React
- SQLite

---

**Project:** MedVision-AI  
**Author:** Mouli Janghel  
**Project Category:** AI / Machine Learning / Medical Imaging  
**Status:** Educational Prototype