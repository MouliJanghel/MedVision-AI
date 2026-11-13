# MedVision-AI
## AI-Assisted Medical Imaging and Diagnosis System

MedVision-AI is an educational AI-assisted medical imaging prototype that demonstrates how deep learning can support medical image classification, 3D medical image segmentation, explainable AI, and scan review workflows.

The system combines a Python-based FastAPI backend with a React dashboard to display AI predictions, Grad-CAM visualizations, scan history, doctor review decisions, and audit logs.

> **Disclaimer:** This project is intended for educational and research demonstration purposes only. It is not a clinically validated medical device and must not be used to diagnose or make treatment decisions.

## Project Features

- **Pneumonia Classification:** Uses a pretrained DenseNet121 model fine-tuned on the RSNA Pneumonia Detection dataset.
- **3D Medical Image Segmentation:** Uses a 3D U-Net model built with MONAI for spleen segmentation from CT scans.
- **Explainable AI:** Generates Grad-CAM heatmaps to visualize image regions influencing classification predictions.
- **Scan Review Workflow:** Supports approving or rejecting AI-generated scan predictions.
- **Audit Logging:** Records scan creation and review activities.
- **Interactive Dashboard:** React-based interface for uploading scans, viewing predictions, reviewing scans, and monitoring activity.
- **REST API:** FastAPI endpoints for model inference and dashboard operations.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python, JavaScript |
| Deep Learning | PyTorch |
| Medical AI | MONAI |
| Classification | DenseNet121 |
| Segmentation | 3D U-Net |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Frontend | React, Vite |
| HTTP Requests | Axios |
| Icons | Lucide React |
| Database | SQLite |
| Development Tools | VS Code, Git, GitHub |

## Model Development and Performance

### 1. Pneumonia Classification

- Dataset: RSNA Pneumonia Detection Challenge.
- Model: DenseNet121 with pretrained weights.
- Task: Binary classification of chest X-ray images.
- Training: 5 epochs.
- Best validation checkpoint: Epoch 2.

**Evaluation results**

| Metric | Result |
|---|---:|
| Accuracy | 76.47% |
| ROC-AUC | 0.8780 |
| Specificity | 74.39% |
| Pneumonia Precision | 49% |
| Pneumonia Recall | 84% |
| Pneumonia F1-score | 62% |

### 2. Spleen Segmentation

- Dataset: Medical Segmentation Decathlon – Task09 Spleen.
- Model: 3D U-Net using MONAI.
- Training samples: 32.
- Validation samples: 9.
- Training: 5 epochs.

**Validation result**

| Metric | Result |
|---|---:|
| Mean Dice Score | 0.2602 |

The segmentation model is an early prototype and requires further training and optimization.

### 3. Explainable AI

Grad-CAM is used to generate heatmaps showing image regions that influenced the DenseNet121 classification output.

These visualizations explain model attention and do not prove the presence or location of a disease.

## System Architecture

1. The user uploads a medical image through the React dashboard.
2. The frontend sends the image to the FastAPI backend.
3. The backend processes the image using the trained deep learning model.
4. The classification model generates a prediction and confidence score.
5. Grad-CAM generates a corresponding heatmap.
6. Scan information and review activities are stored in the database.
7. The dashboard displays results and allows review decisions.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | Backend health check |
| POST | `/predict` | Chest X-ray classification |
| GET | `/scans` | Retrieve scan records |
| POST | `/scans/{scan_id}/review` | Approve or reject a scan |
| GET | `/audit-logs` | Retrieve audit history |
| POST | `/segment` | Perform 3D spleen segmentation |
| GET | `/download-mask/{mask_id}` | Download a segmentation mask |

Interactive API documentation is available through FastAPI Swagger UI.

## Project Structure

```text
MedVision-AI/
│
├── src/
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   └── index.html
│
├── data/
├── models/
├── notebooks/
├── tests/
├── docs/
│
├── README.md
└── .gitignore
```

*Note: Some folders are reserved for future organization and development.*

## Running the Project Locally

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies.
- Node.js and npm.
- Git.
- Trained model checkpoints and the required datasets.

### 1. Clone the repository

```bash
git clone https://github.com/MouliJanghel/MedVision-AI.git
cd MedVision-AI
```

### 2. Activate the Python environment

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If setting up for the first time, create a virtual environment and install the project's required Python dependencies.

### 3. Start the backend

From the project root:

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8002
```

Backend:

```text
http://127.0.0.1:8002
```

Swagger documentation:

```text
http://127.0.0.1:8002/docs
```

### 4. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local frontend address displayed by Vite, typically:

```text
http://localhost:5173/
```

## Current Limitations and Future Improvements

The current implementation is an educational prototype. The following improvements remain:

- Improve classification performance through further training and evaluation.
- Improve the 3D U-Net segmentation Dice score.
- Add DICOM image processing and complete Orthanc integration.
- Implement authentication and role-based access control.
- Improve audit-log security and integrity.
- Add automated tests and deployment configuration.
- Integrate DVC for dataset and model versioning.
- Add Docker-based deployment.

Orthanc, Docker, and DVC are considered future integration work and are not currently presented as completed integrations.

## Author

**Mouli Janghel**

AI & Machine Learning Engineering

## License

This project is developed for educational and research purposes. Add an appropriate open-source license before permitting reuse or redistribution.
