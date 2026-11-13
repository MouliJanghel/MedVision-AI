
import base64
import io
import os
import tempfile
import uuid

import nibabel as nib
import numpy as np
import torch

from nibabel.processing import resample_from_to

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from torchvision import models, transforms
from PIL import Image
from torch import nn

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped
)

from monai.inferers import sliding_window_inference

from src.models.unet_model import model as unet_model

from src.database.db import (
    init_db,
    create_scan,
    review_scan,
    get_scans,
    get_audit_logs
)


# =====================================
# 1. INITIALIZE FASTAPI
# =====================================

app = FastAPI(
    title="MedVision-AI",
    description="AI-Assisted Medical Imaging and Diagnosis System",
    version="1.1.0"
)

init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================
# 2. DEVICE CONFIGURATION
# =====================================

device = torch.device("cpu")


# =====================================
# 3. LOAD DENSENET121 MODEL
# =====================================

model_path = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_densenet121.pth"
)

model = models.densenet121(weights=None)

model.classifier = nn.Linear(
    model.classifier.in_features,
    2
)

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"DenseNet121 checkpoint not found: {model_path}"
    )

checkpoint = torch.load(
    model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

print("DenseNet121 model loaded successfully!")


# =====================================
# 4. LOAD 3D U-NET MODEL
# =====================================

unet_model.to(device)
unet_model.eval()

print("3D U-Net model loaded successfully!")


# =====================================
# 5. CHEST X-RAY PREPROCESSING
# =====================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class_names = ["No Pneumonia", "Pneumonia"]


# =====================================
# 6. 3D CT PREPROCESSING
# =====================================

segmentation_preprocess = Compose([
    LoadImaged(keys=["image"]),

    EnsureChannelFirstd(keys=["image"]),

    Orientationd(
        keys=["image"],
        axcodes="RAS"
    ),

    Spacingd(
        keys=["image"],
        pixdim=(1.5, 1.5, 2.0),
        mode="bilinear"
    ),

    ScaleIntensityRanged(
        keys=["image"],
        a_min=-57,
        a_max=164,
        b_min=0.0,
        b_max=1.0,
        clip=True
    ),

    CropForegroundd(
        keys=["image"],
        source_key="image"
    ),

    EnsureTyped(keys=["image"])
])


# =====================================
# 7. REVIEW REQUEST MODEL
# =====================================

class ReviewRequest(BaseModel):
    decision: str
    reviewed_by: str
    notes: str | None = None


# =====================================
# 8. HOME ENDPOINT
# =====================================

@app.get("/")
def home():

    return {
        "message": "Welcome to MedVision-AI",
        "status": "Backend running successfully",
        "version": "1.1.0"
    }


# =====================================
# 9. HEALTH CHECK
# =====================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "MedVision-AI",
        "densenet121_loaded": True,
        "unet_loaded": True,
        "database": "initialized",
        "gradcam_enabled": True
    }


# =====================================
# 10. SCAN UPLOAD ENDPOINT
# =====================================

@app.post("/upload-scan")
async def upload_scan(file: UploadFile = File(...)):

    filename = file.filename or ""
    lower_name = filename.lower()

    allowed = (
        lower_name.endswith((".dcm", ".png", ".jpg", ".jpeg"))
        or lower_name.endswith((".nii", ".nii.gz"))
    )

    if not allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported scan format"
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded scan is empty"
        )

    return {
        "filename": filename,
        "message": "Scan uploaded successfully",
        "size_bytes": len(content)
    }


# =====================================
# 11. GRAD-CAM COLOR MAP
# =====================================

def create_color_heatmap(cam):

    cam = np.clip(cam, 0, 1)

    # Generate a Jet-style RGB heatmap
    red = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)

    heatmap = np.stack(
        [red, green, blue],
        axis=-1
    )

    heatmap = (heatmap * 255).astype(np.uint8)

    return Image.fromarray(heatmap, mode="RGB")


# =====================================
# 12. GRAD-CAM GENERATION
# =====================================

def generate_gradcam(image_tensor, original_image, predicted_class):

    activations = []
    gradients = []

    target_layer = model.features.denseblock4

    def forward_hook(module, inputs, output):

        activations.append(output)

        output.register_hook(
            lambda grad: gradients.append(grad)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:

        model.zero_grad(set_to_none=True)

        # Enable gradients specifically for Grad-CAM
        with torch.enable_grad():

            outputs = model(image_tensor)

            score = outputs[0, predicted_class]

            score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "Unable to collect Grad-CAM activations or gradients."
            )

        feature_maps = activations[0]
        grads = gradients[0]

        # Calculate channel importance
        weights = grads.mean(
            dim=(2, 3),
            keepdim=True
        )

        # Weighted feature-map combination
        cam = (weights * feature_maps).sum(dim=1)

        cam = torch.relu(cam)

        # Resize to original model input dimensions
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0]

        # Normalize heatmap
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam_array = cam.detach().cpu().numpy()

        # Generate colored heatmap
        heatmap_image = create_color_heatmap(cam_array)

        # Prepare original image
        original_image = original_image.convert("RGB")
        original_image = original_image.resize((224, 224))

        # Overlay heatmap on original X-ray
        overlay = Image.blend(
            original_image,
            heatmap_image,
            alpha=0.40
        )

        # Convert overlay to Base64
        buffer = io.BytesIO()

        overlay.save(
            buffer,
            format="PNG"
        )

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return encoded_image

    finally:

        handle.remove()
        model.zero_grad(set_to_none=True)


# =====================================
# 13. PNEUMONIA PREDICTION + GRAD-CAM
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    filename = file.filename or ""
    extension = os.path.splitext(filename)[1].lower()

    if extension not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PNG or JPG chest X-ray."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:

        # Read image safely
        image = Image.open(
            io.BytesIO(content)
        ).convert("RGB")

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # AI prediction
        with torch.inference_mode():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[
                0, predicted_class
            ].item()

        # Generate Grad-CAM separately with gradients enabled
        heatmap_base64 = generate_gradcam(
            image_tensor,
            image,
            predicted_class
        )

        prediction = class_names[predicted_class]

        confidence_percent = round(
            confidence * 100,
            2
        )

        # Save scan in SQLite
        scan_id = create_scan(
            filename=filename,
            scan_type="chest_xray",
            prediction=prediction,
            confidence=confidence_percent
        )

        return {
            "scan_id": scan_id,
            "filename": filename,
            "prediction": prediction,
            "confidence": confidence_percent,
            "review_status": "pending",
            "gradcam_heatmap": heatmap_base64,
            "gradcam_format": "PNG",
            "message": "Prediction and Grad-CAM generated successfully."
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Prediction error:", str(error))

        raise HTTPException(
            status_code=400,
            detail=f"Unable to process image: {str(error)}"
        )


# =====================================
# 14. VIEW ALL SCANS
# =====================================

@app.get("/scans")
def list_scans():

    scans = get_scans()

    return {
        "total_scans": len(scans),
        "scans": scans
    }


# =====================================
# 15. DOCTOR APPROVAL / REJECTION
# =====================================

@app.post("/scans/{scan_id}/review")
def doctor_review(scan_id: str, request: ReviewRequest):

    try:

        uuid.UUID(scan_id)

        if not request.reviewed_by.strip():
            raise HTTPException(
                status_code=400,
                detail="Doctor name is required"
            )

        decision = request.decision.lower().strip()

        if decision not in ["approved", "rejected"]:
            raise HTTPException(
                status_code=400,
                detail="Decision must be approved or rejected."
            )

        review_scan(
            scan_id=scan_id,
            decision=decision,
            reviewed_by=request.reviewed_by.strip(),
            notes=request.notes
        )

        return {
            "scan_id": scan_id,
            "review_status": decision,
            "reviewed_by": request.reviewed_by.strip(),
            "message": "Doctor review saved successfully"
        }

    except HTTPException:
        raise

    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


# =====================================
# 16. VIEW AUDIT HISTORY
# =====================================

@app.get("/audit-logs")
def audit_history(scan_id: str | None = None):

    if scan_id:
        try:
            uuid.UUID(scan_id)

        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid scan ID"
            )

    logs = get_audit_logs(scan_id)

    return {
        "total_records": len(logs),
        "audit_logs": logs
    }


# =====================================
# 17. DOWNLOAD SEGMENTATION MASK
# =====================================

@app.get("/download-mask/{mask_id}")
def download_mask(mask_id: str):

    try:

        uuid.UUID(mask_id)

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail="Invalid mask ID"
        )

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    output_dir = os.path.join(
        project_root,
        "data",
        "segmentations"
    )

    mask_path = os.path.join(
        output_dir,
        f"predicted_spleen_mask_{mask_id}.nii.gz"
    )

    if not os.path.isfile(mask_path):

        raise HTTPException(
            status_code=404,
            detail="Segmentation mask not found"
        )

    return FileResponse(
        path=mask_path,
        media_type="application/gzip",
        filename="predicted_spleen_mask.nii.gz"
    )


# =====================================
# 18. 3D CT SEGMENTATION
# =====================================

@app.post("/segment")
async def segment(file: UploadFile = File(...)):

    filename = file.filename or ""
    lower_name = filename.lower()

    if not lower_name.endswith((".nii", ".nii.gz")):

        raise HTTPException(
            status_code=400,
            detail="Please upload a NIfTI CT scan (.nii or .nii.gz)."
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded CT scan is empty."
        )

    input_path = None

    try:

        # Save uploaded CT temporarily
        suffix = (
            ".nii.gz"
            if lower_name.endswith(".nii.gz")
            else ".nii"
        )

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False
        ) as temp_file:

            temp_file.write(content)
            input_path = temp_file.name

        # Load original CT
        original_ct = nib.load(input_path)

        if len(original_ct.shape) != 3:

            raise HTTPException(
                status_code=400,
                detail="Only 3D CT scans are supported."
            )

        original_shape = original_ct.shape[:3]

        # Preprocess CT
        processed_data = segmentation_preprocess({
            "image": input_path
        })

        ct_image = processed_data["image"]

        input_tensor = ct_image.unsqueeze(0).to(device)

        # Run 3D U-Net
        with torch.inference_mode():

            outputs = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=unet_model,
                overlap=0.25
            )

            predicted_mask = torch.argmax(
                outputs,
                dim=1
            )[0]

        mask_array = (
            predicted_mask
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        processed_affine = (
            ct_image.affine
            .detach()
            .cpu()
            .numpy()
        )

        segmentation_image = nib.Nifti1Image(
            mask_array,
            processed_affine
        )

        # Restore mask to original CT space
        resampled_image = resample_from_to(
            segmentation_image,
            (original_ct.shape[:3], original_ct.affine),
            order=0
        )

        restored_mask = (
            np.rint(resampled_image.get_fdata())
            .astype(np.uint8)
        )

        restored_image = nib.Nifti1Image(
            restored_mask,
            original_ct.affine,
            header=original_ct.header.copy()
        )

        restored_image.set_data_dtype(np.uint8)

        # Save mask
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        output_dir = os.path.join(
            project_root,
            "data",
            "segmentations"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        mask_id = str(uuid.uuid4())

        output_path = os.path.join(
            output_dir,
            f"predicted_spleen_mask_{mask_id}.nii.gz"
        )

        nib.save(
            restored_image,
            output_path
        )

        foreground_voxels = int(
            np.count_nonzero(restored_mask)
        )

        relative_mask_path = os.path.relpath(
            output_path,
            project_root
        )

        # Save segmentation record
        scan_id = create_scan(
            filename=filename,
            scan_type="spleen_ct",
            prediction="Spleen segmentation completed",
            mask_path=relative_mask_path
        )

        return {
            "scan_id": scan_id,
            "mask_id": mask_id,
            "filename": filename,
            "segmentation": "Completed",
            "segmented_region": "Spleen",
            "original_ct_shape": list(original_shape),
            "processed_mask_shape": list(mask_array.shape),
            "restored_mask_shape": list(restored_mask.shape),
            "foreground_voxels": foreground_voxels,
            "mask_file_format": "NIfTI (.nii.gz)",
            "mask_saved": True,
            "saved_file": relative_mask_path,
            "download_url": f"/download-mask/{mask_id}",
            "message": (
                "3D CT segmentation completed and restored "
                "to original CT space."
            )
        }

    except HTTPException:
        raise

    except Exception as error:

        print("Segmentation error:", str(error))

        raise HTTPException(
            status_code=400,
            detail=f"Unable to segment CT scan: {str(error)}"
        )

    finally:

        if input_path and os.path.exists(input_path):
            os.remove(input_path)
