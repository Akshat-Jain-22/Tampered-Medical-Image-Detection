from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.templating import Jinja2Templates
from app.services.inference import run_inference
from app.utils.file_handler import save_temp_file, is_valid_file_extension
import os

router = APIRouter()

# Setup templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Allowed MIME types
ALLOWED_IMAGE_TYPES = [
    "image/png",
    "image/jpeg",
    "application/dicom",
    "application/octet-stream"
]

@router.get("/detect-medical-tamper")
async def detect_page(request: Request):
    """Render detection interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/detect-medical-tamper")
async def detect_medical_tamper(file: UploadFile = File(...)):
    """
    Detect tampering in medical image

    Supported formats: JPG, PNG, JPEG, DCM

    Returns:
        - classification: "Tampered" or "Authentic"
        - tampered_probability: Float (0-1)
        - authentic_probability: Float (0-1)
        - heatmap: Array showing tampering location
        - heatmap_path: URL to saved heatmap image
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check file extension
    if not is_valid_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Allowed: JPG, PNG, JPEG, DCM"
        )

    # Save uploaded file
    try:
        file_path = save_temp_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

    # Run inference
    try:
        result = run_inference(file_path)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

