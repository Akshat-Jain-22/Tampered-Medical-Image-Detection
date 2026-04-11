import os
import uuid
from fastapi import UploadFile
from app.config.settings import UPLOAD_DIR

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.dcm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def is_valid_file_extension(filename: str) -> bool:
    """Check if file has valid extension"""
    ext = os.path.splitext(filename)[-1].lower()
    return ext in ALLOWED_EXTENSIONS

def save_temp_file(file: UploadFile) -> str:
    """
    Save uploaded file temporarily

    Args:
        file: UploadFile object

    Returns:
        str: Path to saved file

    Raises:
        ValueError: If file is invalid
    """
    # Validate filename
    if not file.filename:
        raise ValueError("No filename provided")

    # Check file extension
    if not is_valid_file_extension(file.filename):
        raise ValueError(f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Create uploads folder if not exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate unique filename
    ext = os.path.splitext(file.filename)[-1]
    file_name = f"file_{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    # Save file
    try:
        with open(file_path, "wb") as f:
            content = file.file.read()

            # Check file size
            if len(content) > MAX_FILE_SIZE:
                raise ValueError("File too large (max 50MB)")

            f.write(content)

        return file_path

    except Exception as e:
        # Clean up if something went wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise Exception(f"Error saving file: {str(e)}")
