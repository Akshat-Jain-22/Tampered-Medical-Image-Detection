import os
import uuid
from fastapi import UploadFile

UPLOAD_DIR = "Uploads/Images"

def save_temp_file(file: UploadFile) -> str:
    # Create uploads folder if not exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Generate unique filename
    file_name = f"{uuid.uuid4()}.png"
    file_path = os.path.join(UPLOAD_DIR, file_name)

    # Save file
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return file_path
