from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import API routes
from app.api.routes.health import router as health_router
from app.api.routes.detect import router as detect_router

# Create FastAPI app
app = FastAPI(
    title="Medical Image Tamper Detection API",
    description="Detects deepfake or tampered medical images",
    version="1.0.0"
)

# Allow other websites/apps to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(health_router, prefix="/api")
app.include_router(detect_router, prefix="/api")

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Medical Tamper Detection API is running"
    }
