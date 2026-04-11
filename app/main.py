import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# Serve static files (CSS, JS, images)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
output_dir = os.path.join(os.path.dirname(BASE_DIR), "output")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# Jinja2 templates
templates_dir = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=templates_dir)

# Register routes
app.include_router(health_router, prefix="/api")
app.include_router(detect_router, prefix="/api")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("Home.html", {"request": request})