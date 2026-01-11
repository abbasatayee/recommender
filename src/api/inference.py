"""
Recommendation System Inference API
A FastAPI-based inference service for NCF and AutoRec models.
"""
import os
import sys

# Setup path - ensure src is in path for imports
current_file_path = os.path.abspath(__file__)
api_dir = os.path.dirname(current_file_path)
src_path = os.path.dirname(api_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import uvicorn
from fastapi import FastAPI

# Use absolute imports from src (works when run as module or directly)
from api.config import (
    NCF_MODEL_PATH, AUTOREC_MODEL_PATH, DEVICE,
    API_TITLE, API_DESCRIPTION, API_VERSION, API_HOST, API_PORT
)
from api.core.service import initialize_models, get_ncf_model, get_autorec_model
from api.routes import general, ncf, autorec


# Initialize models on startup
initialize_models()


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)


# ============================================================================
# REGISTER ROUTES
# ============================================================================

app.include_router(general.router)
app.include_router(ncf.router)
app.include_router(autorec.router)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Recommendation System Inference API")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"NCF Model: {NCF_MODEL_PATH} ({'✓' if get_ncf_model() else '✗'})")
    print(f"AutoRec Model: {AUTOREC_MODEL_PATH} ({'✓' if get_autorec_model() else '✗'})")
    print("=" * 70)
    print("Starting server...")
    print(f"API Documentation: http://localhost:{API_PORT}/docs")
    print("=" * 70)
    
    # Run the app - use the app directly instead of string reference
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False
    )
