"""
General API routes (root, health check).
"""
from fastapi import APIRouter
from ..models.schemas import HealthResponse
from ..config import API_TITLE, API_VERSION, DEVICE, NCF_CONFIG, AUTOREC_CONFIG
from ..core.service import get_ncf_model, get_autorec_model

router = APIRouter()


@router.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": API_TITLE,
        "version": API_VERSION,
        "models": ["NCF", "AutoRec"],
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    """Health check endpoint."""
    ncf_model = get_ncf_model()
    autorec_model = get_autorec_model()
    
    return HealthResponse(
        status="healthy" if (ncf_model or autorec_model) else "unhealthy",
        ncf_loaded=ncf_model is not None,
        autorec_loaded=autorec_model is not None,
        device=DEVICE,
        ncf_config={
            "user_num": NCF_CONFIG["user_num"],
            "item_num": NCF_CONFIG["item_num"]
        },
        autorec_config={
            "user_num": AUTOREC_CONFIG["user_num"],
            "item_num": AUTOREC_CONFIG["item_num"]
        }
    )
