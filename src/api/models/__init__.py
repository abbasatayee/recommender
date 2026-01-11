"""
Pydantic models and schemas.
"""
from .schemas import (
    MovieDetails,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionItem,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    HealthResponse
)

__all__ = [
    "MovieDetails",
    "PredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "PredictionItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendationItem",
    "HealthResponse"
]
