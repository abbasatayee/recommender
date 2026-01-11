"""
Pydantic schemas for API request and response models.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class MovieDetails(BaseModel):
    """Movie details model."""
    movie_id: int
    title: str
    genres: List[str]
    tags: List[str]  # Same as genres
    imdb_url: str


class PredictionRequest(BaseModel):
    """Request for single prediction."""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_id: int = Field(..., ge=0, description="Item ID (0-indexed)")


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    user_id: int
    item_id: int
    score: float
    movie: Optional[MovieDetails] = None
    message: str = "Prediction successful"


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_ids: List[int] = Field(..., min_items=1, description="List of item IDs")


class PredictionItem(BaseModel):
    """Single prediction item with movie details."""
    item_id: int
    score: float
    movie: Optional[MovieDetails] = None


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    user_id: int
    predictions: List[PredictionItem]
    message: str = "Batch prediction successful"


class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    candidate_item_ids: Optional[List[int]] = Field(
        None, description="Optional candidate items. If None, uses all items."
    )


class RecommendationItem(BaseModel):
    """Single recommendation item with movie details."""
    item_id: int
    score: float
    movie: Optional[MovieDetails] = None


class RecommendationResponse(BaseModel):
    """Response for recommendations."""
    user_id: int
    recommendations: List[RecommendationItem]
    k: int
    message: str = "Recommendations generated successfully"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    ncf_loaded: bool
    autorec_loaded: bool
    device: str
    ncf_config: Dict[str, int]
    autorec_config: Dict[str, Any]
