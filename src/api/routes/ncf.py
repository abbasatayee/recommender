"""
NCF model API routes.
"""
from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    RecommendationRequest, RecommendationResponse,
    MovieDetails
)
from ..core.service import get_ncf_engine
from ..core.movie_metadata import get_metadata_manager

router = APIRouter(prefix="/ncf", tags=["NCF"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict score for user-item pair using NCF."""
    try:
        engine = get_ncf_engine()
        score = engine.predict(request.user_id, request.item_id)
        movie_info = get_metadata_manager().get_movie_info(request.item_id)
        movie_details = MovieDetails(**movie_info) if movie_info else None
        
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            score=score,
            movie=movie_details
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch predict scores using NCF."""
    try:
        engine = get_ncf_engine()
        predictions = engine.predict_batch(request.user_id, request.item_ids)
        return BatchPredictionResponse(
            user_id=request.user_id,
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get top-K recommendations using NCF."""
    try:
        engine = get_ncf_engine()
        recommendations = engine.recommend(
            request.user_id,
            request.k,
            request.candidate_item_ids
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            k=len(recommendations)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
