"""
AutoRec model API routes.
"""
from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    RecommendationRequest, RecommendationResponse,
    MovieDetails
)
from ..core.service import get_autorec_engine
from ..core.movie_metadata import get_metadata_manager, get_random_top_rated_movie

router = APIRouter(prefix="/autorec", tags=["AutoRec"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict rating for user-item pair using AutoRec."""
    try:
        engine = get_autorec_engine()
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
    """Batch predict ratings using AutoRec."""
    try:
        engine = get_autorec_engine()
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
    """Get top-K recommendations using AutoRec."""
    try:
        engine = get_autorec_engine()
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


@router.get("/random-top-rated", response_model=PredictionResponse)
async def get_random_top_rated(user_id: int):
    """
    Get a random top-rated movie that the user has not seen before.
    
    Args:
        user_id: User ID (0-indexed) to exclude their seen movies
    """
    try:
        # Validate user_id
        engine = get_autorec_engine()
        engine.validate_user(user_id, engine.config["user_num"], engine.model_name)
        
        result = get_random_top_rated_movie(user_id=user_id, top_n=100, min_ratings=10)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No unseen top-rated movies found for user {user_id}. The user may have already seen all top-rated movies."
            )
        
        item_id, movie_info = result
        movie_details = MovieDetails(**movie_info)
        
        return PredictionResponse(
            user_id=user_id,
            item_id=item_id,
            score=5.0,  # Top-rated movies have high scores
            movie=movie_details,
            message="Random top-rated movie retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get random top-rated movie: {str(e)}")
