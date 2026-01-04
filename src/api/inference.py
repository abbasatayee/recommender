"""
NCF Inference API
A FastAPI-based inference service for Neural Collaborative Filtering model.
"""
import os
import sys

# Get the absolute path of the directory containing this file
current_file_path = os.path.abspath(__file__)
api_dir = os.path.dirname(current_file_path)
# src is the parent of src/api
src_path = os.path.dirname(api_dir)

# Add src to sys.path if found
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
# we need to import the NCF model from the helpers module in order to load the model
from helpers import NCF

MODEL_PATH = "models/NeuMF.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters (from training)
# These should match the training configuration
USER_NUM = 6038  # Number of users in the dataset
ITEM_NUM = 3533  # Number of items in the dataset
FACTOR_NUM = 32  # Embedding dimension
NUM_LAYERS = 3   # Number of MLP layers
DROPOUT = 0.0    # Dropout rate
MODEL_NAME = "NeuMF"

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: str, device: str = "cpu"):
    """
    Load the trained NCF model.
    
    Parameters:
    - model_path: Path to the saved model file
    - device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
    - Loaded model in evaluation mode
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, weights_only=False)
    print(f"Model: {model}")
    model.eval()  # Set to evaluation mode
    print(f"✓ Model loaded successfully on {device}")
    print("=" * 70)
    
    return model

# Load the model at startup
try:
    model = load_model(MODEL_PATH, DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NCF Inference API",
    description="Neural Collaborative Filtering model inference API for recommendation systems",
    version="1.0.0"
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_id: int = Field(..., ge=0, description="Item ID (0-indexed)")

class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    user_id: int
    item_id: int
    score: float
    message: str = "Prediction successful"

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    item_ids: List[int] = Field(..., min_items=1, description="List of item IDs to predict")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    user_id: int
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions with item_id and score")
    message: str = "Batch prediction successful"

class RecommendationRequest(BaseModel):
    """Request model for recommendations"""
    user_id: int = Field(..., ge=0, description="User ID (0-indexed)")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    candidate_item_ids: Optional[List[int]] = Field(None, description="Optional list of candidate items. If None, uses all items.")

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: int
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations with item_id and score, sorted by score")
    k: int
    message: str = "Recommendations generated successfully"

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    device: str
    user_num: int
    item_num: int

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_user_id(user_id: int):
    """Validate user ID is within valid range"""
    if user_id < 0 or user_id >= USER_NUM:
        raise HTTPException(
            status_code=400,
            detail=f"User ID must be between 0 and {USER_NUM - 1}, got {user_id}"
        )

def validate_item_id(item_id: int):
    """Validate item ID is within valid range"""
    if item_id < 0 or item_id >= ITEM_NUM:
        raise HTTPException(
            status_code=400,
            detail=f"Item ID must be between 0 and {ITEM_NUM - 1}, got {item_id}"
        )

def predict_score(user_id: int, item_id: int) -> float:
    """
    Predict interaction score for a single user-item pair.
    
    Parameters:
    - user_id: User ID
    - item_id: Item ID
    
    Returns:
    - Prediction score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    validate_user_id(user_id)
    validate_item_id(item_id)
    
    # Convert to tensors
    user_tensor = torch.LongTensor([user_id]).to(DEVICE)
    item_tensor = torch.LongTensor([item_id]).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        score = model(user_tensor, item_tensor)
        score = score.cpu().item()
    
    return score

def predict_batch(user_id: int, item_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Predict scores for a user and multiple items.
    
    Parameters:
    - user_id: User ID
    - item_ids: List of item IDs
    
    Returns:
    - List of predictions with item_id and score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate inputs
    validate_user_id(user_id)
    for item_id in item_ids:
        validate_item_id(item_id)
    
    # Convert to tensors
    user_tensor = torch.LongTensor([user_id] * len(item_ids)).to(DEVICE)
    item_tensor = torch.LongTensor(item_ids).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
        scores = scores.cpu().numpy()
    
    # Format results
    predictions = [
        {"item_id": int(item_id), "score": float(score)}
        for item_id, score in zip(item_ids, scores)
    ]
    
    # Sort by score (descending)
    predictions.sort(key=lambda x: x["score"], reverse=True)
    
    return predictions

def get_recommendations(user_id: int, k: int = 10, candidate_item_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Get top-K recommendations for a user.
    
    Parameters:
    - user_id: User ID
    - k: Number of recommendations
    - candidate_item_ids: Optional list of candidate items. If None, uses all items.
    
    Returns:
    - List of top-K recommendations with item_id and score
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate user ID
    validate_user_id(user_id)
    
    # Get candidate items
    if candidate_item_ids is None:
        # Use all items
        candidate_item_ids = list(range(ITEM_NUM))
    else:
        # Validate candidate items
        for item_id in candidate_item_ids:
            validate_item_id(item_id)
        # Remove duplicates
        candidate_item_ids = list(set(candidate_item_ids))
    
    if len(candidate_item_ids) == 0:
        raise HTTPException(status_code=400, detail="No valid candidate items provided")
    
    # Limit k to available items
    k = min(k, len(candidate_item_ids))
    
    # Convert to tensors
    user_tensor = torch.LongTensor([user_id] * len(candidate_item_ids)).to(DEVICE)
    item_tensor = torch.LongTensor(candidate_item_ids).to(DEVICE)
    
    # Get predictions
    with torch.no_grad():
        scores = model(user_tensor, item_tensor)
        scores = scores.cpu().numpy()
    
    # Get top-K items
    top_k_indices = np.argsort(scores)[::-1][:k]  # Sort descending, take top K
    
    # Format results
    recommendations = [
        {
            "item_id": int(candidate_item_ids[i]),
            "score": float(scores[i])
        }
        for i in top_k_indices
    ]
    
    return recommendations

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NCF Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=DEVICE,
        user_num=USER_NUM,
        item_num=ITEM_NUM
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict interaction score for a single user-item pair.
    
    - **user_id**: User ID (0-indexed, must be < USER_NUM)
    - **item_id**: Item ID (0-indexed, must be < ITEM_NUM)
    
    Returns the predicted interaction score (higher = more likely user will like item).
    """
    try:
        score = predict_score(request.user_id, request.item_id)
        return PredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            score=score
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """
    Predict interaction scores for a user and multiple items.
    
    - **user_id**: User ID (0-indexed, must be < USER_NUM)
    - **item_ids**: List of item IDs to predict
    
    Returns predictions sorted by score (descending).
    """
    try:
        predictions = predict_batch(request.user_id, request.item_ids)
        return BatchPredictionResponse(
            user_id=request.user_id,
            predictions=predictions
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendation"])
async def recommend(request: RecommendationRequest):
    """
    Get top-K item recommendations for a user.
    
    - **user_id**: User ID (0-indexed, must be < USER_NUM)
    - **k**: Number of recommendations to return (default: 10, max: 100)
    - **candidate_item_ids**: Optional list of candidate items. If not provided, uses all items.
    
    Returns top-K recommendations sorted by score (descending).
    """
    try:
        recommendations = get_recommendations(
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

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NCF Inference API")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Users: {USER_NUM}")
    print(f"Items: {ITEM_NUM}")
    print("=" * 70)
    print("Starting server...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("=" * 70)
    
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

