"""
NCF Inference API
A FastAPI-based inference service for Neural Collaborative Filtering model.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# ============================================================================
# STEP 5: NCF MODEL ARCHITECTURE
# ============================================================================

"""
This step implements the Neural Collaborative Filtering (NCF) model.

The NCF model has three variants:
1. GMF: Only Generalized Matrix Factorization (linear)
2. MLP: Only Multi-Layer Perceptron (non-linear)
3. NeuMF: Neural Matrix Factorization (combines GMF + MLP)
"""

class NCF(nn.Module):
    """
    Neural Collaborative Filtering Model
    
    This model learns user and item embeddings and combines them using
    either GMF (linear) or MLP (non-linear) or both (NeuMF).
    
    Architecture:
    1. Embedding layers: Convert user/item IDs to dense vectors
    2. GMF path: Element-wise product of embeddings (linear interaction)
    3. MLP path: Deep neural network (non-linear interaction)
    4. Prediction layer: Combines GMF and/or MLP outputs to predict score
    """
    
    def __init__(self, user_num, item_num, factor_num, num_layers,
                 dropout, model_name, GMF_model=None, MLP_model=None):
        """
        Initialize the NCF model.
        
        Parameters:
        - user_num: Total number of users
        - item_num: Total number of items
        - factor_num: Dimension of embedding vectors (e.g., 32)
        - num_layers: Number of layers in MLP component
        - dropout: Dropout rate for regularization
        - model_name: 'MLP', 'GMF', 'NeuMF-end', or 'NeuMF-pre'
        - GMF_model: Pre-trained GMF model (for NeuMF-pre)
        - MLP_model: Pre-trained MLP model (for NeuMF-pre)
        """
        super(NCF, self).__init__()
        
        # Store configuration
        self.dropout = dropout
        self.model_name = model_name
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        
        # ====================================================================
        # EMBEDDING LAYERS
        # ====================================================================
        # Embeddings convert user/item IDs (integers) to dense vectors
        
        # GMF embeddings: factor_num dimensions
        # Used for Generalized Matrix Factorization (linear interactions)
        if model_name != 'MLP':  # MLP doesn't use GMF
            self.embed_user_GMF = nn.Embedding(user_num, factor_num)
            self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        
        # MLP embeddings: Larger dimension for deeper networks
        # Dimension = factor_num * 2^(num_layers-1)
        # Example: factor_num=32, num_layers=3 → 32 * 2^2 = 128 dimensions
        if model_name != 'GMF':  # GMF doesn't use MLP
            mlp_embed_dim = factor_num * (2 ** (num_layers - 1))
            self.embed_user_MLP = nn.Embedding(user_num, mlp_embed_dim)
            self.embed_item_MLP = nn.Embedding(item_num, mlp_embed_dim)
        
        # ====================================================================
        # MLP LAYERS (Multi-Layer Perceptron)
        # ====================================================================
        # Build MLP with decreasing dimensions
        # Example with factor_num=32, num_layers=3:
        #   Input: 128*2 = 256 (concatenated user + item embeddings)
        #   Layer 1: 256 → 128
        #   Layer 2: 128 → 64
        #   Layer 3: 64 → 32
        #   Output: 32 dimensions
        
        if model_name != 'GMF':  # GMF doesn't use MLP
            MLP_modules = []
            for i in range(num_layers):
                # Calculate input size for this layer
                input_size = factor_num * (2 ** (num_layers - i))
                
                # Add dropout for regularization
                MLP_modules.append(nn.Dropout(p=self.dropout))
                
                # Add linear layer (halves the dimension)
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                
                # Add ReLU activation (non-linearity)
                MLP_modules.append(nn.ReLU())
            
            # Combine all MLP layers into a sequential module
            self.MLP_layers = nn.Sequential(*MLP_modules)
        
        # ====================================================================
        # PREDICTION LAYER
        # ====================================================================
        # Final layer that outputs the interaction score
        
        if self.model_name in ['MLP', 'GMF']:
            # Single path: just factor_num dimensions
            predict_size = factor_num
        else:
            # NeuMF: concatenate GMF (factor_num) + MLP (factor_num) = 2*factor_num
            predict_size = factor_num * 2
        
        self.predict_layer = nn.Linear(predict_size, 1)
        
        # Initialize weights
        self._init_weight_()
    
    def _init_weight_(self):
        """
        Initialize model weights.
        
        Different initialization strategies:
        - Embeddings: Small random values (std=0.01)
        - MLP layers: Xavier uniform (good for ReLU)
        - Prediction layer: Kaiming uniform (good for sigmoid)
        - Biases: Zero
        """
        if not self.model_name == 'NeuMF-pre':
            # Random initialization for training from scratch
            
            # Embedding initialization: Small random values
            # This prevents embeddings from starting too large
            if hasattr(self, 'embed_user_GMF'):
                nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            if hasattr(self, 'embed_item_GMF'):
                nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            if hasattr(self, 'embed_user_MLP'):
                nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            if hasattr(self, 'embed_item_MLP'):
                nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
            
            # MLP layer initialization: Xavier uniform
            # Good for layers with ReLU activation
            if hasattr(self, 'MLP_layers'):
                for m in self.MLP_layers:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
            
            # Prediction layer initialization: Kaiming uniform
            # Good for layers before sigmoid activation
            nn.init.kaiming_uniform_(self.predict_layer.weight, 
                                    a=1, nonlinearity='sigmoid')
            
            # Initialize all biases to zero
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # Pre-trained initialization (for NeuMF-pre)
            # Copy weights from pre-trained GMF and MLP models
            
            # Copy embedding weights
            self.embed_user_GMF.weight.data.copy_(
                self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(
                self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(
                self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(
                self.MLP_model.embed_item_MLP.weight)
            
            # Copy MLP layer weights
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            # Combine prediction layer weights from GMF and MLP
            predict_weight = torch.cat([
                self.GMF_model.predict_layer.weight, 
                self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = (self.GMF_model.predict_layer.bias + 
                           self.MLP_model.predict_layer.bias) / 2
            
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(predict_bias)
    
    def forward(self, user, item):
        """
        Forward pass: Predict user-item interaction scores.
        
        Parameters:
        - user: Tensor of user IDs [batch_size]
        - item: Tensor of item IDs [batch_size]
        
        Returns:
        - prediction: Tensor of predicted scores [batch_size]
        """
        # ====================================================================
        # GMF PATH (Generalized Matrix Factorization)
        # ====================================================================
        # Linear interaction: element-wise product of embeddings
        # Similar to traditional matrix factorization
        
        if self.model_name != 'MLP':
            # Get embeddings
            embed_user_GMF = self.embed_user_GMF(user)  # [batch_size, factor_num]
            embed_item_GMF = self.embed_item_GMF(item)  # [batch_size, factor_num]
            
            # Element-wise product (linear interaction)
            output_GMF = embed_user_GMF * embed_item_GMF  # [batch_size, factor_num]
        
        # ====================================================================
        # MLP PATH (Multi-Layer Perceptron)
        # ====================================================================
        # Non-linear interaction: deep neural network
        
        if self.model_name != 'GMF':
            # Get embeddings
            embed_user_MLP = self.embed_user_MLP(user)  # [batch_size, mlp_dim]
            embed_item_MLP = self.embed_item_MLP(item)   # [batch_size, mlp_dim]
            
            # Concatenate user and item embeddings
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)  # [batch_size, mlp_dim*2]
            
            # Pass through MLP layers (with dropout and ReLU)
            output_MLP = self.MLP_layers(interaction)  # [batch_size, factor_num]
        
        # ====================================================================
        # COMBINE PATHS
        # ====================================================================
        if self.model_name == 'GMF':
            # Only GMF path
            concat = output_GMF
        elif self.model_name == 'MLP':
            # Only MLP path
            concat = output_MLP
        else:
            # NeuMF: Concatenate both paths
            concat = torch.cat((output_GMF, output_MLP), -1)  # [batch_size, factor_num*2]
        
        # ====================================================================
        # PREDICTION
        # ====================================================================
        # Final linear layer outputs interaction score
        prediction = self.predict_layer(concat)  # [batch_size, 1]
        
        # Flatten to [batch_size]
        return prediction.view(-1)

print("=" * 70)
print("STEP 5: NCF Model Architecture")
print("=" * 70)
print("✓ NCF model class defined")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

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

