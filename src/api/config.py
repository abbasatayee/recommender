"""
Configuration settings for the recommendation system API.
"""
import torch
from typing import Dict, Any

# ============================================================================
# PATHS
# ============================================================================

NCF_MODEL_PATH = "models/NeuMF.pth"
AUTOREC_MODEL_PATH = "models/AutoRec-best.pth"
TRAIN_DATA_PATH = "data/ml-1m.train.rating"
RATINGS_FILE = "data/ml-1m/ratings.dat"
MOVIES_FILE = "data/ml-1m/movies.dat"

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

NCF_CONFIG: Dict[str, Any] = {
    "user_num": 6038,
    "item_num": 3533,
    "factor_num": 32,
    "num_layers": 3,
    "dropout": 0.0
}

AUTOREC_CONFIG: Dict[str, Any] = {
    "user_num": 6040,
    "item_num": 3706,
    "hidden_units": 500,
    "item_based": True
}

# ============================================================================
# ENCODING CONFIGURATION
# ============================================================================

# Encodings to try when loading data files (in order of preference)
SUPPORTED_ENCODINGS = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-8']

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_TITLE = "Recommendation System Inference API"
API_DESCRIPTION = "Neural Collaborative Filtering and AutoRec model inference API"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000
