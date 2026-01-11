"""
Service layer for managing models and inference engines.
"""
import torch
from typing import Optional, Tuple
from .model_loader import load_ncf_model, load_autorec_model
from .data_loader import load_training_matrix, create_training_tensor
from .inference_engine import NCFInferenceEngine, AutoRecInferenceEngine
from .movie_metadata import get_metadata_manager
from ..config import (
    NCF_MODEL_PATH, AUTOREC_MODEL_PATH, TRAIN_DATA_PATH, DEVICE
)


# Global instances
_ncf_model: Optional[torch.nn.Module] = None
_autorec_model: Optional[torch.nn.Module] = None
_autorec_train_tensor: Optional[torch.Tensor] = None
_ncf_engine: Optional[NCFInferenceEngine] = None
_autorec_engine: Optional[AutoRecInferenceEngine] = None


def initialize_models() -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[torch.Tensor]]:
    """
    Initialize all models and data.
    
    Returns:
        Tuple of (ncf_model, autorec_model, autorec_train_tensor)
    """
    global _ncf_model, _autorec_model, _autorec_train_tensor
    
    print("=" * 70)
    print("Initializing Models")
    print("=" * 70)
    
    # Load models
    _ncf_model = load_ncf_model(NCF_MODEL_PATH, DEVICE)
    _autorec_model = load_autorec_model(AUTOREC_MODEL_PATH, DEVICE)
    
    # Load training data for AutoRec
    autorec_train_mat = load_training_matrix(TRAIN_DATA_PATH)
    _autorec_train_tensor = create_training_tensor(autorec_train_mat)
    
    # Initialize movie metadata
    get_metadata_manager()
    
    print("=" * 70)
    
    return _ncf_model, _autorec_model, _autorec_train_tensor


def get_ncf_model() -> Optional[torch.nn.Module]:
    """Get the NCF model instance."""
    global _ncf_model
    if _ncf_model is None:
        initialize_models()
    return _ncf_model


def get_autorec_model() -> Optional[torch.nn.Module]:
    """Get the AutoRec model instance."""
    global _autorec_model
    if _autorec_model is None:
        initialize_models()
    return _autorec_model


def get_ncf_engine() -> NCFInferenceEngine:
    """Get the NCF inference engine."""
    global _ncf_engine
    if _ncf_engine is None:
        _ncf_engine = NCFInferenceEngine(get_ncf_model())
    return _ncf_engine


def get_autorec_engine() -> AutoRecInferenceEngine:
    """Get the AutoRec inference engine."""
    global _autorec_engine, _autorec_train_tensor
    if _autorec_engine is None:
        if _autorec_train_tensor is None:
            initialize_models()
        _autorec_engine = AutoRecInferenceEngine(_autorec_model, _autorec_train_tensor)
    return _autorec_engine
