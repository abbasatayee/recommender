"""
Model loading utilities for NCF and AutoRec models.
"""
import os
import torch
from typing import Optional
from autorec.utils.model import AutoRec
from ..config import AUTOREC_CONFIG, DEVICE


def load_ncf_model(model_path: str, device: str = DEVICE) -> Optional[torch.nn.Module]:
    """
    Load NCF model from checkpoint file.
    
    Args:
        model_path: Path to the saved NCF model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode, or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"⚠ Warning: NCF model not found at {model_path}")
        return None
    
    try:
        print(f"Loading NCF model from {model_path}...")
        model = torch.load(model_path, weights_only=False)
        model.eval()
        print(f"✓ NCF model loaded successfully on {device}")
        return model
    except Exception as e:
        print(f"✗ Error loading NCF model: {e}")
        return None


def load_autorec_model(model_path: str, device: str = DEVICE) -> Optional[torch.nn.Module]:
    """
    Load AutoRec model from checkpoint file.
    
    Args:
        model_path: Path to the saved AutoRec model checkpoint
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded model in evaluation mode, or None if loading fails
    """
    if not os.path.exists(model_path):
        print(f"⚠ Warning: AutoRec model not found at {model_path}")
        return None
    
    try:
        print(f"Loading AutoRec model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = AutoRec(
            num_users=AUTOREC_CONFIG["user_num"],
            num_items=AUTOREC_CONFIG["item_num"],
            num_hidden_units=AUTOREC_CONFIG["hidden_units"],
            item_based=AUTOREC_CONFIG["item_based"]
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        epoch = checkpoint.get('epoch', 'N/A')
        val_rmse = checkpoint.get('val_rmse', 'N/A')
        print(f"✓ AutoRec model loaded successfully on {device}")
        if isinstance(val_rmse, float):
            print(f"  Epoch: {epoch}, Val RMSE: {val_rmse:.6f}")
        else:
            print(f"  Epoch: {epoch}, Val RMSE: {val_rmse}")
        
        return model
    except Exception as e:
        print(f"✗ Error loading AutoRec model: {e}")
        return None
