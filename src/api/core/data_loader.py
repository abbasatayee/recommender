"""
Data loading utilities for training matrices and ratings.
"""
import os
import numpy as np
import pandas as pd
import torch
from typing import Optional
from ..config import (
    AUTOREC_CONFIG, RATINGS_FILE, TRAIN_DATA_PATH,
    SUPPORTED_ENCODINGS, DEVICE
)


def load_training_matrix(data_path: str = TRAIN_DATA_PATH) -> Optional[np.ndarray]:
    """
    Load training rating matrix for AutoRec inference.
    
    First tries to load from the original ratings.dat file (with ratings).
    Falls back to the training file (may only have user-item pairs).
    
    Args:
        data_path: Path to the training data file
    
    Returns:
        Training matrix as numpy array, or None if loading fails
    """
    # First, try to load from the original ratings.dat file
    if os.path.exists(RATINGS_FILE):
        ratings_df = _load_ratings_file()
        if ratings_df is not None:
            return _create_matrix_from_ratings(ratings_df)
    
    # Fallback: try loading from training file (may only have user-item pairs)
    if os.path.exists(data_path):
        return _load_training_file(data_path)
    
    print(f"✗ No training data file found")
    return None


def _load_ratings_file() -> Optional[pd.DataFrame]:
    """Load ratings from the original ratings.dat file."""
    for encoding in SUPPORTED_ENCODINGS:
        try:
            print(f"Loading ratings from original file: {RATINGS_FILE} (encoding: {encoding})...")
            ratings_df = pd.read_csv(
                RATINGS_FILE,
                sep='::',
                header=None,
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python',
                encoding=encoding,
                dtype={
                    'user_id': np.int32,
                    'item_id': np.int32,
                    'rating': np.float32,
                    'timestamp': np.int32
                }
            )
            return ratings_df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠ Error loading with {encoding}: {e}")
            continue
    
    print(f"⚠ Error loading from ratings.dat: Could not decode with any supported encoding")
    return None


def _create_matrix_from_ratings(ratings_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Create training matrix from ratings dataframe."""
    try:
        # Rename columns for consistency
        ratings_df = ratings_df.rename(columns={'user_id': 'user', 'item_id': 'item'})
        
        # Remap to 0-indexed if needed
        unique_users = sorted(ratings_df['user'].unique())
        unique_items = sorted(ratings_df['item'].unique())
        
        if unique_users != list(range(len(unique_users))):
            user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
            ratings_df['user'] = ratings_df['user'].map(user_map)
        
        if unique_items != list(range(len(unique_items))):
            item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
            ratings_df['item'] = ratings_df['item'].map(item_map)
        
        # Create training matrix
        train_mat = np.zeros(
            (AUTOREC_CONFIG["user_num"], AUTOREC_CONFIG["item_num"]),
            dtype=np.float32
        )
        
        for _, row in ratings_df.iterrows():
            user_id = int(row['user'])
            item_id = int(row['item'])
            rating = float(row['rating'])
            
            if (0 <= user_id < AUTOREC_CONFIG["user_num"] and
                0 <= item_id < AUTOREC_CONFIG["item_num"]):
                train_mat[user_id, item_id] = rating
        
        num_ratings = np.count_nonzero(train_mat)
        print(f"✓ Training matrix loaded: shape {train_mat.shape}, {num_ratings} ratings")
        return train_mat
    except Exception as e:
        print(f"⚠ Error processing ratings.dat: {e}")
        return None


def _load_training_file(data_path: str) -> Optional[np.ndarray]:
    """Load training file (may only have user-item pairs, no ratings)."""
    try:
        print(f"Loading training data from {data_path}...")
        train_data = pd.read_csv(
            data_path,
            sep='\t',
            header=None,
            names=['user', 'item'],
            usecols=[0, 1],
            dtype={0: np.int32, 1: np.int32}
        )
        
        train_mat = np.zeros(
            (AUTOREC_CONFIG["user_num"], AUTOREC_CONFIG["item_num"]),
            dtype=np.float32
        )
        
        # If no ratings, set to 1.0 (binary interaction)
        for _, row in train_data.iterrows():
            user_id = int(row['user'])
            item_id = int(row['item'])
            
            if (0 <= user_id < AUTOREC_CONFIG["user_num"] and
                0 <= item_id < AUTOREC_CONFIG["item_num"]):
                train_mat[user_id, item_id] = 1.0  # Default rating
        
        num_ratings = np.count_nonzero(train_mat)
        print(f"✓ Training matrix loaded (binary): shape {train_mat.shape}, {num_ratings} interactions")
        print(f"  ⚠ Warning: Using default rating of 1.0 (no ratings in file)")
        return train_mat
    except Exception as e:
        print(f"✗ Error loading training matrix: {e}")
        return None


def create_training_tensor(train_mat: Optional[np.ndarray]) -> Optional[torch.Tensor]:
    """
    Convert training matrix to PyTorch tensor on the appropriate device.
    
    Args:
        train_mat: Training matrix as numpy array
    
    Returns:
        Training matrix as PyTorch tensor, or None if input is None
    """
    if train_mat is None:
        return None
    return torch.Tensor(train_mat).to(DEVICE)
