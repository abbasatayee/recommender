"""
Evaluation script for AutoRec Rating Prediction model.
Loads pre-trained model and creates visualization comparing actual vs predicted rating distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt

# Add src to path
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
path = current_dir
while True:
    if os.path.basename(path) == "src":
        if path not in sys.path:
            sys.path.insert(0, path)
        break
    parent = os.path.dirname(path)
    if parent == path:
        break
    path = parent

from utils.model import AutoRec
from utils.autorecdata import AutoRecData
from utils.preprocessor import PreProcessor
from helpers.data_downloader import download_ml1m_dataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Paths - use script location to find project root
# Script is at: src/autorec/evaluate_autorec_rating.py
# Project root is: src/../ (one level up from src)
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # project root
DATA_DIR = os.path.join(project_root, 'data')
MODEL_PATH = os.path.join(project_root, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Hyperparameters (must match training)
NUM_HIDDEN_UNITS = 500
ITEM_BASED = True


def load_data():
    """Load and preprocess data."""
    data_path = os.path.join(DATA_DIR, 'ml-1m', 'ratings.dat')
    
    if not os.path.exists(data_path):
        download_ml1m_dataset(data_dir=DATA_DIR)
    
    def load_ml_1m_data(data_path=data_path) -> pd.DataFrame:
        print("=" * 70)
        print("Loading MovieLens 1M Dataset")
        print("=" * 70)
        print(f"Data path: {data_path}")
        return pd.read_csv(
            data_path,
            sep='::',
            header=None,
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python',
            dtype={
                'user_id': np.int32,
                'item_id': np.int32,
                'rating': np.float32,
                'timestamp': np.int32
            }
        )
    
    print("\nLoading ratings data...")
    ratings_df = load_ml_1m_data()
    print(f"✓ Successfully loaded {len(ratings_df):,} ratings")
    print("=" * 70)
    
    # Preprocess data
    preprocessor = PreProcessor()
    print("\nSplitting data into train/test sets and creating rating matrices...")
    print("=" * 70)
    train_mat, test_mat, num_users, num_items = preprocessor.preprocess_ml1m_data(
        ratings_df,
        test_size=0.2,
        random_state=42
    )
    
    print(f"✓ Data preprocessing complete!")
    print(f"Train matrix shape: {train_mat.shape}, Test matrix shape: {test_mat.shape}")
    print("=" * 70)
    
    return train_mat, test_mat, num_users, num_items


def evaluate_model(model, train_mat, test_mat, device):
    """Evaluate AutoRec model and return predictions and metrics."""
    model.eval()
    
    # Convert to tensors
    train_mat_tensor = torch.FloatTensor(train_mat).to(device)
    
    # Get all test ratings
    test_ratings = []
    for user in range(test_mat.shape[0]):
        for item in range(test_mat.shape[1]):
            if test_mat[user, item] > 0:
                test_ratings.append((user, item, test_mat[user, item]))
    
    preds = []
    actuals = []
    
    with torch.no_grad():
        for user, item, actual_rating in test_ratings:
            # For item-based AutoRec, get item vector from training matrix
            # Item vector is the column (all users' ratings for this item)
            item_vec = train_mat_tensor[:, item].unsqueeze(0).to(device)
            
            # Reconstruct the item vector
            reconstructed = model(item_vec)
            # Clamp to valid rating range
            reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
            
            # Get user's predicted rating from reconstructed vector
            pred_rating = reconstructed[0, user].item()
            
            preds.append(pred_rating)
            actuals.append(actual_rating)
    
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mae = np.mean(np.abs(preds - actuals))
    
    return preds, actuals, rmse, mae


def plot_rating_distributions(actual_ratings, predicted_ratings, model_name, save_path):
    """Create visualization comparing actual vs predicted rating distributions."""
    # Larger figure for better visibility
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Actual Rating Distribution (bar chart)
    unique_ratings, counts = np.unique(actual_ratings, return_counts=True)
    bars1 = ax1.bar(unique_ratings, counts, color='#2E86AB', alpha=0.8, 
                    edgecolor='black', linewidth=1.5, width=0.6)
    ax1.set_xlabel('Rating', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax1.set_title('Actual Rating Distribution', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.set_xticklabels([1, 2, 3, 4, 5], fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_axisbelow(True)
    
    # Format y-axis with thousands separator
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add count labels on bars with better positioning
    for rating, count in zip(unique_ratings, counts):
        ax1.text(rating, count, f'{count:,}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Predicted Rating Distribution (histogram)
    # Use 8 bins for clear, interpretable distribution (each bin ~0.5 rating units)
    n, bins, patches = ax2.hist(predicted_ratings, bins=8, color='#F24236', 
                                 alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Predicted Rating', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax2.set_title('Predicted Rating Distribution', fontsize=15, fontweight='bold', pad=15)
    ax2.set_xlim([1, 5])
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticklabels([1, 2, 3, 4, 5], fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_axisbelow(True)
    
    # Format y-axis with thousands separator
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Add statistics text with better formatting
    rmse = np.sqrt(np.mean((predicted_ratings - actual_ratings) ** 2))
    mae = np.mean(np.abs(predicted_ratings - actual_ratings))
    stats_text = f'RMSE: {rmse:.4f}\nMAE:  {mae:.4f}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='#FFE66D', alpha=0.8, 
                      edgecolor='black', linewidth=1.5),
             fontsize=11, family='monospace', fontweight='bold')
    
    # Main title
    plt.suptitle(f'{model_name} - Rating Distribution Comparison', 
                 fontsize=17, fontweight='bold', y=1.02)
    
    # Better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {save_path}")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 70)
    print("AutoRec Rating Prediction - Model Evaluation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_mat, test_mat, num_users, num_items = load_data()
    
    # Count test ratings
    num_test_ratings = np.count_nonzero(test_mat)
    print(f"\nTest set: {num_test_ratings:,} ratings")
    
    # Model path
    model_path = os.path.join(MODEL_PATH, 'AutoRec-best.pth')
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Error: Model file not found: {model_path}")
        return
    
    print(f"\n{'='*70}")
    print("Evaluating AutoRec")
    print(f"{'='*70}")
    
    # Initialize model
    model = AutoRec(
        num_users=num_users,
        num_items=num_items,
        num_hidden_units=NUM_HIDDEN_UNITS,
        item_based=ITEM_BASED
    ).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from: {model_path}")
        
        epoch = checkpoint.get('epoch', 'N/A')
        val_rmse = checkpoint.get('val_rmse', 'N/A')
        if isinstance(val_rmse, float):
            print(f"  Training epoch: {epoch}, Validation RMSE: {val_rmse:.6f}")
        else:
            print(f"  Training epoch: {epoch}, Validation RMSE: {val_rmse}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Evaluate
    print("\nEvaluating on test set...")
    preds, actuals, rmse, mae = evaluate_model(model, train_mat, test_mat, device)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    
    # Create visualization
    save_path = os.path.join(MODEL_PATH, 'AutoRec_rating_distribution.png')
    plot_rating_distributions(actuals, preds, 'AutoRec', save_path)
    
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)
    print(f"{'AutoRec':<15} {rmse:>10.4f} {mae:>10.4f}")
    print("=" * 70)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
