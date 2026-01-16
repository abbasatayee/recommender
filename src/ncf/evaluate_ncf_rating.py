"""
Evaluation script for NCF Rating Prediction models.
Loads pre-trained models and creates visualization comparing actual vs predicted rating distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

from helpers import download_ml1m_dataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Paths - use script location to find project root
# Script is at: src/ncf/evaluate_ncf_rating.py
# Project root is: src/../ (one level up from src)
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # project root
DATA_DIR = os.path.join(project_root, 'data')
MODEL_PATH = os.path.join(project_root, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Debug: print paths
print(f"\nPaths:")
print(f"  Script location: {current_dir}")
print(f"  Project root: {project_root}")
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  MODEL_PATH: {MODEL_PATH}")

# Hyperparameters (must match training)
FACTOR_NUM = 32
NUM_LAYERS = 3
DROPOUT_RATE = 0.2
RATING_MIN = 1.0
RATING_MAX = 5.0
TEST_RATIO = 0.2
BATCH_SIZE = 256

# Model class (same as in training notebook)
class NCFRating(nn.Module):
    """NCF model adapted for rating prediction."""
    
    def __init__(self, num_users, num_items, factor_num, num_layers,
                 dropout, model_name='NeuMF-end', rating_min=1.0, rating_max=5.0,
                 GMF_model=None, MLP_model=None):
        super(NCFRating, self).__init__()
        
        self.model_name = model_name
        self.dropout = dropout
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model
        
        # GMF Embeddings
        if model_name != 'MLP':
            self.embed_user_GMF = nn.Embedding(num_users, factor_num)
            self.embed_item_GMF = nn.Embedding(num_items, factor_num)
        
        # MLP Embeddings
        if model_name != 'GMF':
            mlp_embed_dim = factor_num * (2 ** (num_layers - 1))
            self.embed_user_MLP = nn.Embedding(num_users, mlp_embed_dim)
            self.embed_item_MLP = nn.Embedding(num_items, mlp_embed_dim)
            
            # MLP Layers
            layers = []
            for i in range(num_layers):
                input_size = factor_num * (2 ** (num_layers - i))
                layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(input_size, input_size // 2))
                layers.append(nn.ReLU())
            self.MLP_layers = nn.Sequential(*layers)
        
        # Prediction Layer
        if model_name in ['MLP', 'GMF']:
            predict_size = factor_num
        else:
            predict_size = factor_num * 2
        
        self.predict_layer = nn.Linear(predict_size, 1)
        self._init_weights()
    
    def _init_weights(self):
        if self.model_name != 'NeuMF-pre':
            # Random initialization
            for name, param in self.named_parameters():
                if 'embed' in name:
                    nn.init.normal_(param, std=0.01)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, user, item):
        # GMF path
        if self.model_name != 'MLP':
            user_gmf = self.embed_user_GMF(user)
            item_gmf = self.embed_item_GMF(item)
            output_gmf = user_gmf * item_gmf
        
        # MLP path
        if self.model_name != 'GMF':
            user_mlp = self.embed_user_MLP(user)
            item_mlp = self.embed_item_MLP(item)
            interaction = torch.cat([user_mlp, item_mlp], dim=-1)
            output_mlp = self.MLP_layers(interaction)
        
        # Combine
        if self.model_name == 'GMF':
            concat = output_gmf
        elif self.model_name == 'MLP':
            concat = output_mlp
        else:
            concat = torch.cat([output_gmf, output_mlp], dim=-1)
        
        # Predict and scale to rating range
        logits = self.predict_layer(concat).view(-1)
        rating_range = self.rating_max - self.rating_min
        return torch.sigmoid(logits) * rating_range + self.rating_min


class RatingDataset(data.Dataset):
    """Dataset for explicit rating prediction."""
    
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_idx'].values)
        self.items = torch.LongTensor(df['item_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_data():
    """Load and preprocess data."""
    ratings_file = download_ml1m_dataset(DATA_DIR)
    
    ratings_df = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    
    # Re-index
    user_ids = ratings_df['user_id'].unique()
    item_ids = ratings_df['item_id'].unique()
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    ratings_df['user_idx'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['item_idx'] = ratings_df['item_id'].map(item_to_idx)
    
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    # Train/Test split
    train_df, test_df = train_test_split(ratings_df, test_size=TEST_RATIO, random_state=42)
    
    return train_df, test_df, num_users, num_items


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for user, item, rating in test_loader:
            user, item = user.to(device), item.to(device)
            pred = model(user, item)
            preds.extend(pred.cpu().numpy())
            actuals.extend(rating.numpy())
    
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
    print("NCF Rating Prediction - Model Evaluation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_df, test_df, num_users, num_items = load_data()
    print(f"Users: {num_users}, Items: {num_items}")
    print(f"Test set: {len(test_df):,} ratings")
    
    # Create test dataset
    test_dataset = RatingDataset(test_df)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Models to evaluate
    models_to_eval = [
        ('GMF', 'GMF-Rating.pth'),
        ('MLP', 'MLP-Rating.pth'),
        ('NeuMF-end', 'NeuMF-end-Rating.pth'),
        ('NeuMF-pre', 'NeuMF-pre-Rating.pth')
    ]
    
    results = {}
    
    for model_name, model_file in models_to_eval:
        model_path = os.path.join(MODEL_PATH, model_file)
        
        if not os.path.exists(model_path):
            print(f"\n⚠ Warning: Model file not found: {model_path}")
            print(f"  Skipping {model_name} evaluation")
            continue
        
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name}")
        print(f"{'='*70}")
        
        # Special handling for NeuMF-pre: need to load GMF and MLP first
        gmf_model = None
        mlp_model = None
        if model_name == 'NeuMF-pre':
            # Load GMF model
            gmf_path = os.path.join(MODEL_PATH, 'GMF-Rating.pth')
            if os.path.exists(gmf_path):
                gmf_model = NCFRating(
                    num_users=num_users, num_items=num_items,
                    factor_num=FACTOR_NUM, num_layers=NUM_LAYERS,
                    dropout=DROPOUT_RATE, model_name='GMF',
                    rating_min=RATING_MIN, rating_max=RATING_MAX
                ).to(device)
                gmf_model.load_state_dict(torch.load(gmf_path, map_location=device))
                print(f"✓ GMF model loaded for NeuMF-pre initialization")
            else:
                print(f"⚠ Warning: GMF model not found at {gmf_path}")
                print(f"  Skipping {model_name} evaluation")
                continue
            
            # Load MLP model
            mlp_path = os.path.join(MODEL_PATH, 'MLP-Rating.pth')
            if os.path.exists(mlp_path):
                mlp_model = NCFRating(
                    num_users=num_users, num_items=num_items,
                    factor_num=FACTOR_NUM, num_layers=NUM_LAYERS,
                    dropout=DROPOUT_RATE, model_name='MLP',
                    rating_min=RATING_MIN, rating_max=RATING_MAX
                ).to(device)
                mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
                print(f"✓ MLP model loaded for NeuMF-pre initialization")
            else:
                print(f"⚠ Warning: MLP model not found at {mlp_path}")
                print(f"  Skipping {model_name} evaluation")
                continue
        
        # Initialize model
        model = NCFRating(
            num_users=num_users, num_items=num_items,
            factor_num=FACTOR_NUM, num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE, model_name=model_name,
            rating_min=RATING_MIN, rating_max=RATING_MAX,
            GMF_model=gmf_model, MLP_model=mlp_model
        ).to(device)
        
        # Load weights
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # For NeuMF-pre, filter out nested model keys if present
            if model_name == 'NeuMF-pre':
                # Remove keys that start with "GMF_model." or "MLP_model."
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() 
                    if not k.startswith('GMF_model.') and not k.startswith('MLP_model.')
                }
                state_dict = filtered_state_dict
            
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Model loaded from: {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            continue
        
        # Evaluate
        preds, actuals, rmse, mae = evaluate_model(model, test_loader, device)
        
        print(f"\nResults:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        results[model_name] = {
            'preds': preds,
            'actuals': actuals,
            'rmse': rmse,
            'mae': mae
        }
        
        # Create visualization
        save_path = os.path.join(MODEL_PATH, f'{model_name}_rating_distribution.png')
        plot_rating_distributions(actuals, preds, model_name, save_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['rmse']:>10.4f} {result['mae']:>10.4f}")
    print("=" * 70)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
