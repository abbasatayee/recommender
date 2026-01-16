"""
Evaluation script for HybridAutoRecNCF Rating Prediction model.
Loads pre-trained model and creates visualization comparing actual vs predicted rating distributions.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from helpers.data_downloader import download_ml1m_dataset

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Paths - use script location to find project root
# Script is at: src/hybridautorecncf/evaluate_hybridautorecncf_rating.py
# Project root is: src/../ (one level up from src)
src_dir = os.path.dirname(current_dir)  # src/
project_root = os.path.dirname(src_dir)  # project root
DATA_DIR = os.path.join(project_root, 'data')
MODEL_PATH = os.path.join(project_root, 'models')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Hyperparameters (must match training)
LATENT_DIM = 64
MLP_LAYERS = [128, 64]
HIDDEN_DIMS = (256, 128)  # AutoEncoder hidden dimensions
DROPOUT_RATE = 0.1
BATCH_SIZE = 256


# Model classes (same as in training notebook)
class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims=(256, 128),
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            input_dim: dimensionality of input vector (e.g. user rating vector)
            latent_dim: size of bottleneck (MUST match NCF latent_dim)
            hidden_dims: encoder hidden layer sizes
            dropout_rate: dropout applied to hidden layers
        """
        super().__init__()

        # =======================
        # Encoder
        # =======================
        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        # Bottleneck (NO activation)
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # =======================
        # Decoder
        # =======================
        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        # No activation for rating prediction (raw reconstruction)

        self.decoder = nn.Sequential(*decoder_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


class NCF(nn.Module):
    def __init__(
        self,
        latent_dim,
        mlp_layers=(128, 64),
        dropout_rate=0.1,
    ):
        super().__init__()

        # MLP
        mlp_modules = []
        input_dim = latent_dim * 2

        for h in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, h))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout_rate))
            input_dim = h

        self.mlp = nn.Sequential(*mlp_modules)

        # Final prediction layer
        mlp_out_dim = mlp_layers[-1] if len(mlp_layers) > 0 else input_dim
        self.output = nn.Linear(latent_dim + mlp_out_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_z, item_z):
        # GMF branch
        gmf_out = user_z * item_z  # pure element-wise product

        # MLP branch
        mlp_input = torch.cat([user_z, item_z], dim=1)
        mlp_out = self.mlp(mlp_input)

        # Final prediction
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        pred = self.output(concat)

        # Return raw prediction (use MSE loss, no sigmoid scaling)
        return pred.squeeze(-1)


class HybridAutoRecNCF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, mlp_layers, 
                 hidden_dims=(256, 128), dropout_rate=0.1):
        super().__init__()

        self.user_autorec = AutoEncoder(num_items, latent_dim, hidden_dims, dropout_rate)
        self.item_autorec = AutoEncoder(num_users, latent_dim, hidden_dims, dropout_rate)

        self.ncf = NCF(latent_dim, mlp_layers, dropout_rate)

    def forward(self, user_vecs, item_vecs, user_ids, item_ids):
        # AutoRec forward
        # user_vecs: (batch_size, num_items) - each row is a user's rating vector
        # item_vecs: (batch_size, num_users) - each row is an item's rating vector
        user_recon, user_z = self.user_autorec(user_vecs)
        item_recon, item_z = self.item_autorec(item_vecs)

        # Each element in the batch corresponds to a (user, item) pair
        # user_z[i] is the latent for the user in pair i
        # item_z[i] is the latent for the item in pair i
        pred = self.ncf(user_z, item_z)
        return pred, user_recon, item_recon


class HybridDataset(data.Dataset):
    def __init__(self, rating_matrix, interactions_df):
        """
        Args:
            rating_matrix: (num_users, num_items) rating matrix
            interactions_df: DataFrame with columns ['user_id', 'item_id', 'rating']
        """
        self.rating_matrix = rating_matrix
        self.interactions_df = interactions_df.reset_index(drop=True)
        
        # Create masks (1 where rating exists, 0 otherwise)
        self.user_mask = (rating_matrix > 0).astype(np.float32)
        self.item_mask = (rating_matrix.T > 0).astype(np.float32)  # Transpose for items
    
    def __len__(self):
        return len(self.interactions_df)
    
    def __getitem__(self, idx):
        row = self.interactions_df.iloc[idx]
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        rating = float(row['rating'])
        
        # Get user vector (ratings across all items)
        user_vec = torch.FloatTensor(self.rating_matrix[user_id])
        
        # Get item vector (ratings across all users)
        item_vec = torch.FloatTensor(self.rating_matrix[:, item_id])
        
        # Get masks
        user_mask = torch.FloatTensor(self.user_mask[user_id])
        item_mask = torch.FloatTensor(self.item_mask[item_id])
        
        return (
            torch.LongTensor([user_id]),
            torch.LongTensor([item_id]),
            torch.FloatTensor([rating]),
            user_vec,
            item_vec,
            user_mask,
            item_mask
        )


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
    
    # Preprocess: remap IDs and create train/test split
    print("\nPreprocessing data...")
    print("=" * 70)
    
    # Remap user and item IDs to be contiguous
    unique_users = sorted(ratings_df['user_id'].unique())
    unique_items = sorted(ratings_df['item_id'].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
    
    ratings_df['user_id'] = ratings_df['user_id'].map(user_to_idx)
    ratings_df['item_id'] = ratings_df['item_id'].map(item_to_idx)
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    # Create train/test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        ratings_df[['user_id', 'item_id', 'rating']],
        test_size=0.2,
        random_state=42
    )
    
    # Create rating matrices
    train_mat = np.zeros((num_users, num_items), dtype=np.float32)
    for _, row in train_df.iterrows():
        train_mat[int(row['user_id']), int(row['item_id'])] = row['rating']
    
    test_mat = np.zeros((num_users, num_items), dtype=np.float32)
    for _, row in test_df.iterrows():
        test_mat[int(row['user_id']), int(row['item_id'])] = row['rating']
    
    # IMPORTANT: Use train_mat for test dataset to prevent data leakage
    # User/item vectors should only contain training data, not test ratings
    print(f"✓ Preprocessing complete!")
    print(f"  - Users: {num_users}, Items: {num_items}")
    print(f"  - Train interactions: {len(train_df):,}")
    print(f"  - Test interactions: {len(test_df):,}")
    print(f"  - Using train_mat for test dataset (no data leakage)")
    print("=" * 70)
    
    return train_mat, train_df, test_df, num_users, num_items


def evaluate_model(model, test_loader, device):
    """Evaluate model and return predictions and metrics."""
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            (user_ids, item_ids, ratings,
             user_vecs, item_vecs,
             user_mask, item_mask) = batch

            user_ids = user_ids.to(device).squeeze()
            item_ids = item_ids.to(device).squeeze()
            ratings = ratings.to(device).squeeze()
            user_vecs = user_vecs.to(device)
            item_vecs = item_vecs.to(device)
            user_mask = user_mask.to(device)
            item_mask = item_mask.to(device)

            pred, user_recon, item_recon = model(
                user_vecs, item_vecs, user_ids, item_ids
            )
            
            # Use raw predictions (no clamping) to match training notebook's evaluate function
            preds.extend(pred.cpu().numpy())
            actuals.extend(ratings.cpu().numpy())
    
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
    print("HybridAutoRecNCF Rating Prediction - Model Evaluation")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_mat, train_df, test_df, num_users, num_items = load_data()
    
    print(f"\nTest set: {len(test_df):,} ratings")
    
    # Create test dataset using train_mat (proper evaluation - no data leakage)
    # User/item vectors are constructed from training data only
    test_dataset = HybridDataset(train_mat, test_df)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # Model path
    model_path = os.path.join(MODEL_PATH, 'HybridAutoRecNCF-Rating.pth')
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Error: Model file not found: {model_path}")
        return
    
    print(f"\n{'='*70}")
    print("Evaluating HybridAutoRecNCF")
    print(f"{'='*70}")
    
    # Initialize model (must match training architecture exactly)
    model = HybridAutoRecNCF(
        num_users=num_users,
        num_items=num_items,
        latent_dim=LATENT_DIM,
        mlp_layers=MLP_LAYERS,
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Evaluate
    print("\nEvaluating on test set...")
    preds, actuals, rmse, mae = evaluate_model(model, test_loader, device)
    
    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    
    # Create visualization
    save_path = os.path.join(MODEL_PATH, 'HybridAutoRecNCF_rating_distribution.png')
    plot_rating_distributions(actuals, preds, 'HybridAutoRecNCF', save_path)
    
    print(f"\n{'='*70}")
    print("Evaluation Summary")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'RMSE':>10} {'MAE':>10}")
    print("-" * 70)
    print(f"{'HybridAutoRecNCF':<20} {rmse:>10.4f} {mae:>10.4f}")
    print("=" * 70)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
