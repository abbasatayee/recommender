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


import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from helpers import NCF


MODEL_PATH = "models/NeuMF-end.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters (from training)
# These should match the training configuration
USER_NUM = 6038  # Number of users in the dataset
ITEM_NUM = 3533  # Number of items in the dataset
FACTOR_NUM = 32  # Embedding dimension
NUM_LAYERS = 3   # Number of MLP layers
DROPOUT = 0.0    # Dropout rate
MODEL_NAME = "NeuMF-end"

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


# Extract embeddings as NumPy arrays
user_embeddings = model.embed_user_GMF.weight.detach().cpu().numpy()
item_embeddings = model.embed_item_GMF.weight.detach().cpu().numpy()

pca = PCA(n_components=2)
user_emb_2d = pca.fit_transform(user_embeddings)
item_emb_2d = pca.fit_transform(item_embeddings)


tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

item_emb_2d = tsne.fit_transform(item_embeddings)


# plt.figure(figsize=(8, 6))
# plt.scatter(item_emb_2d[:, 0], item_emb_2d[:, 1], s=5, alpha=0.6)
# plt.title("Item Embeddings (2D Projection)")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.show()



# plt.figure(figsize=(8, 6))
# plt.scatter(user_emb_2d[:, 0], user_emb_2d[:, 1], s=5, alpha=0.6)
# plt.title("User Embeddings (2D Projection)")
# plt.xlabel("Component 1")
# plt.ylabel("Component 2")
# plt.show()


user_id = 42  # pick any user

user_vec = user_embeddings[user_id]
print(f"User vector: {user_vec}")
scores = item_embeddings @ user_vec  # dot product similarity
print(f"Scores: {scores}")
top_items = np.argsort(scores)[-50:]
print(f"Top 50 items for user {user_id}: {top_items}")


plt.figure(figsize=(8, 6))
plt.scatter(item_emb_2d[:, 0], item_emb_2d[:, 1], s=3, alpha=0.2)
plt.scatter(
    item_emb_2d[top_items, 0],
    item_emb_2d[top_items, 1],
    s=30,
    label="Top Items"
)
plt.scatter(
    user_emb_2d[user_id, 0],
    user_emb_2d[user_id, 1],
    s=100,
    marker="X",
    label="User"
)
plt.legend()
plt.title(f"User {user_id} and Preferred Items")
plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 6: PCA")
    print("=" * 70)
    print("✓ PCA class defined")
    print("=" * 70)
    print("STEP 7: PCA")
    print("=" * 70)
    print("✓ PCA class defined")
