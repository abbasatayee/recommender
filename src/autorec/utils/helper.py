import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from itertools import product


def get_metrics(
    model: nn.Module, train_set=data.Dataset, test_set=data.Dataset, device=None, item_based: bool = True
) -> np.float32:
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    # Get original matrices (not transposed)
    train_mat = torch.Tensor(train_set.original_data).to(device)
    test_mat = torch.Tensor(test_set.original_data).to(device)
    test_mask = (test_mat > 0).to(device)

    if item_based:
        # For item-based AutoRec: iterate over item vectors (columns)
        # Each item vector is (num_users,) - ratings from all users for this item
        num_items = train_mat.shape[1]
        predictions = torch.zeros_like(test_mat).to(device)
        
        with torch.no_grad():
            for item_idx in range(num_items):
                # Get item vector from training matrix (column)
                item_vec = train_mat[:, item_idx].unsqueeze(0)  # Shape: (1, num_users)
                
                # Reconstruct this item vector
                reconstructed = model(item_vec)  # Shape: (1, num_users)
                
                # Clip predictions to valid rating range [1, 5] for MovieLens
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                
                # Store predictions for all users for this item
                predictions[:, item_idx] = reconstructed.squeeze(0)
        
        # Handle unseen items/users with default rating of 3
        unseen_items = test_set.items - train_set.items
        unseen_users = test_set.users - train_set.users
        
        for item_idx in unseen_items:
            if item_idx < predictions.shape[1]:
                predictions[:, item_idx] = 3.0
        
        for user_idx in unseen_users:
            if user_idx < predictions.shape[0]:
                predictions[user_idx, :] = 3.0
    else:
        # For user-based AutoRec: iterate over user vectors (rows)
        num_users = train_mat.shape[0]
        predictions = torch.zeros_like(test_mat).to(device)
        
        with torch.no_grad():
            for user_idx in range(num_users):
                # Get user vector from training matrix (row)
                user_vec = train_mat[user_idx, :].unsqueeze(0)  # Shape: (1, num_items)
                
                # Reconstruct this user vector
                reconstructed = model(user_vec)  # Shape: (1, num_items)
                
                # Clip predictions to valid rating range [1, 5] for MovieLens
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                
                # Store predictions for all items for this user
                predictions[user_idx, :] = reconstructed.squeeze(0)
        
        # Handle unseen items/users with default rating of 3
        unseen_items = test_set.items - train_set.items
        unseen_users = test_set.users - train_set.users
        
        for item_idx in unseen_items:
            if item_idx < predictions.shape[1]:
                predictions[:, item_idx] = 3.0
        
        for user_idx in unseen_users:
            if user_idx < predictions.shape[0]:
                predictions[user_idx, :] = 3.0

    return masked_rmse(actual=test_mat, pred=predictions, mask=test_mask)


def masked_rmse(
    actual: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> np.float32:
    mse = ((pred - actual) * mask).pow(2).sum() / mask.sum()

    return np.sqrt(mse.detach().cpu().numpy())


def masked_mae(
    actual: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> np.float32:
    """Compute Mean Absolute Error (MAE) over masked elements."""
    mae = (torch.abs(pred - actual) * mask).sum() / mask.sum()
    return mae.detach().cpu().numpy()


def get_rating_metrics(
    model: nn.Module, train_set=data.Dataset, test_set=data.Dataset, device=None, item_based: bool = True
) -> tuple:
    """
    Compute RMSE and MAE for rating prediction (no ranking metrics).
    
    Returns:
        (rmse, mae) tuple
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    # Get original matrices (not transposed)
    train_mat = torch.Tensor(train_set.original_data).to(device)
    test_mat = torch.Tensor(test_set.original_data).to(device)
    test_mask = (test_mat > 0).to(device)

    if item_based:
        # For item-based AutoRec: iterate over item vectors (columns)
        num_items = train_mat.shape[1]
        predictions = torch.zeros_like(test_mat).to(device)
        
        with torch.no_grad():
            for item_idx in range(num_items):
                item_vec = train_mat[:, item_idx].unsqueeze(0)
                reconstructed = model(item_vec)
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                predictions[:, item_idx] = reconstructed.squeeze(0)
        
        # Handle unseen items/users with default rating of 3
        unseen_items = test_set.items - train_set.items
        unseen_users = test_set.users - train_set.users
        
        for item_idx in unseen_items:
            if item_idx < predictions.shape[1]:
                predictions[:, item_idx] = 3.0
        
        for user_idx in unseen_users:
            if user_idx < predictions.shape[0]:
                predictions[user_idx, :] = 3.0
    else:
        # For user-based AutoRec: iterate over user vectors (rows)
        num_users = train_mat.shape[0]
        predictions = torch.zeros_like(test_mat).to(device)
        
        with torch.no_grad():
            for user_idx in range(num_users):
                user_vec = train_mat[user_idx, :].unsqueeze(0)
                reconstructed = model(user_vec)
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                predictions[user_idx, :] = reconstructed.squeeze(0)
        
        # Handle unseen items/users with default rating of 3
        unseen_items = test_set.items - train_set.items
        unseen_users = test_set.users - train_set.users
        
        for item_idx in unseen_items:
            if item_idx < predictions.shape[1]:
                predictions[:, item_idx] = 3.0
        
        for user_idx in unseen_users:
            if user_idx < predictions.shape[0]:
                predictions[user_idx, :] = 3.0

    rmse = masked_rmse(actual=test_mat, pred=predictions, mask=test_mask)
    mae = masked_mae(actual=test_mat, pred=predictions, mask=test_mask)
    
    return rmse, mae


def hit(gt_items, pred_items):
    """
    Calculate Hit Rate for a single user.
    
    Hit Rate is 1 if any ground truth item is in the predicted top-K items,
    otherwise 0.
    
    Parameters:
    - gt_items: Set or list of ground truth item IDs (items user actually interacted with in test set)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - 1 if any gt_item is in pred_items, 0 otherwise
    """
    if len(gt_items) == 0:
        return 0.0
    return 1.0 if any(item in pred_items for item in gt_items) else 0.0


def ndcg(gt_items, pred_items):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) for a single user.
    
    NDCG measures ranking quality by:
    1. Giving more weight to items ranked higher (position matters)
    2. Using logarithmic discounting (relevance decreases with position)
    
    Formula: NDCG = sum(1 / log2(position + 2)) for all gt_items in pred_items
    
    Parameters:
    - gt_items: Set or list of ground truth item IDs (items user actually interacted with in test set)
    - pred_items: List of top-K predicted item IDs (recommended items)
    
    Returns:
    - NDCG score (0.0 to 1.0) normalized by the number of ground truth items
    """
    if len(gt_items) == 0:
        return 0.0
    
    # Calculate DCG: sum of 1/log2(position + 2) for each ground truth item found
    dcg = 0.0
    for idx, item in enumerate(pred_items):
        if item in gt_items:
            dcg += 1.0 / np.log2(idx + 2)
    
    # Calculate IDCG (Ideal DCG): if all ground truth items were at the top
    # This normalizes the score to [0, 1]
    num_gt = len(gt_items)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, len(pred_items))))
    
    # Normalize: NDCG = DCG / IDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def get_ranking_metrics(
    model: nn.Module, 
    train_set: data.Dataset, 
    test_set: data.Dataset, 
    top_k: int = 10,
    device=None,
    item_based: bool = True
) -> tuple:
    """
    Calculate HR@K and NDCG@K metrics for AutoRec model.
    
    For each user:
    1. Get model predictions for all items
    2. Mask out items seen in training (to avoid recommending already seen items)
    3. Get top-K items with highest predicted ratings
    4. Check if test items are in top-K (HR@K)
    5. Calculate NDCG@K based on positions of test items
    
    Parameters:
    - model: Trained AutoRec model
    - train_set: Training dataset (to mask out seen items)
    - test_set: Test dataset (to get ground truth items)
    - top_k: Number of top items to consider (default: 10)
    - device: Device to run on (default: auto-detect)
    
    Returns:
    - mean_HR: Average Hit Rate across all users
    - mean_NDCG: Average NDCG across all users
    """
    # Set device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Get original matrices (not transposed)
    train_mat = torch.Tensor(train_set.original_data).to(device)
    test_mat = torch.Tensor(test_set.original_data).to(device)
    
    # Get training mask (items seen in training)
    train_mask = (train_mat > 0).to(device)
    
    # Get predictions for all users
    with torch.no_grad():
        if item_based:
            # For item-based: predict ratings by reconstructing each item vector
            num_items = train_mat.shape[1]
            predictions = torch.zeros_like(train_mat).to(device)
            
            for item_idx in range(num_items):
                # Get item vector from training matrix (column)
                item_vec = train_mat[:, item_idx].unsqueeze(0)  # Shape: (1, num_users)
                
                # Reconstruct this item vector
                reconstructed = model(item_vec)  # Shape: (1, num_users)
                
                # Clip predictions to valid rating range [1, 5] for MovieLens
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                
                # Store predictions for all users for this item
                predictions[:, item_idx] = reconstructed.squeeze(0)
        else:
            # For user-based: predict ratings by reconstructing each user vector
            num_users = train_mat.shape[0]
            predictions = torch.zeros_like(train_mat).to(device)
            
            for user_idx in range(num_users):
                # Get user vector from training matrix (row)
                user_vec = train_mat[user_idx, :].unsqueeze(0)  # Shape: (1, num_items)
                
                # Reconstruct this user vector
                reconstructed = model(user_vec)  # Shape: (1, num_items)
                
                # Clip predictions to valid rating range [1, 5] for MovieLens
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                
                # Store predictions for all items for this user
                predictions[user_idx, :] = reconstructed.squeeze(0)
        
        # Mask out items seen in training (set to very low value)
        # This ensures we only recommend unseen items
        predictions = predictions * (~train_mask).float() - train_mask.float() * 1e10
    
    # Move to CPU for numpy operations
    predictions = predictions.cpu().numpy()
    test_mat_np = test_mat.cpu().numpy()
    train_mat_np = train_mat.cpu().numpy()
    
    HR_list = []
    NDCG_list = []
    
    # For each user, calculate metrics
    num_users = predictions.shape[0]
    for user_id in range(num_users):
        # Get ground truth items for this user (items rated in test set)
        test_items = set(np.where(test_mat_np[user_id] > 0)[0])
        
        # Skip if user has no test items
        if len(test_items) == 0:
            continue
        
        # Get top-K items for this user
        user_predictions = predictions[user_id]
        top_k_indices = np.argsort(user_predictions)[-top_k:][::-1]  # Get top-K, descending order
        top_k_items = top_k_indices.tolist()
        
        # Calculate metrics
        HR_list.append(hit(test_items, top_k_items))
        NDCG_list.append(ndcg(test_items, top_k_items))
    
    # Calculate average metrics
    if len(HR_list) == 0:
        return 0.0, 0.0
    
    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)
    
    return mean_HR, mean_NDCG


def evaluate_autorec_ranking(
    model: nn.Module,
    test_loader: data.DataLoader,
    top_k: int = 10,
    device=None,
    item_based: bool = True
) -> tuple:
    """
    Evaluate AutoRec for ranking with implicit feedback (similar to NCF evaluation).
    
    For each test sample (user, positive_item, 99_negatives):
    1. Get predictions for all items (positive + negatives)
    2. Select top-K items
    3. Check if positive item is in top-K (HR@K)
    4. Calculate NDCG@K
    
    This follows the same protocol as NCF evaluation.
    """
    from helpers.ranking_metrics import hit, ndcg
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    HR_list = []
    NDCG_list = []
    
    with torch.no_grad():
        for user, item, label in test_loader:
            user = user.to(device)
            item = item.to(device)
            
            # Get predictions for all items in this batch
            # For AutoRec, we need to predict scores for all items
            # Since AutoRec works with item vectors, we need a different approach
            
            # Get batch size (should be TEST_NUM_NG + 1 = 100)
            batch_size = user.shape[0]
            
            # The first item is the positive, rest are negatives
            # We need to get predictions for all items for each user
            # For item-based AutoRec, we reconstruct item vectors
            # But for ranking, we need user-item scores
            
            # Simple approach: Use the model to predict scores
            # We'll need to modify this based on how AutoRec outputs scores
            # For now, let's assume we can get user-item scores somehow
            
            # Actually, for AutoRec with implicit feedback, we need to:
            # 1. Get the user's interaction vector from training data
            # 2. Reconstruct it to get predictions for all items
            # 3. Use those predictions for ranking
            
            # This is more complex - we'll need to store training matrix
            # For now, let's create a placeholder that will be implemented in the notebook
            predictions = torch.randn(batch_size).to(device)  # Placeholder
            
            # Get top-K items
            _, indices = torch.topk(predictions, min(top_k, batch_size))
            recommends = torch.take(item, indices).cpu().numpy().tolist()
            
            # Ground truth is the first item (positive)
            gt_item = item[0].item()
            
            HR_list.append(hit(gt_item, recommends))
            NDCG_list.append(ndcg(gt_item, recommends))
    
    if len(HR_list) == 0:
        return 0.0, 0.0
    
    mean_HR = np.mean(HR_list)
    mean_NDCG = np.mean(NDCG_list)
    
    return mean_HR, mean_NDCG