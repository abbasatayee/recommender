import numpy as np
import pandas as pd
from typing import Tuple
import torch.utils.data as data
from sklearn.model_selection import train_test_split

class AutoRecData(data.Dataset):
    r"""Dataset for Item-based AutoRec.
    
    For item-based AutoRec, we need to iterate over item vectors (columns of the rating matrix).
    Each item vector r^(i) = (R_{1i}, ..., R_{mi}) contains ratings from all users for item i.

    Parameters
    ----------
    data : np.ndarray
        Rating matrix of shape (num_users, num_items). Will be transposed internally
        to iterate over item vectors (columns).
    """

    def __init__(self, data: np.ndarray, item_based: bool = True) -> None:
        super(AutoRecData, self).__init__()

        # For item-based AutoRec, transpose to get item vectors as rows
        # Original: (num_users, num_items) -> Transposed: (num_items, num_users)
        if item_based:
            self.data = data.T  # Transpose: now each row is an item vector
        else:
            self.data = data  # User-based: each row is a user vector
        
        # Store original matrix for reference
        self.original_data = data
        self.items = set(data.nonzero()[1])  # Items with ratings
        self.users = set(data.nonzero()[0])  # Users with ratings

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> None:
        return self.data[index]

