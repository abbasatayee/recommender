import numpy as np
import torch.utils.data as data


class AutoRecImplicitData(data.Dataset):
    """
    Dataset for AutoRec with implicit feedback and negative sampling.
    
    Similar to NCFData, this handles:
    - Positive samples (user-item pairs)
    - Negative sampling during training
    - Binary labels (1 for positive, 0 for negative)
    """
    
    def __init__(self, features, num_item, train_mat=None, num_ng=0, is_training=None):
        super(AutoRecImplicitData, self).__init__()
        
        # Store positive samples (user-item pairs)
        self.features_ps = features
        
        # Store metadata
        self.num_item = num_item
        self.train_mat = train_mat  # Used to check if (user, item) exists
        self.num_ng = num_ng  # Number of negatives per positive
        self.is_training = is_training  # Training or testing mode
        
        # Initialize labels (will be filled during negative sampling)
        self.labels = [0 for _ in range(len(features))]
        
        # These will be populated by ng_sample() during training
        self.features_ng = []  # Negative samples
        self.features_fill = []  # Combined positives + negatives
        self.labels_fill = []  # Labels for combined features
    
    def ng_sample(self):
        """Generate negative samples for training."""
        assert self.is_training, 'Negative sampling only needed during training'
        
        self.features_ng = []
        
        # For each positive pair, generate num_ng negative samples
        for x in self.features_ps:
            u = x[0]  # User ID
            # Generate num_ng negative items for this user
            for t in range(self.num_ng):
                # Sample a random item
                j = np.random.randint(self.num_item)
                
                # Make sure this item is NOT in the user's training set
                # Keep sampling until we find a negative item
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                
                # Add this negative sample
                self.features_ng.append([u, j])
        
        # Create labels: 1 for positives, 0 for negatives
        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]
        
        # Combine positives and negatives
        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng
        
        print(f"✓ Generated {len(self.features_ng)} negative samples")
        print(f"  - Total samples (positives + negatives): {len(self.features_fill)}")
    
    def __len__(self):
        if self.is_training:
            return (self.num_ng + 1) * len(self.labels)
        else:
            return len(self.features_ps)
    
    def __getitem__(self, idx):
        # During training: use combined features (positives + negatives)
        # During testing: use only positive features
        if self.is_training:
            features = self.features_fill
            labels = self.labels_fill
        else:
            features = self.features_ps
            labels = self.labels
        
        # Get the specific sample
        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        
        return user, item, label
