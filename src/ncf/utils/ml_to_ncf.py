import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

def preprocess_ml1m_to_ncf_format(ratings_file, data_dir, test_ratio=0.2, test_negatives=99):
    ratings = pd.read_csv(
        ratings_file,
        sep='::',
        engine='python',
        names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
        dtype={'UserID': np.int32, 'MovieID': np.int32, 'Rating': np.float32, 'Timestamp': np.int32}
    )
    
    print(f"✓ Loaded {len(ratings)} ratings")
    print(f"  - Unique users: {ratings['UserID'].nunique()}")
    print(f"  - Unique movies: {ratings['MovieID'].nunique()}")
    
    # Filter ratings >= 4 (positive interactions)
    # In recommendation systems, we typically treat ratings >= 4 as positive
    print("\nFiltering positive interactions (ratings >= 4)...")
    positive_ratings = ratings[ratings['Rating'] >= 4].copy()
    print(f"✓ {len(positive_ratings)} positive interactions (out of {len(ratings)} total)")
    
    # Remap user and item IDs to be contiguous (0-indexed)
    print("\nRemapping user and item IDs to be contiguous...")
    unique_users = sorted(positive_ratings['UserID'].unique())
    unique_items = sorted(positive_ratings['MovieID'].unique())
    
    user_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    positive_ratings['user'] = positive_ratings['UserID'].map(user_map)
    positive_ratings['item'] = positive_ratings['MovieID'].map(item_map)
    
    user_num = len(unique_users)
    item_num = len(unique_items)
    print(f"✓ Remapped to {user_num} users and {item_num} items")
    
    # Create user-item pairs
    user_item_pairs = positive_ratings[['user', 'item']].values
    
    # Split into train and test sets
    print(f"\nSplitting data (train: {1-test_ratio:.0%}, test: {test_ratio:.0%})...")
    np.random.seed(42)  # For reproducibility
    n_total = len(user_item_pairs)
    n_test = int(n_total * test_ratio)
    
    # Shuffle indices
    indices = np.random.permutation(n_total)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    train_pairs = user_item_pairs[train_indices]
    test_pairs = user_item_pairs[test_indices]
    
    print(f"✓ Training pairs: {len(train_pairs)}")
    print(f"✓ Test pairs: {len(test_pairs)}")
    
    # Save training data
    train_file = os.path.join(data_dir, 'ml-1m.train.rating')
    print(f"\nSaving training data to {train_file}...")
    train_df = pd.DataFrame(train_pairs, columns=['user', 'item'])
    train_df.to_csv(train_file, sep='\t', header=False, index=False)
    print(f"✓ Saved {len(train_df)} training pairs")
    
    # Create training matrix (for negative sampling)
    print("\nCreating training interaction matrix...")
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for u, i in train_pairs:
        train_mat[u, i] = 1.0
    print(f"✓ Training matrix created: {train_mat.nnz} interactions")
    
    # Generate test negative samples
    print(f"\nGenerating test negative samples ({test_negatives} negatives per test case)...")
    test_negative_file = os.path.join(data_dir, 'ml-1m.test.negative')
    
    with open(test_negative_file, 'w') as f:
        for u, i in test_pairs:
            # Write the positive pair
            negatives = []
            attempts = 0
            max_attempts = test_negatives * 10  # Safety limit
            
            # Sample negative items (items not in training set for this user)
            while len(negatives) < test_negatives and attempts < max_attempts:
                neg_item = np.random.randint(item_num)
                # Make sure it's not in training set for this user
                if (u, neg_item) not in train_mat:
                    negatives.append(neg_item)
                attempts += 1
            
            # If we couldn't find enough negatives, pad with random items
            while len(negatives) < test_negatives:
                neg_item = np.random.randint(item_num)
                if neg_item not in negatives:
                    negatives.append(neg_item)
            
            # Write in NCF format: (user, item)\tneg1\tneg2\t...\tneg99
            line = f"({u}, {i})" + "\t" + "\t".join(map(str, negatives)) + "\n"
            f.write(line)
    
    print(f"✓ Generated test negative samples: {len(test_pairs)} test cases")
    
    # Save test data (for reference, though NCF mainly uses test.negative)
    test_file = os.path.join(data_dir, 'ml-1m.test.rating')
    test_df = pd.DataFrame(test_pairs, columns=['user', 'item'])
    test_df.to_csv(test_file, sep='\t', header=False, index=False)

    
    return train_file, test_file, test_negative_file, user_num, item_num, train_mat