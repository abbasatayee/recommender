import os
import urllib.request
import zipfile

def download_ml1m_dataset(data_dir='./data'):
    """
    Downloads the MovieLens 1M dataset from the official source.
    
    Parameters:
    - data_dir: Directory where data will be stored
    
    Returns:
    - Path to the ratings.dat file
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Dataset URL and paths
    dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    zip_path = os.path.join(data_dir, 'ml-1m.zip')
    extract_path = os.path.join(data_dir, 'ml-1m')
    
    # The zip file contains a folder 'ml-1m', so after extraction:
    # Option 1: ./data/ml-1m/ratings.dat (if zip extracts to data_dir)
    # Option 2: ./data/ml-1m/ml-1m/ratings.dat (if zip structure is nested)
    # Let's check both possibilities
    ratings_file_option1 = os.path.join(extract_path, 'ratings.dat')
    ratings_file_option2 = os.path.join(extract_path, 'ml-1m', 'ratings.dat')
    
    # Check if already downloaded and extracted
    if os.path.exists(ratings_file_option1):
        print(f"✓ Dataset already exists at {ratings_file_option1}")
        return ratings_file_option1
    elif os.path.exists(ratings_file_option2):
        print(f"✓ Dataset already exists at {ratings_file_option2}")
        return ratings_file_option2
    
    # Download the dataset
    if not os.path.exists(zip_path):
        print(f"Downloading MovieLens 1M dataset from {dataset_url}...")
        print("This may take a few minutes...")
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("✓ Download complete!")
    else:
        print(f"✓ Zip file already exists at {zip_path}")
    
    # Extract the dataset
    if not os.path.exists(ratings_file_option1) and not os.path.exists(ratings_file_option2):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Extraction complete!")
        
        # Clean up zip file to save space
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("✓ Removed zip file to save space")
    else:
        print(f"✓ Dataset already extracted")
    
    # Find the ratings file (check both possible locations)
    ratings_file = None
    if os.path.exists(ratings_file_option1):
        ratings_file = ratings_file_option1
    elif os.path.exists(ratings_file_option2):
        ratings_file = ratings_file_option2
    else:
        # If still not found, search for ratings.dat in the extracted directory
        for root, dirs, files in os.walk(extract_path):
            if 'ratings.dat' in files:
                ratings_file = os.path.join(root, 'ratings.dat')
                break
    
    # Verify the ratings file exists
    if not ratings_file or not os.path.exists(ratings_file):
        raise FileNotFoundError(
            f"Expected ratings file not found. Checked:\n"
            f"  - {ratings_file_option1}\n"
            f"  - {ratings_file_option2}\n"
            f"  - Searched in {extract_path}"
        )
    
    print(f"✓ Dataset ready. Ratings file at: {ratings_file}")
    return ratings_file
