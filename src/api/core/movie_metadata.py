"""
Movie metadata loading and management.
"""
import os
import pandas as pd
import numpy as np
import urllib.parse
import random
from typing import Dict, List, Optional, Any
from ..config import SUPPORTED_ENCODINGS, MOVIES_FILE, RATINGS_FILE, LINKS_FILE
from .tmdb_service import get_tmdb_service


class MovieInfo:
    """Container for movie information."""
    
    def __init__(self, movie_id: int, title: str, genres: List[str], imdb_id: Optional[int] = None, tmdb_id: Optional[int] = None):
        """
        Initialize movie information.
        
        Args:
            movie_id: Original movie ID from the dataset
            title: Movie title
            genres: List of genre strings
            imdb_id: IMDb ID (numeric, without 'tt' prefix)
            tmdb_id: TMDB ID
        """
        self.movie_id = movie_id
        self.title = title
        self.genres = genres
        self.imdb_id = imdb_id
        self.tmdb_id = tmdb_id
        self.imdb_url = self._create_imdb_url()
        self.tmdb_url = self._create_tmdb_url()
        
        # Lazy-load poster and trailer URLs (only fetch when to_dict() is called)
        self._poster_url = None
        self._backdrop_url = None
        self._trailer_url = None
        self._images_loaded = False
    
    def _create_imdb_url(self) -> str:
        """Create IMDb URL for the movie."""
        if self.imdb_id:
            # Format IMDb ID with leading zeros (tt + 7 digits)
            imdb_id_str = f"{self.imdb_id:07d}"
            return f"https://www.imdb.com/title/tt{imdb_id_str}/"
        else:
            # Fallback to search URL if no IMDb ID
            encoded_title = urllib.parse.quote_plus(self.title)
            return f"https://www.imdb.com/find?q={encoded_title}"
    
    
    def _create_tmdb_url(self) -> Optional[str]:
        """Create TMDB URL for the movie."""
        if self.tmdb_id:
            return f"https://www.themoviedb.org/movie/{self.tmdb_id}"
        return None
    
    def _load_images(self) -> None:
        """Lazy-load poster, backdrop, and trailer URLs from TMDB service."""
        if self._images_loaded:
            return
        
        tmdb_service = get_tmdb_service()
        
        if self.tmdb_id:
            # Only log if there's an issue or if we successfully get URLs
            self._poster_url = tmdb_service.get_poster_url(self.tmdb_id)
            self._backdrop_url = tmdb_service.get_backdrop_url(self.tmdb_id)
            self._trailer_url = tmdb_service.get_trailer_url(self.tmdb_id)
            
            # Try to get IMDb poster if TMDB poster is not available
            if not self._poster_url and self.imdb_id:
                self._poster_url = tmdb_service.get_imdb_poster_url(self.imdb_id)
        elif self.imdb_id:
            # Try IMDb poster even without TMDB ID
            self._poster_url = tmdb_service.get_imdb_poster_url(self.imdb_id)
        
        self._images_loaded = True
    
    @property
    def poster_url(self) -> Optional[str]:
        """Get poster URL (lazy-loaded)."""
        self._load_images()
        return self._poster_url
    
    @property
    def backdrop_url(self) -> Optional[str]:
        """Get backdrop URL (lazy-loaded)."""
        self._load_images()
        return self._backdrop_url
    
    @property
    def trailer_url(self) -> Optional[str]:
        """Get trailer URL (lazy-loaded)."""
        self._load_images()
        return self._trailer_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "imdb_url": self.imdb_url,
            "tags": self.genres,  # Genres can be used as tags
        }
        
        # Load images if not already loaded
        self._load_images()
        
        # Add optional fields (always include, even if None, for API consistency)
        if self.tmdb_url:
            result["tmdb_url"] = self.tmdb_url
        
        # Always include these fields (None if not available)
        result["poster_url"] = self._poster_url
        result["backdrop_url"] = self._backdrop_url
        result["trailer_url"] = self._trailer_url
        
        if self.imdb_id:
            result["imdb_id"] = self.imdb_id
        if self.tmdb_id:
            result["tmdb_id"] = self.tmdb_id
        
        return result


class MovieMetadataManager:
    """Manages movie metadata loading and item-to-movie ID mapping."""
    
    def __init__(self):
        """Initialize the metadata manager."""
        self.movie_metadata: Dict[int, MovieInfo] = {}
        self.item_to_movie_mapping: Dict[int, int] = {}
        self.movie_links: Dict[int, Dict[str, Optional[int]]] = {}  # movie_id -> {imdb_id, tmdb_id}
        self._load_links()
        self._load_metadata()
        self._create_mapping()
    
    def _load_links(self) -> None:
        """Load movie links (IMDb and TMDB IDs) from link.csv file."""
        if not os.path.exists(LINKS_FILE):
            print(f"⚠ Warning: Links file not found at {LINKS_FILE}")
            return
        
        try:
            print(f"Loading movie links from {LINKS_FILE}...")
            links_df = pd.read_csv(
                LINKS_FILE,
                dtype={'movieId': np.int32, 'imdbId': str, 'tmdbId': str}
            )
            
            self.movie_links = {}
            for _, row in links_df.iterrows():
                movie_id = int(row['movieId'])
                imdb_id = None
                tmdb_id = None
                
                # Parse IMDb ID (handle NaN and convert to int)
                imdb_str = str(row['imdbId']).strip()
                if imdb_str and imdb_str != 'nan':
                    try:
                        imdb_id = int(float(imdb_str))
                    except (ValueError, TypeError):
                        pass
                
                # Parse TMDB ID (handle NaN and convert to int)
                tmdb_str = str(row['tmdbId']).strip()
                if tmdb_str and tmdb_str != 'nan':
                    try:
                        tmdb_id = int(float(tmdb_str))
                    except (ValueError, TypeError):
                        pass
                
                self.movie_links[movie_id] = {
                    'imdb_id': imdb_id,
                    'tmdb_id': tmdb_id
                }
            
            print(f"✓ Loaded links for {len(self.movie_links)} movies")
        except Exception as e:
            print(f"⚠ Warning: Could not load links file: {e}")
    
    def _load_metadata(self) -> None:
        """Load movie metadata from movies.dat file."""
        if not os.path.exists(MOVIES_FILE):
            print(f"⚠ Warning: Movies file not found at {MOVIES_FILE}")
            return
        
        for encoding in SUPPORTED_ENCODINGS:
            try:
                print(f"Loading movie metadata from {MOVIES_FILE} (encoding: {encoding})...")
                movies_df = pd.read_csv(
                    MOVIES_FILE,
                    sep='::',
                    header=None,
                    names=['movie_id', 'title', 'genres'],
                    engine='python',
                    encoding=encoding,
                    dtype={'movie_id': np.int32}
                )
                
                self.movie_metadata = {}
                for _, row in movies_df.iterrows():
                    movie_id = int(row['movie_id'])
                    title = str(row['title']).strip()
                    genres_str = str(row['genres']).strip()
                    genres = [g.strip() for g in genres_str.split('|')] if genres_str else []
                    
                    # Get IMDb and TMDB IDs from links
                    links = self.movie_links.get(movie_id, {})
                    imdb_id = links.get('imdb_id')
                    tmdb_id = links.get('tmdb_id')
                    
                    self.movie_metadata[movie_id] = MovieInfo(
                        movie_id, title, genres, imdb_id=imdb_id, tmdb_id=tmdb_id
                    )
                
                print(f"✓ Loaded metadata for {len(self.movie_metadata)} movies")
                return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"✗ Error loading movie metadata with {encoding}: {e}")
                continue
        
        print(f"✗ Error loading movie metadata: Could not decode file with any supported encoding")
    
    def _create_mapping(self) -> None:
        """Create mapping from remapped item_id to original MovieID."""
        if not os.path.exists(RATINGS_FILE):
            return
        
        for encoding in SUPPORTED_ENCODINGS:
            try:
                ratings_df = pd.read_csv(
                    RATINGS_FILE,
                    sep='::',
                    header=None,
                    names=['user_id', 'item_id', 'rating', 'timestamp'],
                    engine='python',
                    encoding=encoding,
                    dtype={'user_id': np.int32, 'item_id': np.int32}
                )
                
                unique_items = sorted(ratings_df['item_id'].unique())
                self.item_to_movie_mapping = {
                    new_id: old_id for new_id, old_id in enumerate(unique_items)
                }
                return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠ Warning: Could not create item-to-movie mapping with {encoding}: {e}")
                continue
        
        print(f"⚠ Warning: Could not create item-to-movie mapping: Could not decode file")
    
    def get_movie_info(self, item_id: int) -> Optional[Dict[str, Any]]:
        """
        Get movie information for a given item_id.
        
        Args:
            item_id: Remapped item ID (0-indexed)
        
        Returns:
            Dictionary with movie details, or None if not found
        """
        movie_id = self.item_to_movie_mapping.get(item_id)
        if movie_id is None:
            return None
        
        movie_info = self.movie_metadata.get(movie_id)
        if movie_info is None:
            return None
        
        return movie_info.to_dict()


# Global instance
_metadata_manager: Optional[MovieMetadataManager] = None
_top_rated_movies_cache: Dict[tuple, List[int]] = {}  # Cache key: (top_n, min_ratings)


def get_metadata_manager() -> MovieMetadataManager:
    """Get or create the global metadata manager instance."""
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MovieMetadataManager()
    return _metadata_manager


def get_movie_info(item_id: int) -> Optional[Dict[str, Any]]:
    """Get movie information for a given item_id (convenience function)."""
    return get_metadata_manager().get_movie_info(item_id)


def get_top_rated_movies(top_n: int = 100, min_ratings: int = 10) -> List[int]:
    """
    Get list of top-rated movie item IDs based on average ratings.
    Uses the already-loaded training tensor to avoid reloading the ratings file.
    
    Args:
        top_n: Number of top movies to return (default: 100)
        min_ratings: Minimum number of ratings required for a movie to be considered (default: 10)
    
    Returns:
        List of item IDs (0-indexed) for top-rated movies, sorted by average rating (descending)
    """
    global _top_rated_movies_cache
    
    # Check cache first
    cache_key = (top_n, min_ratings)
    if cache_key in _top_rated_movies_cache:
        return _top_rated_movies_cache[cache_key]
    
    # Use the already-loaded training tensor from the service instead of loading ratings file
    from .service import get_autorec_engine
    
    engine = get_autorec_engine()
    train_tensor = engine.training_tensor
    
    if train_tensor is None:
        _top_rated_movies_cache[cache_key] = []
        return []
    
    # Convert to numpy for easier computation
    train_mat = train_tensor.cpu().numpy()
    
    # Calculate average ratings per item (column-wise)
    # train_mat shape: (num_users, num_items)
    # For each item (column), calculate mean rating and count of ratings
    item_ratings = []
    num_items = train_mat.shape[1]
    
    for item_id in range(num_items):
        item_ratings_vec = train_mat[:, item_id]
        # Get non-zero ratings (actual ratings, not missing values)
        non_zero_ratings = item_ratings_vec[item_ratings_vec > 0]
        
        if len(non_zero_ratings) >= min_ratings:
            avg_rating = float(np.mean(non_zero_ratings))
            num_ratings = len(non_zero_ratings)
            item_ratings.append((item_id, avg_rating, num_ratings))
    
    if not item_ratings:
        _top_rated_movies_cache[cache_key] = []
        return []
    
    # Sort by average rating (descending)
    item_ratings.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N movies
    top_item_ids = [item_id for item_id, _, _ in item_ratings[:top_n]]
    
    # Cache the result
    _top_rated_movies_cache[cache_key] = top_item_ids
    
    return top_item_ids


def get_user_seen_movies(user_id: int) -> set:
    """
    Get set of item IDs that a user has already rated/seen.
    Uses the already-loaded training matrix from the service.
    
    Args:
        user_id: User ID (0-indexed, as used by the model)
    
    Returns:
        Set of item IDs (0-indexed) that the user has seen
    """
    from .service import get_autorec_engine
    
    # Get the training tensor from the engine (already loaded)
    engine = get_autorec_engine()
    train_tensor = engine.training_tensor
    
    if train_tensor is None:
        return set()
    
    # Validate user_id
    if not (0 <= user_id < train_tensor.shape[0]):
        return set()
    
    # Get all items this user has rated (non-zero entries in the user's row)
    user_ratings = train_tensor[user_id, :].cpu().numpy()
    seen_item_ids = set(np.where(user_ratings > 0)[0].tolist())
    
    return seen_item_ids


def get_random_top_rated_movie(user_id: Optional[int] = None, top_n: int = 100, min_ratings: int = 10) -> Optional[tuple]:
    """
    Get a random top-rated movie, excluding movies the user has already seen.
    
    Args:
        user_id: User ID (0-indexed). If provided, excludes movies the user has seen.
        top_n: Number of top movies to consider (default: 100)
        min_ratings: Minimum number of ratings required (default: 10)
    
    Returns:
        Tuple of (item_id, movie_info_dict), or None if no movies found
    """
    top_movies = get_top_rated_movies(top_n=top_n, min_ratings=min_ratings)
    
    if not top_movies:
        return None
    
    # Filter out movies the user has seen
    if user_id is not None:
        seen_movies = get_user_seen_movies(user_id)
        top_movies = [item_id for item_id in top_movies if item_id not in seen_movies]
    
    if not top_movies:
        return None
    
    # Randomly select one
    random_item_id = random.choice(top_movies)
    
    # Get movie info
    movie_info = get_movie_info(random_item_id)
    
    if movie_info is None:
        return None
    
    return (random_item_id, movie_info)
