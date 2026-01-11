"""
Movie metadata loading and management.
"""
import os
import pandas as pd
import numpy as np
import urllib.parse
from typing import Dict, List, Optional, Any
from ..config import SUPPORTED_ENCODINGS, MOVIES_FILE, RATINGS_FILE


class MovieInfo:
    """Container for movie information."""
    
    def __init__(self, movie_id: int, title: str, genres: List[str]):
        """
        Initialize movie information.
        
        Args:
            movie_id: Original movie ID from the dataset
            title: Movie title
            genres: List of genre strings
        """
        self.movie_id = movie_id
        self.title = title
        self.genres = genres
        self.imdb_search_url = self._create_imdb_url()
    
    def _create_imdb_url(self) -> str:
        """Create IMDb search URL for the movie."""
        encoded_title = urllib.parse.quote_plus(self.title)
        return f"https://www.imdb.com/find?q={encoded_title}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "movie_id": self.movie_id,
            "title": self.title,
            "genres": self.genres,
            "imdb_url": self.imdb_search_url,
            "tags": self.genres  # Genres can be used as tags
        }


class MovieMetadataManager:
    """Manages movie metadata loading and item-to-movie ID mapping."""
    
    def __init__(self):
        """Initialize the metadata manager."""
        self.movie_metadata: Dict[int, MovieInfo] = {}
        self.item_to_movie_mapping: Dict[int, int] = {}
        self._load_metadata()
        self._create_mapping()
    
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
                    
                    self.movie_metadata[movie_id] = MovieInfo(movie_id, title, genres)
                
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


def get_metadata_manager() -> MovieMetadataManager:
    """Get or create the global metadata manager instance."""
    global _metadata_manager
    if _metadata_manager is None:
        _metadata_manager = MovieMetadataManager()
    return _metadata_manager


def get_movie_info(item_id: int) -> Optional[Dict[str, Any]]:
    """Get movie information for a given item_id (convenience function)."""
    return get_metadata_manager().get_movie_info(item_id)
