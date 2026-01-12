"""
TMDB API service for fetching movie images and trailers.
"""
import os
import time
from typing import Optional, Dict, Any
import requests
from functools import lru_cache

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Find project root (go up from src/api/core/ to project root)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    env_path = os.path.join(project_root, '.env')
    
    # Load .env from project root if it exists
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✓ Loaded .env file from {env_path}")
    else:
        # Try loading from current working directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, will use environment variables only
    pass
except Exception as e:
    print(f"⚠ Warning: Could not load .env file: {e}")


class TMDBService:
    """Service for fetching movie metadata from TMDB API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize TMDB service.
        
        Args:
            api_key: TMDB API key (optional, can be set via TMDB_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        self.base_url = "https://api.themoviedb.org/3"
        self.image_base_url = "https://image.tmdb.org/t/p"
        self.cache: Dict[int, Dict[str, Any]] = {}
        self.cache_ttl = 3600  # Cache for 1 hour
        self.last_fetch_time: Dict[int, float] = {}
        self._api_key_warning_shown = False
        
        # Show warning once at initialization if API key is not set
        if not self.api_key:
            print("=" * 70)
            print("⚠ TMDB API Key Not Set")
            print("=" * 70)
            print("To enable movie poster images and trailers:")
            print("  1. Get a free API key from https://www.themoviedb.org/settings/api")
            print("  2. Set it as an environment variable: export TMDB_API_KEY='your_key'")
            print("  3. Restart the API server")
            print("=" * 70)
            self._api_key_warning_shown = True
    
    def _is_cache_valid(self, tmdb_id: int) -> bool:
        """Check if cached data is still valid."""
        if tmdb_id not in self.cache:
            return False
        if tmdb_id not in self.last_fetch_time:
            return False
        return (time.time() - self.last_fetch_time[tmdb_id]) < self.cache_ttl
    
    def _fetch_movie_details(self, tmdb_id: int) -> Optional[Dict[str, Any]]:
        """Fetch movie details from TMDB API."""
        if not self.api_key:
            # Warning already shown at initialization, no need to repeat
            return None
        
        # Check cache first
        if self._is_cache_valid(tmdb_id):
            return self.cache[tmdb_id]
        
        try:
            # Fetch movie details including images and videos
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {
                "api_key": self.api_key,
                "append_to_response": "images,videos"
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            self.cache[tmdb_id] = data
            self.last_fetch_time[tmdb_id] = time.time()
            
            return data
        except requests.exceptions.HTTPError as e:
            print(f"⚠ HTTP Error fetching TMDB data for movie {tmdb_id}: {e}")
            if hasattr(e.response, 'status_code'):
                print(f"   Status code: {e.response.status_code}")
                if e.response.status_code == 401:
                    print(f"   Invalid API key. Please check your TMDB_API_KEY.")
            return None
        except Exception as e:
            print(f"⚠ Warning: Could not fetch TMDB data for movie {tmdb_id}: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            return None
    
    def get_poster_url(self, tmdb_id: int, size: str = "w500") -> Optional[str]:
        """
        Get poster image URL for a movie.
        
        Args:
            tmdb_id: TMDB movie ID
            size: Image size (w300, w500, w780, original)
        
        Returns:
            Direct URL to poster image, or None if not available
        """
        if not tmdb_id:
            return None
        
        movie_data = self._fetch_movie_details(tmdb_id)
        if not movie_data:
            return None
        
        # Try to get poster_path from movie data
        poster_path = movie_data.get("poster_path")
        if not poster_path:
            # Try from images
            images = movie_data.get("images", {})
            posters = images.get("posters", [])
            if posters:
                poster_path = posters[0].get("file_path")
        
        if poster_path:
            # Remove leading slash if present
            poster_path = poster_path.lstrip("/")
            return f"{self.image_base_url}/{size}/{poster_path}"
        
        return None
    
    def get_backdrop_url(self, tmdb_id: int, size: str = "w1280") -> Optional[str]:
        """
        Get backdrop image URL for a movie.
        
        Args:
            tmdb_id: TMDB movie ID
            size: Image size (w300, w780, w1280, original)
        
        Returns:
            Direct URL to backdrop image, or None if not available
        """
        if not tmdb_id:
            return None
        
        movie_data = self._fetch_movie_details(tmdb_id)
        if not movie_data:
            return None
        
        backdrop_path = movie_data.get("backdrop_path")
        if not backdrop_path:
            images = movie_data.get("images", {})
            backdrops = images.get("backdrops", [])
            if backdrops:
                backdrop_path = backdrops[0].get("file_path")
        
        if backdrop_path:
            backdrop_path = backdrop_path.lstrip("/")
            return f"{self.image_base_url}/{size}/{backdrop_path}"
        
        return None
    
    def get_trailer_url(self, tmdb_id: int) -> Optional[str]:
        """
        Get YouTube trailer URL for a movie.
        
        Args:
            tmdb_id: TMDB movie ID
        
        Returns:
            YouTube trailer URL, or None if not available
        """
        if not tmdb_id:
            return None
        
        movie_data = self._fetch_movie_details(tmdb_id)
        if not movie_data:
            return None
        
        videos = movie_data.get("videos", {})
        results = videos.get("results", [])
        
        # Look for official trailer first, then any trailer
        for video in results:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                if video.get("official", False):
                    return f"https://www.youtube.com/watch?v={video['key']}"
        
        # If no official trailer, get first YouTube trailer
        for video in results:
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={video['key']}"
        
        return None
    
    def get_imdb_poster_url(self, imdb_id: int) -> Optional[str]:
        """
        Get IMDb poster URL using OMDB API (requires OMDB_API_KEY).
        
        Note: This requires an OMDB API key. You can get one free at omdbapi.com
        
        Args:
            imdb_id: IMDb ID (numeric, without 'tt' prefix)
        
        Returns:
            IMDb poster URL, or None if not available
        """
        omdb_api_key = os.getenv("OMDB_API_KEY")
        if not omdb_api_key:
            return None
        
        try:
            # Format IMDb ID with leading zeros
            imdb_id_str = f"{imdb_id:07d}"
            url = f"http://www.omdbapi.com/?i=tt{imdb_id_str}&apikey={omdb_api_key}"
            
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Response") == "True":
                return data.get("Poster")
        except Exception as e:
            print(f"⚠ Warning: Could not fetch OMDB data for IMDb ID {imdb_id}: {e}")
        
        return None


# Global instance
_tmdb_service: Optional[TMDBService] = None


def get_tmdb_service() -> TMDBService:
    """Get or create the global TMDB service instance."""
    global _tmdb_service
    if _tmdb_service is None:
        _tmdb_service = TMDBService()
    return _tmdb_service
