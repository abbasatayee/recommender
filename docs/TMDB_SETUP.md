# TMDB API Setup for Movie Images and Trailers

This guide explains how to set up TMDB API access to get movie poster images and trailers in your recommendations.

## Why Use TMDB API?

The recommendation API now supports fetching:

- **Poster images**: High-quality movie poster images
- **Backdrop images**: Wide background images for movie pages
- **Trailer URLs**: YouTube trailer links for movies

## Setup Instructions

### 1. Get a TMDB API Key (Free)

1. Go to [TMDB](https://www.themoviedb.org/)
2. Create a free account
3. Go to [API Settings](https://www.themoviedb.org/settings/api)
4. Request an API key (automatic approval for free tier)
5. Copy your API key

### 2. Set the API Key

You can set the API key in two ways:

#### Option A: Environment Variable (Recommended)

```bash
export TMDB_API_KEY="your_api_key_here"
```

#### Option B: In your code

```python
from src.api.core.tmdb_service import TMDBService

service = TMDBService(api_key="your_api_key_here")
```

### 3. Optional: OMDB API Key for IMDb Posters

If you want to fall back to IMDb posters when TMDB doesn't have them:

1. Go to [OMDB API](http://www.omdbapi.com/apikey.aspx)
2. Get a free API key (1000 requests/day)
3. Set it as an environment variable:

```bash
export OMDB_API_KEY="your_omdb_api_key_here"
```

## Usage

Once the API key is set, the recommendation endpoints will automatically include:

- `poster_url`: Direct URL to the movie poster image
- `backdrop_url`: Direct URL to the backdrop image
- `trailer_url`: YouTube URL for the movie trailer

### Example Response

```json
{
  "user_id": 0,
  "recommendations": [
    {
      "item_id": 0,
      "score": 0.95,
      "movie": {
        "movie_id": 1,
        "title": "Toy Story (1995)",
        "genres": ["Animation", "Children's", "Comedy"],
        "poster_url": "https://image.tmdb.org/t/p/w500/...",
        "backdrop_url": "https://image.tmdb.org/t/p/w1280/...",
        "trailer_url": "https://www.youtube.com/watch?v=...",
        "imdb_url": "https://www.imdb.com/title/tt0114709/",
        "tmdb_url": "https://www.themoviedb.org/movie/862"
      }
    }
  ]
}
```

## Frontend Usage

### Display Poster Image

```html
<img src="{{ movie.poster_url }}" alt="{{ movie.title }}" />
```

### Display Backdrop

```html
<div style="background-image: url('{{ movie.backdrop_url }}')">
  <!-- Movie content -->
</div>
```

### Embed Trailer

```html
<!-- Using YouTube iframe -->
<iframe
  src="https://www.youtube.com/embed/{{ extract_video_id(movie.trailer_url) }}"
  frameborder="0"
  allowfullscreen
>
</iframe>
```

Or use a YouTube embed library that accepts the full URL.

## Caching

The TMDB service automatically caches API responses for 1 hour to:

- Reduce API calls
- Improve response times
- Stay within rate limits

## Without API Key

If no API key is set, the API will still work but:

- `poster_url` will be `null`
- `backdrop_url` will be `null`
- `trailer_url` will be `null`
- Other fields (title, genres, IMDb/TMDB URLs) will still be available

## Rate Limits

- **TMDB Free Tier**: 40 requests per 10 seconds
- **OMDB Free Tier**: 1000 requests per day

The caching mechanism helps stay within these limits.
