# Docker Setup Guide

This guide explains how to run the Recommendation System API using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, but recommended)

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode (background):**
   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the image:**
   ```bash
   docker build -t recommender-api .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name recommender-api \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     -v $(pwd)/data:/app/data:ro \
     recommender-api
   ```

3. **View logs:**
   ```bash
   docker logs -f recommender-api
   ```

4. **Stop the container:**
   ```bash
   docker stop recommender-api
   docker rm recommender-api
   ```

## Environment Variables

### Optional Environment Variables

- `TMDB_API_KEY`: TMDB API key for fetching movie posters and trailers
- `OMDB_API_KEY`: OMDB API key for IMDb poster fallback

### Setting Environment Variables

#### With Docker Compose

1. Create a `.env` file in the project root (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API keys:
   ```env
   TMDB_API_KEY=your_key_here
   OMDB_API_KEY=your_key_here
   ```

3. Update `docker-compose.yml` to load the `.env` file:
   ```yaml
   environment:
     - TMDB_API_KEY=${TMDB_API_KEY}
     - OMDB_API_KEY=${OMDB_API_KEY}
   ```

#### With Docker Run

```bash
docker run -d \
  --name recommender-api \
  -p 8000:8000 \
  -e TMDB_API_KEY=your_key_here \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  recommender-api
```

## Data and Models

The Docker setup uses volume mounts to access your local data and model files:

- `./models` → `/app/models` (read-only)
- `./data` → `/app/data` (read-only)

This means:
- You can update models and data without rebuilding the Docker image
- The container has read-only access to prevent accidental modifications
- Make sure your `models/` and `data/` directories contain the required files

### Required Files

- **Models**: `models/NeuMF.pth` and/or `models/AutoRec-best.pth`
- **Data**: 
  - `data/ml-1m.train.rating` or `data/ml-1m/ratings.dat`
  - `data/ml-1m/movies.dat`
  - `data/link.csv`

## Accessing the API

Once the container is running, the API will be available at:

- **API Base**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Troubleshooting

### Container won't start

1. **Check logs:**
   ```bash
   docker-compose logs
   # or
   docker logs recommender-api
   ```

2. **Verify model files exist:**
   ```bash
   ls -la models/
   ```

3. **Verify data files exist:**
   ```bash
   ls -la data/
   ```

### API returns errors

1. **Check health endpoint:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify models are loaded correctly** (check the health response)

### Port already in use

If port 8000 is already in use, change it in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Use port 8001 on host
```

Then access the API at http://localhost:8001

## Development

For development, you might want to mount the source code as a volume for live updates:

```yaml
volumes:
  - ./src:/app/src
  - ./models:/app/models:ro
  - ./data:/app/data:ro
```

Note: This requires installing dependencies in the container or using a development Dockerfile.

## Production Considerations

For production deployment:

1. **Use specific image tags** instead of `latest`
2. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

3. **Use environment-specific configuration files**
4. **Set up proper logging** (consider using Docker logging drivers)
5. **Use a reverse proxy** (nginx, traefik) for SSL/TLS termination
6. **Consider using GPU support** if you have CUDA-enabled Docker:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```
