# Recommendation System

A PyTorch implementation of neural collaborative filtering models for recommendation systems, including:

- **NCF (Neural Collaborative Filtering)**: Implementation of He et al. "Neural Collaborative Filtering" at WWW'17
- **AutoRec**: Autoencoder-based collaborative filtering model
- **HybridAutoRecNCF**: Hybrid model combining AutoRec and NCF

This project provides both training notebooks and a production-ready FastAPI inference service for making recommendations using trained models.

## Dataset

The project uses the MovieLens 1M dataset. The data is processed from the format provided by Xiangnan's [NCF repository](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data).

## Model Training

### NCF Model

- Factor number: **32**
- MLP layers: **3**
- Epochs: **20**
- The implementation replicates the performance of the original NCF paper using the same settings (batch size, learning rate, initialization methods) as Xiangnan's [keras repository](https://github.com/hexiangnan/neural_collaborative_filtering).

### AutoRec Model

- Item-based autoencoder architecture
- Hidden units: **500**
- Trained on explicit ratings from the MovieLens dataset

## Requirements

The project requires Python 3.6+ with the following key dependencies:

- `torch` (PyTorch)
- `pandas`
- `numpy`
- `fastapi`
- `uvicorn`
- `scikit-learn`
- `matplotlib`
- `tqdm`

For a complete list of dependencies, see `requirements.txt`. Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Inference API

A FastAPI-based inference service is provided for making predictions and recommendations using trained NCF and AutoRec models.

### Running the API

Start the inference server from the project root:

```bash
python src/api/inference.py
```

Or using uvicorn directly:

```bash
uvicorn src.api.inference:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### General Endpoints

##### Root Endpoint

```bash
GET /
```

Returns API information and available models.

##### Health Check

```bash
GET /health
```

Returns the health status of the API, loaded models, and configuration information.

**Response:**

```json
{
  "status": "healthy",
  "ncf_loaded": true,
  "autorec_loaded": true,
  "device": "cuda",
  "ncf_config": {
    "user_num": 6038,
    "item_num": 3533
  },
  "autorec_config": {
    "user_num": 6040,
    "item_num": 3706
  }
}
```

#### NCF Endpoints

All NCF endpoints are prefixed with `/ncf`.

##### 1. Single Prediction

```bash
POST /ncf/predict
```

Predict interaction score for a single user-item pair using NCF.

**Request Body:**

```json
{
  "user_id": 0,
  "item_id": 100
}
```

**Response:**

```json
{
  "user_id": 0,
  "item_id": 100,
  "score": 0.85,
  "movie": {
    "title": "Movie Title",
    "genres": ["Action", "Adventure"]
  }
}
```

##### 2. Batch Prediction

```bash
POST /ncf/predict/batch
```

Predict scores for a user and multiple items using NCF.

**Request Body:**

```json
{
  "user_id": 0,
  "item_ids": [100, 200, 300]
}
```

**Response:**

```json
{
  "user_id": 0,
  "predictions": [
    { "item_id": 300, "score": 0.92 },
    { "item_id": 100, "score": 0.85 },
    { "item_id": 200, "score": 0.78 }
  ]
}
```

##### 3. Recommendations

```bash
POST /ncf/recommend
```

Get top-K item recommendations for a user using NCF.

**Request Body:**

```json
{
  "user_id": 0,
  "k": 10,
  "candidate_item_ids": null
}
```

**Response:**

```json
{
  "user_id": 0,
  "recommendations": [
    { "item_id": 1234, "score": 0.95 },
    { "item_id": 5678, "score": 0.92 }
  ],
  "k": 10
}
```

#### AutoRec Endpoints

All AutoRec endpoints are prefixed with `/autorec` and follow the same structure as NCF endpoints:

- `POST /autorec/predict` - Single prediction
- `POST /autorec/predict/batch` - Batch prediction
- `POST /autorec/recommend` - Top-K recommendations

### Example Usage

#### Using curl:

```bash
# Health check
curl http://localhost:8000/health

# NCF single prediction
curl -X POST "http://localhost:8000/ncf/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "item_id": 100}'

# NCF recommendations
curl -X POST "http://localhost:8000/ncf/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "k": 10}'

# AutoRec recommendations
curl -X POST "http://localhost:8000/autorec/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "k": 10}'
```

#### Using Python:

```python
import requests

# NCF single prediction
response = requests.post(
    "http://localhost:8000/ncf/predict",
    json={"user_id": 0, "item_id": 100}
)
print(response.json())

# NCF recommendations
response = requests.post(
    "http://localhost:8000/ncf/recommend",
    json={"user_id": 0, "k": 10}
)
print(response.json())

# AutoRec recommendations
response = requests.post(
    "http://localhost:8000/autorec/recommend",
    json={"user_id": 0, "k": 10}
)
print(response.json())
```

### Configuration

The API configuration can be modified in `src/api/config.py`:

- `NCF_MODEL_PATH`: Path to the trained NCF model file (default: `models/NeuMF.pth`)
- `AUTOREC_MODEL_PATH`: Path to the trained AutoRec model file (default: `models/AutoRec-best.pth`)
- `DEVICE`: Device to use ('cpu' or 'cuda', auto-detected)
- `NCF_CONFIG`: NCF model configuration (user_num, item_num, factor_num, etc.)
- `AUTOREC_CONFIG`: AutoRec model configuration (user_num, item_num, hidden_units, etc.)
- `API_PORT`: Port for the API server (default: 8000)

## Project Structure

```
recommender/
├── data/                    # Dataset files (MovieLens 1M)
│   └── ml-1m/
├── models/                  # Trained model checkpoints
├── src/
│   ├── api/                 # FastAPI inference service
│   │   ├── core/            # Core inference logic
│   │   ├── routes/          # API route handlers
│   │   └── inference.py     # API entry point
│   ├── ncf/                 # NCF model training notebooks
│   ├── autorec/             # AutoRec model training notebooks
│   ├── hybridautorecncf/    # Hybrid model training notebooks
│   └── helpers/              # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Additional Documentation

The project includes several documentation files explaining key concepts:

- **EXPLANATION_NCF_RATINGS.md**: Explains why NCF predicts interaction scores (not ratings) despite being used for rating prediction
- **IMPLICIT_VS_EXPLICIT_FEEDBACK.md**: Discussion on implicit vs explicit feedback in recommendation systems
- **NCF_OUTPUT_RANGE.md**: Information about NCF model output ranges and interpretation

## Training Models

Model training notebooks are located in:

- `src/ncf/` - NCF model training and evaluation
- `src/autorec/` - AutoRec model training and evaluation
- `src/hybridautorecncf/` - Hybrid model training

Each notebook includes data preprocessing, model training, evaluation, and model saving functionality.

## Model Files

Pre-trained models should be placed in the `models/` directory:

- `NeuMF.pth` - NCF (NeuMF) model
- `AutoRec-best.pth` - AutoRec model
- `HybridAutoRecNCF.pth` - Hybrid model

The API will automatically load these models on startup if they exist.

## License

This project implements models from academic papers:

- NCF: He et al. "Neural Collaborative Filtering" (WWW 2017)
- AutoRec: Sedhain et al. "AutoRec: Autoencoders Meet Collaborative Filtering" (WWW 2015)
