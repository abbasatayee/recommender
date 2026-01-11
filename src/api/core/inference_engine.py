"""
Core inference engine for NCF and AutoRec models.
"""
import torch
import numpy as np
from typing import List, Optional
from fastapi import HTTPException

from ..config import NCF_CONFIG, AUTOREC_CONFIG, DEVICE
from ..models.schemas import PredictionItem, RecommendationItem, MovieDetails
from .movie_metadata import get_metadata_manager


class InferenceEngine:
    """Base class for inference engines."""
    
    def validate_user(self, user_id: int, max_users: int, model_name: str) -> None:
        """Validate user ID."""
        if not (0 <= user_id < max_users):
            raise HTTPException(
                status_code=400,
                detail=f"User ID must be in [0, {max_users}) for {model_name}, got {user_id}"
            )
    
    def validate_item(self, item_id: int, max_items: int, model_name: str) -> None:
        """Validate item ID."""
        if not (0 <= item_id < max_items):
            raise HTTPException(
                status_code=400,
                detail=f"Item ID must be in [0, {max_items}) for {model_name}, got {item_id}"
            )
    
    def _create_movie_details(self, item_id: int) -> Optional[MovieDetails]:
        """Create movie details for an item ID."""
        movie_info = get_metadata_manager().get_movie_info(item_id)
        if movie_info:
            return MovieDetails(**movie_info)
        return None


class NCFInferenceEngine(InferenceEngine):
    """Inference engine for NCF model."""
    
    def __init__(self, model: Optional[torch.nn.Module]):
        """
        Initialize NCF inference engine.
        
        Args:
            model: Loaded NCF model
        """
        self.model = model
        self.config = NCF_CONFIG
        self.model_name = "NCF"
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict score for a single user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Prediction score
        """
        if self.model is None:
            raise HTTPException(status_code=503, detail="NCF model not loaded")
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        self.validate_item(item_id, self.config["item_num"], self.model_name)
        
        with torch.no_grad():
            user_t = torch.LongTensor([user_id]).to(DEVICE)
            item_t = torch.LongTensor([item_id]).to(DEVICE)
            score = self.model(user_t, item_t).cpu().item()
        
        return score
    
    def predict_batch(self, user_id: int, item_ids: List[int]) -> List[PredictionItem]:
        """
        Predict scores for a user and multiple items.
        
        Args:
            user_id: User ID
            item_ids: List of item IDs
        
        Returns:
            List of predictions sorted by score (descending)
        """
        if self.model is None:
            raise HTTPException(status_code=503, detail="NCF model not loaded")
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        for item_id in item_ids:
            self.validate_item(item_id, self.config["item_num"], self.model_name)
        
        with torch.no_grad():
            user_t = torch.LongTensor([user_id] * len(item_ids)).to(DEVICE)
            item_t = torch.LongTensor(item_ids).to(DEVICE)
            scores = self.model(user_t, item_t).cpu().numpy()
        
        predictions = []
        for item_id, score in zip(item_ids, scores):
            movie_details = self._create_movie_details(item_id)
            predictions.append(PredictionItem(
                item_id=int(item_id),
                score=float(score),
                movie=movie_details
            ))
        
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidate_item_ids: Optional[List[int]] = None
    ) -> List[RecommendationItem]:
        """
        Get top-K recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            candidate_item_ids: Optional list of candidate items
        
        Returns:
            List of top-K recommendations sorted by score (descending)
        """
        if self.model is None:
            raise HTTPException(status_code=503, detail="NCF model not loaded")
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        
        if candidate_item_ids is None:
            candidate_item_ids = list(range(self.config["item_num"]))
        else:
            for item_id in candidate_item_ids:
                self.validate_item(item_id, self.config["item_num"], self.model_name)
            candidate_item_ids = list(set(candidate_item_ids))
        
        if not candidate_item_ids:
            raise HTTPException(status_code=400, detail="No valid candidate items")
        
        k = min(k, len(candidate_item_ids))
        
        with torch.no_grad():
            user_t = torch.LongTensor([user_id] * len(candidate_item_ids)).to(DEVICE)
            item_t = torch.LongTensor(candidate_item_ids).to(DEVICE)
            scores = self.model(user_t, item_t).cpu().numpy()
        
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        recommendations = []
        for i in top_k_idx:
            item_id = candidate_item_ids[i]
            score = scores[i]
            movie_details = self._create_movie_details(item_id)
            recommendations.append(RecommendationItem(
                item_id=int(item_id),
                score=float(score),
                movie=movie_details
            ))
        
        return recommendations


class AutoRecInferenceEngine(InferenceEngine):
    """Inference engine for AutoRec model."""
    
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        training_tensor: Optional[torch.Tensor]
    ):
        """
        Initialize AutoRec inference engine.
        
        Args:
            model: Loaded AutoRec model
            training_tensor: Training matrix as PyTorch tensor
        """
        self.model = model
        self.training_tensor = training_tensor
        self.config = AUTOREC_CONFIG
        self.model_name = "AutoRec"
        self.item_based = self.config["item_based"]
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a single user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Prediction score (rating)
        """
        if self.model is None or self.training_tensor is None:
            raise HTTPException(
                status_code=503,
                detail="AutoRec model or training matrix not loaded"
            )
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        self.validate_item(item_id, self.config["item_num"], self.model_name)
        
        with torch.no_grad():
            if self.item_based:
                # Item-based: get item vector (column), predict for all users
                item_vec = self.training_tensor[:, item_id].unsqueeze(0)
                reconstructed = self.model(item_vec)
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                score = reconstructed[0, user_id].cpu().item()
            else:
                # User-based: get user vector (row), predict for all items
                user_vec = self.training_tensor[user_id, :].unsqueeze(0)
                reconstructed = self.model(user_vec)
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                score = reconstructed[0, item_id].cpu().item()
        
        return score
    
    def predict_batch(self, user_id: int, item_ids: List[int]) -> List[PredictionItem]:
        """
        Predict ratings for a user and multiple items.
        
        Args:
            user_id: User ID
            item_ids: List of item IDs
        
        Returns:
            List of predictions sorted by score (descending)
        """
        if self.model is None or self.training_tensor is None:
            raise HTTPException(
                status_code=503,
                detail="AutoRec model or training matrix not loaded"
            )
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        for item_id in item_ids:
            self.validate_item(item_id, self.config["item_num"], self.model_name)
        
        predictions = []
        
        with torch.no_grad():
            if self.item_based:
                for item_id in item_ids:
                    item_vec = self.training_tensor[:, item_id].unsqueeze(0)
                    reconstructed = self.model(item_vec)
                    reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                    score = reconstructed[0, user_id].cpu().item()
                    
                    movie_details = self._create_movie_details(item_id)
                    predictions.append(PredictionItem(
                        item_id=int(item_id),
                        score=float(score),
                        movie=movie_details
                    ))
            else:
                user_vec = self.training_tensor[user_id, :].unsqueeze(0)
                reconstructed = self.model(user_vec)
                reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)
                for item_id in item_ids:
                    score = reconstructed[0, item_id].cpu().item()
                    
                    movie_details = self._create_movie_details(item_id)
                    predictions.append(PredictionItem(
                        item_id=int(item_id),
                        score=float(score),
                        movie=movie_details
                    ))
        
        predictions.sort(key=lambda x: x.score, reverse=True)
        return predictions
    
    def recommend(
        self,
        user_id: int,
        k: int = 10,
        candidate_item_ids: Optional[List[int]] = None
    ) -> List[RecommendationItem]:
        """
        Get top-K recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            candidate_item_ids: Optional list of candidate items
        
        Returns:
            List of top-K recommendations sorted by score (descending)
        """
        if self.model is None or self.training_tensor is None:
            raise HTTPException(
                status_code=503,
                detail="AutoRec model or training matrix not loaded"
            )
        
        self.validate_user(user_id, self.config["user_num"], self.model_name)
        
        if candidate_item_ids is None:
            candidate_item_ids = list(range(self.config["item_num"]))
        else:
            for item_id in candidate_item_ids:
                self.validate_item(item_id, self.config["item_num"], self.model_name)
            candidate_item_ids = list(set(candidate_item_ids))
        
        if not candidate_item_ids:
            raise HTTPException(status_code=400, detail="No valid candidate items")
        
        k = min(k, len(candidate_item_ids))
        predictions = self.predict_batch(user_id, candidate_item_ids)
        
        recommendations = [
            RecommendationItem(
                item_id=pred.item_id,
                score=pred.score,
                movie=pred.movie
            )
            for pred in predictions[:k]
        ]
        
        return recommendations
