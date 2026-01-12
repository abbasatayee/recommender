# Implicit vs Explicit Feedback: NCF vs AutoRec

## The Key Difference

### NCF (Neural Collaborative Filtering)

- **Training**: Uses **implicit feedback** (binary classification)
  - Positive samples: User-item pairs that exist in training data (label = 1)
  - Negative samples: Randomly sampled user-item pairs that don't exist (label = 0)
  - Loss function: `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
- **Model Output**: **Interaction scores (logits)**, NOT ratings
  - Unbounded real numbers (can be negative or positive, typically -∞ to +∞)
  - **NOT between 0 and 1** - these are raw logits, not probabilities
  - Higher scores = higher likelihood of user-item interaction
  - These scores represent the model's confidence that a user will interact with an item
  - **NOT in 1-5 rating scale**
  - To get probabilities (0-1), apply sigmoid: `torch.sigmoid(logit)`
- **Use Case**: Ranking and recommendation
  - Scores are used to rank items (higher = more likely to interact)
  - Perfect for "top-K recommendations"
  - The absolute value is less important than the relative ranking

### AutoRec (Autoencoder for Collaborative Filtering)

- **Training**: Uses **explicit feedback** (rating prediction)
  - Input: Actual ratings from users (1-5 scale for MovieLens)
  - Loss function: Typically MSE (Mean Squared Error) for regression
  - Reconstructs the rating matrix
- **Model Output**: **Predicted ratings** (1-5 scale)
  - Outputs are clamped to [1.0, 5.0] range
  - Directly predicts what rating a user would give an item
  - These ARE actual ratings
- **Use Case**: Rating prediction
  - Can predict specific rating values
  - Useful when you need to know "how much" a user likes an item

## Why NCF Can't Predict "Ratings"

NCF models are trained to distinguish between:

- **Positive interactions** (user watched/clicked/bought item) → label = 1
- **Negative interactions** (user didn't interact) → label = 0

The model learns to output:

- **High logits** for positive interactions
- **Low/negative logits** for negative interactions

These logits are **not calibrated** to a 1-5 rating scale. They're just scores that indicate interaction likelihood.

## Current Implementation Issue

In `src/api/core/inference_engine.py`, the NCF model's raw logits are being returned as "scores":

```python
score = self.model(user_t, item_t).cpu().item()  # This is a logit, not a rating!
return score
```

This is **technically correct** for ranking/recommendation purposes, but **conceptually misleading** if you're calling it a "rating."

## Solutions

### Option 1: Keep as-is (Current Approach)

- Use NCF scores for **ranking only**
- Don't interpret them as ratings
- Rename API endpoints to clarify: "interaction_score" instead of "rating"

### Option 2: Transform to Rating Scale (If Needed)

If you need to convert NCF logits to a 1-5 rating scale, you could:

```python
import torch.nn.functional as F

# Convert logit to probability
probability = torch.sigmoid(score)  # Maps to [0, 1]

# Map probability to 1-5 rating scale
rating = 1.0 + (probability * 4.0)  # Maps to [1, 5]
```

However, this is **arbitrary** and not grounded in the training data. The model wasn't trained to predict ratings, so this transformation is just a heuristic.

### Option 3: Use AutoRec for Rating Prediction

- Use AutoRec when you need actual rating predictions (1-5 scale)
- Use NCF when you need ranking/recommendations (interaction likelihood)

## Best Practice

1. **For Recommendations**: Use NCF scores as-is (they're perfect for ranking)
2. **For Rating Prediction**: Use AutoRec (it's trained for this)
3. **Document clearly**: In API responses, distinguish between:
   - `interaction_score` (NCF) - likelihood of interaction
   - `predicted_rating` (AutoRec) - predicted rating on 1-5 scale

## Summary

| Aspect               | NCF                               | AutoRec                  |
| -------------------- | --------------------------------- | ------------------------ |
| Training Data        | Implicit (binary)                 | Explicit (ratings 1-5)   |
| Loss Function        | BCEWithLogitsLoss                 | MSE                      |
| Output Type          | Interaction scores (logits)       | Ratings (1-5)            |
| Output Range         | Unbounded logits (-∞ to +∞)       | Bounded [1.0, 5.0]       |
| Output Values        | Raw logits (e.g., -2.5, 0.0, 3.2) | Ratings (e.g., 2.1, 4.5) |
| Use Case             | Ranking/Recommendation            | Rating Prediction        |
| Can predict ratings? | No (only interaction likelihood)  | Yes (actual ratings)     |
