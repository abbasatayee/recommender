# Explanation: Why NCF "Predicts Ratings" Despite Implicit Feedback Training

## Your Question

> "Both models predict rating to an item either the NCF was trained on implicit feedback and negative sampling, how it's possible that it predicts the rating score?"

This is an excellent observation that highlights a common misconception!

## The Answer

**NCF does NOT actually predict ratings** - it predicts **interaction scores (logits)**. The current implementation is using these scores as if they were ratings, which is conceptually incorrect but functionally works for ranking.

## Detailed Explanation

### 1. What NCF Actually Does

**Training Process:**

- Input: User-item pairs from training data (positive samples, label = 1)
- Negative Sampling: Random user-item pairs that don't exist (negative samples, label = 0)
- Loss Function: `BCEWithLogitsLoss` (Binary Cross-Entropy with Logits)
- Task: Binary classification - "Will this user interact with this item?"

**Model Output:**

- Raw logits (unbounded real numbers, typically -∞ to +∞)
- Higher logits = higher probability of interaction
- Lower/negative logits = lower probability of interaction

**Example Output:**

```
User 1, Item 5: score = 2.3  (likely to interact)
User 1, Item 10: score = -1.5 (unlikely to interact)
User 1, Item 20: score = 4.7 (very likely to interact)
```

These are **NOT ratings** - they're interaction likelihood scores!

### 2. Why It "Works" for Recommendations

For ranking/recommendation purposes, you don't need actual ratings:

- You just need to know which items are more likely to be interacted with
- Higher scores = better recommendations
- The absolute value doesn't matter, only the relative ordering

So using NCF logits for ranking is **perfectly fine** - but calling them "ratings" is misleading.

### 3. The Current Implementation

In `src/api/core/inference_engine.py`:

```python
# NCF - Returns logits (interaction scores)
score = self.model(user_t, item_t).cpu().item()  # e.g., 2.3, -1.5, 4.7
return score  # This is NOT a rating!
```

The code returns these logits directly, which works for ranking but is conceptually incorrect if you interpret them as ratings.

### 4. Comparison with AutoRec

**AutoRec (Explicit Feedback):**

- Trained on actual ratings (1-5 scale)
- Outputs are clamped to [1.0, 5.0]
- Actually predicts ratings

```python
# AutoRec - Returns actual ratings
reconstructed = self.model(user_vec)
reconstructed = torch.clamp(reconstructed, min=1.0, max=5.0)  # Clamp to [1, 5]
score = reconstructed[0, item_id].cpu().item()  # e.g., 3.2, 4.5, 2.1
return score  # This IS a rating!
```

## The Problem

The API currently treats both models' outputs the same way:

- Both return a "score"
- But NCF scores are logits (unbounded)
- AutoRec scores are ratings (1-5)

This creates confusion because:

1. NCF scores can be negative or very large (e.g., -5.2 or 12.3)
2. AutoRec scores are always between 1.0 and 5.0
3. They represent different things!

## Solutions

### Option 1: Clarify in Documentation (Current Approach)

- Document that NCF returns "interaction scores" not "ratings"
- Use NCF for ranking/recommendations
- Use AutoRec for actual rating prediction

### Option 2: Transform NCF Logits to Rating Scale (If Needed)

If you really need NCF to output something that looks like a rating:

```python
import torch.nn.functional as F

# Convert logit to probability [0, 1]
probability = torch.sigmoid(logit)

# Map to 1-5 scale (arbitrary transformation)
rating = 1.0 + (probability * 4.0)  # Maps [0,1] to [1,5]
```

**Warning:** This is arbitrary! The model wasn't trained to predict ratings, so this transformation has no theoretical basis. It's just a heuristic.

### Option 3: Use Appropriate Model for Each Task

- **Ranking/Recommendations**: Use NCF (interaction scores are perfect for this)
- **Rating Prediction**: Use AutoRec (trained for this purpose)

## Summary

| Aspect               | NCF                         | AutoRec                |
| -------------------- | --------------------------- | ---------------------- |
| **Training**         | Implicit feedback (binary)  | Explicit ratings (1-5) |
| **Output Type**      | Interaction scores (logits) | Predicted ratings      |
| **Output Range**     | Unbounded (-∞ to +∞)        | Bounded [1.0, 5.0]     |
| **What it predicts** | Likelihood of interaction   | Actual rating value    |
| **Use for**          | Ranking/Recommendations     | Rating prediction      |

**Key Insight:** NCF doesn't predict ratings - it predicts interaction likelihood. The current code uses these scores for ranking (which works), but they shouldn't be interpreted as ratings.

## Updated Code

I've updated the documentation in `src/api/core/inference_engine.py` to clarify:

- NCF methods now document that they return "interaction scores (logits)" not ratings
- AutoRec methods clearly state they return "predicted ratings (1.0 to 5.0)"

See `IMPLICIT_VS_EXPLICIT_FEEDBACK.md` for more detailed technical explanation.
