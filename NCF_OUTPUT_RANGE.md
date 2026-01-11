# NCF Output Range: Logits vs Probabilities

## Quick Answer

**NO, NCF interaction scores are NOT between 0 and 1.** They are **unbounded logits** (can be any real number: negative, zero, or positive).

## Detailed Explanation

### What NCF Actually Outputs

The NCF model's forward pass ends with:

```python
# Final linear layer outputs interaction score
prediction = self.predict_layer(concat)  # Linear layer - NO activation!
return prediction.view(-1)  # Returns raw logits
```

**Key Point:** There's **NO sigmoid activation** applied to the output. The model returns raw logits directly.

### What Are Logits?

**Logits** are the raw, unnormalized outputs from the final linear layer:

- **Unbounded**: Can be any real number (-∞ to +∞)
- **Examples**: -5.2, 0.0, 2.3, 10.7, -1.5, 15.9
- **Meaning**: Higher values = higher confidence of interaction

### Logits vs Probabilities

| Type                          | Range    | How to Get              | Example Values          |
| ----------------------------- | -------- | ----------------------- | ----------------------- |
| **Logits** (what NCF outputs) | (-∞, +∞) | Direct from model       | -5.2, 0.0, 2.3, 10.7    |
| **Probabilities**             | [0, 1]   | Apply sigmoid to logits | 0.005, 0.5, 0.91, 0.999 |

### Converting Logits to Probabilities

If you want probabilities (0 to 1), you need to apply the sigmoid function:

```python
import torch
import torch.nn.functional as F

# NCF outputs logits (unbounded)
logit = model(user, item)  # e.g., 2.3

# Convert to probability [0, 1]
probability = torch.sigmoid(logit)  # e.g., 0.91 (91% chance of interaction)
```

**Sigmoid transformation:**

- logit = -5.0 → probability ≈ 0.007 (0.7%)
- logit = 0.0 → probability = 0.5 (50%)
- logit = 2.3 → probability ≈ 0.91 (91%)
- logit = 5.0 → probability ≈ 0.993 (99.3%)

### Why BCEWithLogitsLoss?

The model uses `BCEWithLogitsLoss` which:

1. **Expects logits** (not probabilities)
2. **Internally applies sigmoid** during loss calculation
3. **Is numerically stable** (avoids overflow/underflow)

This is why the model doesn't apply sigmoid - the loss function does it internally!

### Current Implementation

In `src/api/core/inference_engine.py`:

```python
score = self.model(user_t, item_t).cpu().item()  # Returns logit, e.g., 2.3
return score  # This is NOT between 0 and 1!
```

**Example outputs you might see:**

- User 1, Item 5: `score = 2.3` (logit, not probability)
- User 1, Item 10: `score = -1.5` (negative logit)
- User 1, Item 20: `score = 4.7` (high positive logit)

### If You Want Probabilities

If you need probabilities (0 to 1), modify the inference code:

```python
import torch.nn.functional as F

# Get logit
logit = self.model(user_t, item_t)

# Convert to probability
probability = torch.sigmoid(logit).cpu().item()  # Now between 0 and 1

return probability  # e.g., 0.91 instead of 2.3
```

### Summary

| Question                            | Answer                                     |
| ----------------------------------- | ------------------------------------------ |
| Are NCF scores between 0 and 1?     | **NO** - they are unbounded logits         |
| What range are they?                | (-∞, +∞) - any real number                 |
| Can they be negative?               | **YES** - negative logits are common       |
| Can they be > 1?                    | **YES** - positive logits can be any value |
| How to get 0-1 range?               | Apply `torch.sigmoid()` to logits          |
| What does the model output?         | Raw logits (unbounded)                     |
| What does BCEWithLogitsLoss expect? | Logits (it applies sigmoid internally)     |

### Visual Example

```
NCF Model Output (Logits):
  Item A: -2.5  (low likelihood)
  Item B:  0.0  (neutral)
  Item C:  2.3  (high likelihood)
  Item D:  5.1  (very high likelihood)

After Sigmoid (Probabilities):
  Item A:  0.08  (8% chance)
  Item B:  0.50  (50% chance)
  Item C:  0.91  (91% chance)
  Item D:  0.99  (99% chance)
```

**For ranking/recommendations:** You don't need probabilities - logits work perfectly fine since you only care about relative ordering (higher = better).
