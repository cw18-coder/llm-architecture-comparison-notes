# Transformer Architecture Fundamentals

## Core Components Recap

### Multi-Head Attention (MHA)
The heart of transformers - allows each token to "look at" and gather information from other tokens in the sequence.

**Key intuition**: Instead of processing tokens sequentially (like RNNs), attention lets every token directly access information from every other token in parallel.

**Mechanism**:
- **Queries (Q)**: "What am I looking for?"
- **Keys (K)**: "What information do I contain?"
- **Values (V)**: "What information do I actually provide?"

```python
# Simplified attention mechanism
def attention(Q, K, V):
    # Compute attention scores: how much should each token attend to others?
    scores = Q @ K.T / sqrt(d_k)
    # Apply softmax to get probabilities
    attention_weights = softmax(scores)
    # Weighted combination of values
    output = attention_weights @ V
    return output
```

**Multi-Head**: Run multiple attention mechanisms in parallel, each focusing on different types of relationships.

### Feed-Forward Networks (FFN)
Simple MLP layers that process each token independently after attention.

**Purpose**: 
- Transform and refine the representations
- Typically much larger than attention layers (often 4x the hidden dimension)
- Where much of the model's "knowledge" is stored

### Normalization & Residual Connections
**Layer Normalization**: Stabilizes training by normalizing activations
**Residual Connections**: Allow gradients to flow directly, enabling deeper networks

## The Standard Transformer Block
```
Input → [Add & Norm] → Multi-Head Attention → [Add & Norm] → Feed-Forward → Output
         ↑_______________|                      ↑_______________|
         residual connections
```

---

## Now, let's explore the key question that motivates DeepSeek's innovations...