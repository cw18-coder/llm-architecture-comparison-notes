# Positional Encoding: From Fixed to Rotary

## The Core Problem

**Question**: Why do transformers need positional information at all?

The self-attention mechanism is *permutation invariant* - it treats the sequence "The cat sat" identically to "sat The cat". Without position info, transformers can't understand word order!

## Traditional Fixed Positional Encoding

**Original Transformer Approach**: Add fixed sinusoidal patterns to token embeddings.

```python
import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings"""
    pe = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Even indices: sine
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            # Odd indices: cosine  
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return pe

# Example: 10 positions, 8 dimensions
pos_encoding = get_positional_encoding(10, 8)
print("Shape:", pos_encoding.shape)
print("Position 0:", pos_encoding[0])
print("Position 5:", pos_encoding[5])
```

**Key Properties**:
- Each position gets a unique "fingerprint"
- Different frequencies encode different aspects of position
- **Limitation**: Fixed maximum sequence length during training

---

## Rotary Position Encoding (RoPE)

**Core Innovation**: Instead of adding position info to embeddings, *rotate* the query and key vectors based on their positions.

### Geometric Intuition

Think of each pair of dimensions as a 2D coordinate. RoPE rotates these coordinates by an angle proportional to the token's position.

```python
def apply_rope(x, cos, sin):
    """Apply rotary position encoding to tensor x"""
    # Split into even and odd dimensions
    x1 = x[..., ::2]   # even indices [0, 2, 4, ...]
    x2 = x[..., 1::2]  # odd indices [1, 3, 5, ...]
    
    # Apply rotation matrix
    # [cos -sin] [x1]   [x1*cos - x2*sin]
    # [sin  cos] [x2] = [x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # Interleave back together
    output = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return output.flatten(-2)

def precompute_rope_params(seq_len, dim, base=10000):
    """Precompute cos and sin values for efficiency"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin
```

### Why RoPE is Brilliant

**Relative Position Awareness**: The dot product between query and key naturally encodes their relative distance:

```
Q(pos_i) • K(pos_j) = Q_original • K_original • cos(θ(pos_i - pos_j))
```

**Length Generalization**: Can handle sequences longer than training length (within limits).

**Efficiency**: No extra parameters, just rotation operations.

---

## Visual Comparison

```python
# Fixed PE: Position info is "added" to content
token_embedding + positional_encoding → input_to_attention

# RoPE: Position info is "rotated into" the queries and keys
query = rotate(Q_projection(x), position_angle)
key = rotate(K_projection(x), position_angle)
```

## Trade-offs Summary

| Aspect | Fixed PE | RoPE |
|--------|----------|------|
| **Length Generalization** | Poor | Good |
| **Relative Position** | Implicit | Explicit |
| **Computational Cost** | Addition only | Rotation ops |
| **Memory** | Extra parameters | No extra params |
| **Training Stability** | Well-established | Slightly more complex |

---

## Connecting to Our LLM Discussion

RoPE has become the standard because it solves the length generalization problem elegantly. Most models we'll discuss (DeepSeek, Llama, Qwen) use RoPE.

**Next**: Now that we understand how position is encoded, we can better appreciate how DeepSeek's MLA compresses the Key-Value cache while preserving positional relationships!