# Day 07: Beyond Dot-Product -- RBF Attention Mechanisms

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Transformer Architecture

> **Architecture Diagram**: ![RBF Attention Diagram](../diagrams/07-rbf-attention.png)

---

## One-Line Summary

RBF (Radial Basis Function) Attention replaces the standard dot-product `Q·K^T` in Transformers with a distance-based kernel `exp(-||Q - K||^2 / (2σ^2))`. This fundamentally changes how tokens measure similarity: **from angular alignment to spatial proximity**.

---

## Why Replace Dot-Product?

### The Limitation of Dot-Product Attention

Standard Transformer attention computes:

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

The core operation `Q·K^T` is an **inner product** (dot product):

- High when Q and K point in the same direction (small angle)
- Low when they are orthogonal or opposite

Inner product focuses on directional similarity between vectors, largely ignoring vector magnitude.

**The problem**: dot-product similarity has two known limitations:

1. **Magnitude sensitivity**: Large-norm queries dominate the softmax regardless of semantic meaning
2. **Linear decision boundary**: The condition `Q·K = 0` defines a hyperplane, which cannot capture complex nonlinear relationships

### RBF Attention's Solution

```
RBF_Attention(Q, K, V) = softmax( -||Q - K||^2 / (2σ^2) ) · V
```

The term `||Q - K||^2` is the **squared Euclidean distance** between Q and each K:

```
||Q - K||^2 = Σ (Q_i - K_i)^2
             = ||Q||^2 - 2(Q·K) + ||K||^2
```

When Q and K are close (small distance), the score is large (close to 1).

When Q and K are far apart, the score is exponentially suppressed.

---

## Mathematical Comparison

### Dot-Product vs RBF

```
Dot-Product:                    RBF:
  sim(q, k) = q·k                sim(q, k) = exp(-||q - k||^2 / (2σ^2))

Properties:                      Properties:
  - Linear in q and k            - Nonlinear (exponential)
  - Unbounded range (-∞, +∞)     - Bounded range (0, 1)
  - Direction similarity         - Spatial proximity
  - Sensitive to ||q||·||k||     - Sensitive to ||q - k||
```

### The Bandwidth Parameter σ

The parameter σ controls the "attention radius":

```
Large σ → exp(-small distance / large σ^2) ≈ 1:
  All tokens appear similar → attention becomes uniform (like average pooling)

Small σ → exp(-distance / small σ^2) decays rapidly:
  Only near-identical tokens are similar → attention becomes highly sparse

σ ≈ hidden_dim / √2 (sweet spot):
  Balanced between focused and distributed attention
```

### Relationship: RBF Can Express Dot-Product

```
exp(-||q - k||^2 / (2σ^2))
= exp(-||q||^2/(2σ^2)) · exp(-||k||^2/(2σ^2)) · exp(q·k/σ^2)

If we pre-normalize so that ||q||^2 = ||k||^2 = constant:
= C · exp(q·k/σ^2)

For small σ: exp(q·k/σ^2) ≈ 1 + q·k/σ^2 ≈ dot-product attention!
```

RBF attention is a **strict generalization** of dot-product attention. It can learn to approximate dot-product when beneficial, but it can also capture nonlinear relationships that dot-product cannot express.

---

## Implementation

```python
import torch
import torch.nn.functional as F
import math

class RBFAttention(torch.nn.Module):
    """
    Radial Basis Function Attention.
    Replaces Q·K^T with exp(-||Q - K||^2 / (2σ^2))
    """
    def __init__(self, d_model, n_heads, sigma=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        # Learnable bandwidth parameter sigma per head
        if sigma is not None:
            self.sigma = torch.nn.Parameter(torch.ones(n_heads) * sigma)
        else:
            # Default: sigma = sqrt(head_dim) -- standard initialization
            self.sigma = torch.nn.Parameter(torch.ones(n_heads) * math.sqrt(self.head_dim))

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Rearrange: [batch, heads, seq, dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute squared distances
        # ||q - k||^2 = ||q||^2 - 2*q·k + ||k||^2
        q_sq = (Q ** 2).sum(dim=-1, keepdim=True)   # [B, H, S, 1]
        k_sq = (K ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # [B, H, 1, S]
        dot = Q @ K.transpose(-1, -2)                # [B, H, S, S]

        dist_sq = q_sq - 2 * dot + k_sq              # [B, H, S, S]

        # RBF kernel: exp(-||q-k||^2 / (2*sigma^2))
        # sigma per head, unsqueezed for broadcasting
        sigma = self.sigma.view(1, self.n_heads, 1, 1)
        scores = -dist_sq / (2 * sigma ** 2)         # [B, H, S, S]

        # Attention mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # Apply to values
        output = (attn_weights @ V).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(output), attn_weights

class HybridAttention(torch.nn.Module):
    """
    Hybrid: combine dot-product and RBF attention.

    score = alpha * dot_product + (1 - alpha) * rbf

    Alpha is learned per head, so the model chooses the right mix.
    """
    def __init__(self, d_model, n_heads, sigma=None):
        super().__init__()
        self.rbf_attn = RBFAttention(d_model, n_heads, sigma)
        self.alpha = torch.nn.Parameter(torch.ones(n_heads) * 0.5)

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Dot-product scores
        dot_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # RBF scores (reuse calculation logic)
        q_sq = (Q ** 2).sum(-1, keepdim=True)
        k_sq = (K ** 2).sum(-1, keepdim=True).transpose(-1, -2)
        rbf_scores = -((q_sq - 2 * torch.matmul(Q, K.transpose(-1, -2)) + k_sq) /
                        (2 * self.rbf_attn.sigma.view(1, self.n_heads, 1, 1) ** 2))

        # Hybrid combination with learned mixing coefficient
        alpha = torch.sigmoid(self.alpha).view(1, self.n_heads, 1, 1)
        scores = alpha * dot_scores + (1 - alpha) * rbf_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = (attn @ V).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)
```

---

## Deep Dive

### When Does RBF Attention Help?

| Scenario                  | Dot-Product | RBF     | Why                                                                 |
|---------------------------|:-----------:|:-------:|---------------------------------------------------------------------|
| Semantic text similarity  | ★★★★        | ★★★★★   | RBF captures "closeness" in meaning-space more effectively           |
| Long-range dependencies   | ★★★★        | ★★      | Dot-product's unbounded range connects distant tokens better         |
| Numerical/tabular data    | ★★          | ★★★★★   | Euclidean distance is a natural metric for structured data           |
| Code completion           | ★★★         | ★★★★    | Code contains both directional (intent) and proximity (variable) signals |
| Multi-modal alignment     | ★★★         | ★★★★★   | Modalities live in spaces where distance is more informative than angle |

### The Sigma Learning Trade-off

```
Training dynamics:
  Epoch 1-10:   Sigma is large → attention is nearly uniform → model trains stably
  Epoch 10-50:  Sigma shrinks → attention becomes more selective → specialization begins
  Epoch 50+:    Different heads converge to different sigma values
                (some heads act like dot-product, others highly local)
```

This behavior mirrors how multi-head standard attention works: some heads learn to attend locally while others attend globally. However, RBF makes this specialization explicit through σ rather than implicit through the learned Q and K projections.

### Potential Issues

1. **Computational cost**: Computing `||Q-K||^2` requires `q_sq + k_sq - 2*dot`, which is slightly more expensive than dot-product alone (though negligible compared to the softmax operation).

2. **Sigma initialization sensitivity**: If sigma is too small at initialization, attention collapses to a one-hot distribution. If too large, it becomes uniform averaging.

3. **Compatibility with existing weights**: You cannot simply convert a pre-trained Transformer to RBF attention -- the Q and K projections would need fine-tuning.

---

## Further Reading

- [RBF Attention for Improved Similarity in Transformers](https://www.reddit.com/r/MachineLearning/comments/1s9cdq0/) -- Original Reddit discussion
- [Gaussian Attention: A RBF Approach to Self-Attention](https://arxiv.org/) -- Related work
- [Performers: Rethinking Attention with Positive Orthogonal Random Features](https://arxiv.org/abs/2009.14794) -- Alternative attention kernels
- [cosformer: Rethinking Softmax in Attention](https://arxiv.org/abs/2202.08791) -- Non-dot-product attention kernels

---

_Prev: [Day 06 - Quantization](06-quantization.md)  |  Next: [Day 08 - Memory & KV Cache Optimization](08-memory-kv-cache.md)_
