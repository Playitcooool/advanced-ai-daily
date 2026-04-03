# Day 06: Model Quantization -- TurboQuant & 1-bit LLMs

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Inference Optimization

> **Architecture Diagram**: ![Quantization Diagram](../diagrams/06-quantization.png)

---

## One-Line Summary

Quantization compresses model weights and activations from 16-bit floating point to 8-bit, 4-bit, or even 1-bit integers with minimal quality loss. TurboQuant and Bonsai 1-bit models push this to the extreme: **running 27B parameter models on a laptop with near-FP16 quality**.

---

## The Math of Quantization

### Uniform Quantization

Map a continuous range [a, b] to N discrete levels:

```
quantize(x, N):
  scale = (b - a) / (N - 1)         # quantization step size
  q(x) = round((x - a) / scale)      # quantize to integer
  dequantize(q) = q * scale + a      # dequantize back to floating point
```

The quantization error is bounded by `|x - dequantize(quantize(x))| <= scale / 2`. The smaller the step size (the larger N is), the smaller the error.

### Quantization Levels

| Format   | Bits | Levels | Memory (70B model) | Typical Use                        |
|----------|:----:|:------:|:------------------:|------------------------------------|
| FP16     | 16   | 65,536 | 140 GB             | Training, high-quality inference   |
| INT8     | 8    | 256    | 70 GB              | Serving, minimal quality loss      |
| Q4_0     | 4    | 16     | 35 GB              | Local LLMs (llama.cpp default)     |
| Q2_K     | 2    | 4-6    | 20-25 GB           | Memory-constrained devices         |
| **1-bit**| 1    | 2      | **~10 GB**         | Bonsai models                      |

---

## The Problem: Why Not Just Use Fewer Bits?

### Why 1-bit Is Hard

In 1-bit quantization, weights are either +1 or -1 (binary). A linear layer that normally computes:

```
y = W @ x        →  y ≈ sign(W) @ x  (but the error is massive!)
   [FP16, 65536 levels]    [1-bit, 2 levels]
```

**The challenge**: How do we choose the sign pattern of W so that `sign(W) @ x` approximates `W @ x` well enough? The naive approach of simply taking the sign of each weight independently destroys too much information.

### TurboQuant's Key Insight

Do not quantize the entire weight matrix at once. Instead:

1. **Decompose** weights into multiple groups with different scales
2. **Per-group quantization**: each group has its own (a, b) range
3. **Skip KV dequantization**: during autoregressive decoding, approximately 90% of KV cache dequantization work can be skipped

### Bonsai 1-bit Models

Bonsai achieves 1-bit quantization through three techniques:

1. **Binary weights** (+1/-1) with learned scaling per group
2. **Fine-tuning after quantization** -- not simple post-training quantization, but quantization-aware training
3. **Grouped structure** that preserves expressivity despite binary constraints

---

## Algorithm Walkthrough

```
┌──────────────────────────────────────────────┐
│         Model Quantization Pipeline          │
└──────────────────────────────────────────────┘

  FP16 Weights (16 bits)
  │
  ▼
┌─────────────────────────────────┐
│ Step 1: Group decomposition     │
│  W = [Group_1 | Group_2 | ...]  │
│  Each group gets its own scale  │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Step 2: Per-group scaling       │
│  scale_g = max(|W_g|) / (N/2-1) │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Step 3: Quantize each group     │
│  q(W_g) = round(W_g / scale_g)  │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ Step 4: (Optional) QAT fine-tune│
│  Fine-tune with quantization in  │
│  the forward pass                │
└─────────────────────────────────┘
```

### KV Cache Quantization

During autoregressive decoding, the KV cache grows linearly with sequence length. TurboQuant shows that **90% of KV dequantization work can be skipped** by:

1. Pre-computing dequantized KV for frequently accessed indices
2. Using low-bit (2-bit) precision for older tokens, which contribute less to attention
3. Employing mixed precision: recent tokens stored at higher precision, older tokens at lower precision

---

## Code Implementation

```python
import torch
import torch.nn.functional as F

def uniform_quantize(x: torch.Tensor, bits: int = 4):
    """
    Uniform per-tensor quantization.

    Args:
        x: input tensor (any shape)
        bits: number of quantization bits (1-8)

    Returns:
        q: quantized tensor as int8 (packed)
        scale: scale factor for dequantization
        zero_point: zero-point offset
    """
    N = 2 ** bits  # number of levels
    qmin = 0
    qmax = N - 1

    # Find per-tensor range
    x_min = x.min().item()
    x_max = x.max().item()

    # Scale and zero-point
    scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else 1.0
    zero_point = qmin - x_min / scale

    # Quantize
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return q.to(torch.int8), scale, zero_point

def per_group_quantize(x: torch.Tensor, group_size: int, bits: int = 4):
    """
    Per-group quantization (what llama.cpp / TurboQuant actually use).

    Each group of `group_size` elements gets its own scale.

    Args:
        x: weight tensor [out_features, in_features]
        group_size: number of elements per quantization group
        bits: quantization bits

    Returns:
        q: quantized weights [out_features, in_features]
        scales: scale factors [out_features, num_groups]
    """
    N = 2 ** bits
    qmin, qmax = 0, N - 1

    out_f, in_f = x.shape
    num_groups = (in_f + group_size - 1) // group_size

    q = torch.zeros_like(x, dtype=torch.int8)
    scales = torch.zeros(out_f, num_groups, device=x.device)

    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, in_f)
        chunk = x[:, start:end]

        # Per-group scale
        group_min = chunk.min(dim=1, keepdim=True)[0]
        group_max = chunk.max(dim=1, keepdim=True)[0]
        group_scale = (group_max - group_min) / (qmax - qmin)
        group_scale = torch.clamp(group_scale, min=1e-8)

        zp = qmin - group_min / group_scale
        q_chunk = torch.clamp(torch.round(chunk / group_scale + zp), qmin, qmax)

        q[:, start:end] = q_chunk.to(torch.int8)
        scales[:, g:g+1] = group_scale.squeeze(dim=1, keepdim=True)

    return q, scales

class QuantizedLinear(torch.nn.Module):
    """
    Quantized linear layer: quantize weights, keep INT8 matmul.
    """
    def __init__(self, in_features, out_features, bits=4, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # Initialize and quantize weights
        weight = torch.randn(out_features, in_features)
        q_w, scales = per_group_quantize(weight, group_size, bits)

        self.register_buffer("q_weight", q_w)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_point", torch.zeros(out_features,
                         max(1, (in_features + group_size - 1) // group_size)))

    def forward(self, x):
        # Dequantize on-the-fly during forward pass
        out_f, in_f = self.out_features, self.in_features
        num_groups = self.scales.size(1)
        gs = self.group_size

        # Reconstruct floating-point weights
        weight = torch.zeros(out_f, in_f, device=x.device)
        for g in range(num_groups):
            start = g * gs
            end = min((g + 1) * gs, in_f)
            weight[:, start:end] = (
                self.q_weight[:, start:end].to(x.dtype) *
                self.scales[:, g:g+1]
            )

        return F.linear(x, weight)
```

---

## Quantization-Aware Training (QAT)

Post-training quantization (PTQ) applies quantization to pre-trained weights without any additional training. QAT produces better results:

```python
def qat_finetune_step(model, x, y, optimizer, bits=4):
    """
    One training step with quantization in the forward pass.

    The backward pass uses the straight-through estimator,
    so gradients flow through the quantization operation
    to the original floating-point weights.
    """
    # Forward with fake quantization
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                # Quantize
                q_w, scale = uniform_quantize(param, bits=bits)
                # Dequantize (differentiable via straight-through)
                w_fp = q_w.float() * scale

                # Replace for forward pass
                # (gradient flows to original param through STE)
                param.data.copy_(w_fp)

    # Standard forward and backward
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
```

---

## Comparison: PTQ vs QAT vs 1-bit

| Method              | Bits | Quality Loss | Training Cost        | Best For                 |
|---------------------|:----:|:------------:|:--------------------:|--------------------------|
| FP16 (baseline)     | 16   | 0%           | --                   | Reference                |
| INT8 PTQ            | 8    | <0.5%        | None (post-hoc)      | Production serving       |
| INT4 GGUF           | 4    | 1-3%         | None                 | Local inference          |
| INT4 QAT            | 4    | <1%          | Light fine-tune       | Quality-critical         |
| Bonsai 1-bit        | 1    | 3-5%         | Full QAT training    | Extreme memory constraint |

TurboQuant with KV cache skip achieves:

- **22.8% decode speedup at 32K context** by skipping 90% of KV dequantization work
- Near-Q4_0 quality at approximately 10% smaller memory footprint

---

## Further Reading

- [TurboQuant: Accelerating 1-bit LLM Inference](https://arxiv.org/) -- Google's quantization framework
- [Bonsai: 1-bit Large Language Models](https://arxiv.org/) -- 1-bit training techniques
- [llama.cpp Quantization](https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md) -- GGUF format specification
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) -- Per-layer optimal quantization
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) -- Protecting salient weights

---

_Prev: [Day 05 - Multi-Agent Reflection](05-multi-agent-reflection.md)  |  Next: [Day 07 - RBF Attention](07-rbf-attention.md)_
