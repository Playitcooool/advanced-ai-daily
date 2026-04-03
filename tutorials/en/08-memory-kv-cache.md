# Day 08: Memory & KV Cache Optimization

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Inference Optimization

> **Architecture Diagram**: ![KV Cache Diagram](../diagrams/08-memory-kv-cache.png)

---

## One-Line Summary

During autoregressive generation, the KV cache stores all previous key and value states and grows **linearly with sequence length**. For long contexts, the KV cache can consume more memory than the model weights themselves. **Optimizing KV cache is the key bottleneck for long-context inference.**

---

## The Memory Bottleneck

### KV Cache Growth

```
Standard autoregressive decoding:
  Step 1:     prompt → KV for 100 tokens   → cache: 100 × d_kv
  Step 2:     generate token 101           → cache: 101 × d_kv
  Step 3:     generate token 102           → cache: 102 × d_kv
  ...
  Step 1000:  generate token 1099          → cache: 1099 × d_kv

KV cache memory = 2 × n_layers × n_heads × d_head × seq_len × 2 (FP16)
                 = O(seq_len)  ← linear growth
```

For a 70B model with 128K context, the KV cache can exceed **50 GB** -- nearly half the model's own memory footprint.

### Where Memory Goes

```
┌──────────────────────────────────────────────────┐
│        Memory Breakdown (70B, 128K context)      │
├──────────────────────────────────────────────────┤
│  Model Weights (FP16)    │ 140 GB  ████████████  │
│  KV Cache (FP16)         │  96 GB  ████████      │ ← Dominates at long context!
│  Activations             │   8 GB  ▌             │
│  Optimizer states        │ 280 GB  (training only)│
└──────────────────────────────────────────────────┘
```

---

## Optimization Strategies

### 1. KV Cache Quantization

Quantize the cached K and V tensors to lower precision without any model retraining:

```python
class QuantizedKVCache:
    """
    KV Cache with INT8 quantization.

    Reduces KV cache memory by 50% with negligible quality loss.
    """
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cache = {
            'k': torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=torch.int8),
            'v': torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=torch.int8),
        }
        self.scales_k = torch.ones(n_layers, n_heads, 1, 1)  # [layer, head, 1, 1]
        self.scales_v = torch.ones(n_layers, n_heads, 1, 1)
        self.length = 0

    def append(self, layer_id, k_new, v_new):
        """Append new token's KV to cache (with quantization)."""
        k_new = k_new.detach()
        v_new = v_new.detach()

        # Per-head quantization scales
        self.scales_k[layer_id] = torch.abs(k_new).max(dim=-2, keepdim=True)[0] / 127.0
        self.scales_v[layer_id] = torch.abs(v_new).max(dim=-2, keepdim=True)[0] / 127.0

        # Quantize
        self.cache['k'][layer_id, self.length] = torch.round(
            k_new / self.scales_k[layer_id]).clamp(-128, 127).to(torch.int8)
        self.cache['v'][layer_id, self.length] = torch.round(
            v_new / self.scales_v[layer_id]).clamp(-128, 127).to(torch.int8)
        self.length += 1

    def get(self, layer_id):
        """Retrieve full KV cache (dequantized)."""
        k = self.cache['k'][layer_id, :self.length].to(torch.float32) * self.scales_k[layer_id]
        v = self.cache['v'][layer_id, :self.length].to(torch.float32) * self.scales_v[layer_id]
        return k, v
```

### 2. Selective Eviction

Do not store all tokens equally. Keep the important ones and discard the rest:

```python
class SelectiveKVCache:
    """
    Retain only tokens with high attention contribution.

    Strategy:
      1. Compute token importance scores during generation
      2. Keep top-k most important tokens
      3. Evict low-importance tokens from cache
    """
    def __init__(self, max_tokens, eviction_threshold=0.85):
        self.max_tokens = max_tokens
        self.threshold = eviction_threshold
        self.cache = {}  # layer_id -> (k, v) tuple
        self.importance = {}  # layer_id -> importance scores

    def update(self, layer_id, k, v, attention_weights):
        """Update KV cache with selective eviction."""
        current_len = k.size(1)  # sequence length dimension

        if current_len <= self.max_tokens:
            self.cache[layer_id] = (k, v)
            # Track importance: max attention weight received per token
            if layer_id not in self.importance:
                self.importance[layer_id] = torch.zeros(current_len, device=k.device)
            self.importance[layer_id] = torch.max(
                self.importance[layer_id],
                attention_weights.max(dim=1)[0]
            )
        else:
            # Evict least important tokens
            _, keep_indices = torch.topk(self.importance[layer_id], self.max_tokens)
            k_kept = k[:, keep_indices]
            v_kept = v[:, keep_indices]
            self.cache[layer_id] = (k_kept, v_kept)
            self.importance[layer_id] = self.importance[layer_id][keep_indices]
```

### 3. PagedAttention (vLLM)

vLLM's key innovation: treat KV cache memory like operating system page tables.

```
Traditional approach:
  Pre-allocate contiguous memory for max_seq_len → wastes memory

PagedAttention:
  Split KV cache into fixed-size "pages" (similar to OS 4KB pages)
  Pages can be non-contiguous in physical memory → eliminates fragmentation
  Logical-to-physical page table maps sequence positions to memory blocks
```

```python
class PagedAttentionKVCache:
    """
    PagedAttention-style KV cache management.

    Similar to OS virtual memory:
      - Logical KV blocks map to physical memory pages
      - No pre-allocation needed
      - Near-zero memory waste due to fragmentation
    """
    def __init__(self, page_size=16, max_num_pages=10000, head_dim=128, n_heads=32):
        self.page_size = page_size
        self.head_dim = head_dim
        self.n_heads = n_heads

        # Physical memory pool (non-contiguous)
        self.physical_k = torch.zeros(max_num_pages, page_size, n_heads, head_dim)
        self.physical_v = torch.zeros(max_num_pages, page_size, n_heads, head_dim)
        self.free_pages = list(range(max_num_pages))

        # Per-request page tables
        self.page_tables = {}  # request_id -> [page_idx_0, page_idx_1, ...]

    def allocate_page(self, request_id):
        """Allocate a new physical page for a request."""
        page_idx = self.free_pages.pop(0)
        if request_id not in self.page_tables:
            self.page_tables[request_id] = []
        self.page_tables[request_id].append(page_idx)
        return page_idx

    def get_kv_for_position(self, request_id, pos, layer_id):
        """Retrieve KV for a specific position via page table lookup."""
        page_idx = self.page_tables[request_id][pos // self.page_size]
        pos_in_page = pos % self.page_size
        k = self.physical_k[page_idx, pos_in_page:pos_in_page+1]
        v = self.physical_v[page_idx, pos_in_page:pos_in_page+1]
        return k, v
```

### 4. Mixed-Precision KV Cache

Recent tokens matter more in attention calculations, so store them at higher precision:

```
[Recent 256 tokens]     → FP16  (full precision)
[Tokens 256-4096]       → INT8  (good quality)
[Tokens 4096 and later] → INT4  (acceptable for old context)
```

```python
class MixedPrecisionKVCache:
    """
    Tiered KV cache: recent tokens in higher precision,
    older tokens in lower precision.
    """
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len,
                 tier_config=None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.current_len = 0

        if tier_config is None:
            # Default: 3 tiers
            tier_config = [
                (256, torch.float16),   # Recent: FP16
                (4096, torch.int8),     # Mid: INT8
                (max_seq_len, torch.int8),  # Old: INT4 (substitute for illustration)
            ]
        self.tiers = tier_config

    def store(self, layer_id, k, v):
        """Store KV in appropriate tier based on position."""
        batch_size, seq_len = k.shape[0], k.shape[1]
        self.current_len += seq_len
        # Route each position range to its corresponding tier
```

---

## TurboQuant KV Skip Optimization

TurboQuant's key insight for KV cache: **approximately 90% of KV dequantization computation can be skipped**.

### The Reasoning

During autoregressive decoding:

- Only the **last token** needs freshly computed attention against all previous tokens
- For older tokens, their attention patterns change very slowly between steps
- We can therefore **skip dequantizing and re-scoring** most historical KV entries between consecutive steps

### The Algorithm

```python
def turbo_kv_decode(last_k, last_v, kv_cache_quantized,
                    threshold=0.1, window_size=64):
    """
    TurboQuant KV skip optimization during decoding.

    Args:
        last_k: [1, heads, dim] latest key
        last_v: [1, heads, dim] latest value
        kv_cache_quantized: quantized KV cache
        threshold: similarity threshold below which to skip dequantization
        window_size: number of recent tokens to always compute fully
    """
    seq_len = kv_cache_quantized.shape[0]

    # Always compute attention for recent tokens within the sliding window
    if seq_len <= window_size:
        return full_attention(last_k, kv_cache_quantized)

    # For older tokens, check whether the attention pattern has changed significantly
    # Only dequantize and re-score when the change exceeds the threshold
    # This avoids redundant dequantization for tokens whose contribution
    # has remained stable across recent decoding steps.
```

**Result**: **22.8% decode speedup at 32K context** with negligible quality loss.

---

## Comparison of Approaches

| Method              | Memory Saved | Quality Loss | Retrain? | Complexity |
|---------------------|:------------:|:------------:|:--------:|:----------:|
| INT8 KV Cache       | 50%          | <0.1%        | No       | Low        |
| INT4 KV Cache       | 75%          | ~0.3%        | No       | Low        |
| Selective Eviction  | 60-80%       | 0.5-1%       | No       | Medium     |
| PagedAttention      | <5%*         | 0%           | No       | Medium     |
| Mixed Precision     | 40-60%       | ~0.2%        | No       | Medium     |
| Turbo KV Skip       | 0% (speedup) | <0.1%        | No       | High       |

*PagedAttention reduces memory fragmentation rather than raw size. It is a memory management optimization.

---

## Further Reading

- [TurboQuant: Accelerating LLM Inference](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/) -- TurboQuant explanation
- [PagedAttention in vLLM](https://vllm.ai/) -- vLLM's core KV cache optimization
- [KVCache-Transformer: A Survey](https://arxiv.org/) -- Comprehensive survey
- [StreamingLLM](https://arxiv.org/abs/2309.17453) -- Attention sinks and cache eviction
- [GEAR](https://arxiv.org/abs/2403.05527) -- Low-rank and sparse approximation of KV cache

---

_Prev: [Day 07 - RBF Attention](07-rbf-attention.md)  |  Next: --_
