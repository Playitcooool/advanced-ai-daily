# Day 03: Speculative Decoding
# 第 03 天: 投机解码

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Inference Acceleration 推理加速

> **Watch the animation**: ![Speculative Decoding Animation](../gifs/03-speculative-decoding.gif)

---

## One-Line Summary

Use a small model to "guess" the next K tokens, then verify them all with the large model in a single pass. **Strictly lossless** -- the output distribution is identical to running the large model alone.

用小模型"猜"后面 K 个 token，然后用大模型一次性验证。**严格无损** -- 输出分布与只跑大模型完全相同。

---

## The Problem | 问题所在

LLM inference is **memory-bandwidth bound**. Generating one token requires reading all model parameters through GPU memory. For a 100B model, that's ~200GB per token. The GPU compute units sit mostly idle.

LLM 推理是**内存带宽瓶颈**。生成一个 token 需要读取全部模型参数。100B 模型每次需要约 200GB 显存带宽。GPU 计算单元大量空闲。

```
GPU Compute:      [████░░░░░░░░░░░░░░░░]  15% utilization
Memory Bandwidth: [████████████████████]  95% utilization  ← BOTTLENECK
```

### The Core Idea | 核心思路

If we verify K tokens in one pass instead of generating 1, the same memory bandwidth produces Kx the output.

如果一次验证 K 个 token 而非生成 1 个，同样的带宽产出 K 倍的 token。

---

## Algorithm | 算法详解

### Standard Autoregressive Decoding | 标准自回归解码

```
Prompt → [LLM] → token_1 → [LLM] → token_2 → [LLM] → ...
         (forward)   (forward)   (forward)

1 generated token = 1 full LLM forward pass per token.
```

### Speculative Decoding | 投机解码

```
Step 1 - DRAFT (cheap):
  Prompt → [Draft Model, 1B] → d1, d2, d3, d4, d5
  5 cheap forward passes on small model

Step 2 - VERIFY (expensive, but only 1!):
  [Target Model, 70B] → p(x|prompt), p(x|d1), ..., p(x|d1...d4)
  1 forward pass produces distributions at all 5 positions

Step 3 - ACCEPT / REJECT:
  For each position i:
    if rand() < min(1, p(d_i) / q(d_i)):
       accept ✓  (keep draft token, free!)
    else:
       reject ✗  (resample from target's distribution)
       break     (discard all subsequent draft tokens)

If 4 of 5 accepted: 4 tokens from 1 expensive pass → ~5x speedup!
```

### Why Is It Lossless? | 为什么无损？

The acceptance probability `min(1, p/q)` is carefully designed so the final output distribution equals exactly running the large model:

- When the large model agrees (p/q ≥ 1): 100% accept the draft
- When the large model disagrees (p/q < 1): accept with probability p/q, otherwise resample from the target distribution

The mathematical proof guarantees distribution equivalence.

接受概率 `min(1, p/q)` 精心保证了最终分布等价于纯大模型采样。数学上可证明。

---

## Code | 代码实现

```python
@torch.no_grad()
def speculative_decode_step(draft_tokens, target_probs, draft_probs):
    """
    Accept/reject draft tokens while preserving exact target distribution.

    Args:
        draft_tokens:  [K]   small model's candidate tokens
        target_probs:  [K, V] large model's distributions at each position
        draft_probs:   [K, V] small model's distributions at each position

    Returns:
        accepted tokens (may include 1 resampled token at rejection point)
    """
    accepted = []
    for i, d_i in enumerate(draft_tokens):
        p = target_probs[i, d_i]          # target's confidence in draft
        q = draft_probs[i, d_i]           # draft's confidence
        if torch.rand(1).item() < min(1.0, p.item() / (q.item() + 1e-10)):
            accepted.append(d_i)           # accept -- free token!
        else:
            # Reject + resample from target distribution
            resampled = torch.multinomial(target_probs[i], 1).item()
            accepted.append(resampled)
            break  # discard all tokens after the first rejection
    return accepted
```

### Expected Speedup | 期望加速

If acceptance rate is α (how often draft agrees with target):

```
E[accepted] = Σ α^i  for i = 1 to K  =  α(1 - α^K) / (1 - α)

Speedup ≈ 1 / (1 - α)   (for large K)

Examples | 示例:
  α = 0.7 → ~3.3x speedup
  α = 0.5 → ~2.0x
  α = 0.3 → ~1.4x
```

---

## Modern Variants | 现代变体

### 1. Medusa

Instead of a separate draft model, add extra LM Heads to the target model:

```
Standard:  ... → [Transformer] → [LM Head] → token_1
Medusa:    ... → [Transformer] → [Head 1] → token_1
                                  → [Head 2] → token_2
                                  → [Head 3] → token_3
```

No extra training for the main model, just tune small heads.

### 2. Lookahead Decoding

Use n-gram cache as the draft: if a similar context appeared before, reuse its continuation. Zero parameters.

用历史 n-gram 缓存做 Draft，零参数。

### 3. EAGLE / EAGLE-2

Feature extrapolation instead of separate draft model. Uses the target model's hidden states to predict future tokens linearly. No extra parameters at all.

---

## When to Use | 适用场景

| Scenario | Acceptance Rate α | Speedup | Reason |
|----------|:-:|:-:|------|
| Code completion | 0.7-0.8 | 2.5-4x | Strong patterns in code |
| Translation | 0.5-0.6 | 1.8-2.2x | Some predictability |
| Chat | 0.3-0.5 | 1.2-1.8x | Open-ended |
| Math reasoning | 0.2-0.3 | 1.1-1.3x | Hard to predict |

---

## Further Reading

- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057)
- [EAGLE](https://arxiv.org/abs/2404.02818)

---

_Prev: [Day 02 - MoE](02-mixture-of-experts.md)  |  Next: [Day 04 - Test-Time Compute](04-test-time-compute.md)_
_上一篇: [Day 02 - MoE](02-mixture-of-experts.md)  |  下一篇: [Day 04 - 推理时计算](04-test-time-compute.md)_
