# Speculative Decoding - 投机解码

> **日期**: 2026-04-03 | **难度**: 进阶 | **类别**: 推理加速 / LLM 系统

---

## 一句话总结

用一个小模型 "猜" 后面几个 token，然后用大模型一次性验证。猜对了就赚，猜错了不亏。**这是一种严格无损的加速方法 -- 输出分布与大模型完全相同**，不会降低任何质量。

---

## 为什么需要 Speculative Decoding?

### 问题: LLM 推理是内存带宽绑定的

生成式 LLM 每次只能预测一个 token，然后要等 GPU 显存把整个模型参数读一遍。100B 参数的模型，生成一个 token 需要读 200GB 显存。GPU 的计算单元大量空闲，**瓶颈在显存带宽**。

```
GPU Compute:      [██░░░░░░░░░░░░░░░░]  15% 利用率
Memory Bandwidth: [███████████████████]  95% 利用率
                          ↑
                    这才是瓶颈
```

### 核心思路

如果一次验证 K 个 token 而不是生成 1 个 token，同样的内存带宽消耗，产出了 K 倍的 token -- 吞吐量提升了。

---

## 算法详解

### 流程图

```
┌──────────────────────────────────────────────────────────────┐
│                    标准自回归解码                              │
│                                                              │
│  Prompt → [LLM] → token_1 → [LLM] → token_2 → [LLM] → ...   │
│           (forward)   (forward)   (forward)                   │
│                                                              │
│  每生成 1 个 token，需要做 1 次完整的 LLM forward pass         │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   Speculative Decoding                        │
│                                                              │
│  步骤 1 (Draft):  小模型生成 K 个候选 token                     │
│                   prompt → [Draft] → d1,d2,...,dK            │
│                   (K 次 cheap forward pass)                   │
│                                                              │
│  步骤 2 (Verify): 大模型同时验证 K 个候选的                      │
│                   [LLM] → p(x|prompt), p(x|d1,...,d_{K-1})  │
│                   (1 次 forward pass 输出 K 个分布)           │
│                                                              │
│  步骤 3 (Accept/Reject):                                      │
│                   For each position i:                         │
│                     if d_i ~ p(x | ...): accept ✓            │
│                     else: sample from p(x | ...), reject ✗   │
│                                                              │
│  如果 K-1 个都猜对了 → K tokens / 1 pass = Kx 加速!            │
│  如果 1 个都没猜对 → 回退到从 LLM 分布采样 = 无损               │
└──────────────────────────────────────────────────────────────┘
```

### 接受/拒绝的数学细节

关键: **如何保证最终分布与只跑大模型完全一样？**

```python
def speculative_decode_step(draft_tokens, target_probs, draft_probs):
    """
    对每个 draft token 做接受/拒绝判定。
    
    重要: 这个接受概率设计保证了最终输出分布
    = 只用大模型生成的分布 (exact match)
    
    Args:
        draft_tokens: [K] 小模型生成的候选 token
        target_probs: [K, V] 大模型在每个位置的对下一个 token 的概率分布
        draft_probs:  [K, V] 小模型在每个位置对下一个 token 的概率分布
    """
    accepted_tokens = []
    for i, d_i in enumerate(draft_tokens):
        p = target_probs[i, d_i]      # 大模型认为这个 token 的概率
        q = draft_probs[i, d_i]       # 小模型认为这个 token 的概率
        
        # 接受概率: min(1, p/q)
        # 直觉: 
        #   - 如果大模型也觉得 d_i 是好选择 (p/q >= 1) → 100% 接受
        #   - 如果大模型觉得 d_i 不太对 (p/q < 1) → 以 p/q 概率接受
        if torch.rand(1).item() < min(1.0, p / (q + 1e-10)):
            accepted_tokens.append(d_i)   # 接受
        else:
            # 拒绝！从大模型的分布中重新采样
            resampled = torch.multinomial(target_probs[i], 1).item()
            accepted_tokens.append(resampled)
            break  # 第一个拒绝的位置之后的全部丢弃
    return accepted_tokens
```

### 期望加速比

设每个 draft token 的接受率为 α（小模型和大模型的一致性）：

```
期望接受 token 数 = Σ (α^i) for i=1 to K = α(1-α^K)/(1-α)

加速比 = 期望接受数 / 1 ≈ 1 / (1 - α)    (当 K 很大时)

举例:
  α = 0.7 (70% 猜中率) → 加速 ~3.3x
  α = 0.5 (50% 猜中率) → 加速 2x
  α = 0.3 (30% 猜中率) → 加速 ~1.4x
```

---

## 现代变体

### 1. Medusa (多头投机解码)

不训练独立的 Draft 模型，而是在大模型旁边加几个 **额外的 LM Head**：

```
标准 LLM:
  [Embed] → [Layer 1] → [Layer 2] → ... → [LM Head] → token_1

Medusa LLM:
  [Embed] → [Layer 1] → [Layer 2] → ... → [LM Head 1] → token_1
                                      → [Head 2] → token_2
                                      → [Head 3] → token_3
                                      → [Head 4] → token_4
```

**好处**: 不需要独立训练小模型，直接在已有大模型上微调几个小头。

### 2. Lookahead Decoding

用 **n-gram 缓存** 作为 Draft 模型。如果之前出现过类似的上下文模式，直接复用之前的输出：

```python
# Lookahead: 基于历史 n-gram 推测
cache = {
    "The capital of France is": "Paris, which",
    "Python is a programming": "language that",
}
draft = cache.get(current_prefix, "").split()  # 直接查缓存
```

### 3. EAGLE / EAGLE-2

EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency) 在 Draft 阶段用特征外推（feature extrapolation）而非独立模型：

- 利用大模型的 hidden states 做线性外推
- Draft 阶段不需要任何额外参数
- 比 Medusa 更省内存

---

## 何时最有效

| 场景 | 接受率 α | 预计加速 | 原因 |
|------|---------|---------|------|
| 代码补全 | 0.7-0.8 | 2.5-4x | 代码有强模式 |
| 翻译 | 0.5-0.6 | 1.8-2.2x | 句式有一定可预测性 |
| 闲聊 | 0.3-0.5 | 1.2-1.8x | 开放性高，难预测 |
| 数学推理 | 0.2-0.3 | 1.1-1.3x | 复杂推理，小模型难跟 |

---

## 扩展阅读

- [Speculative Decoding (原始论文)](https://arxiv.org/abs/2211.17192)
- [Medusa: Speculative Decoding with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057)
- [EAGLE: Speculative Sampling with Rethinking Feature Extrapolation](https://arxiv.org/abs/2404.02818)

---

_上一个: [Day 2 - MoE](02-mixture-of-experts.md) | 下一个: [Day 4 - Test-Time Compute](04-test-time-compute.md)_
