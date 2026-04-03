# Day 01: GRPO -- Group Relative Policy Optimization
# 第 01 天: GRPO 组相对策略优化

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: Reinforcement Learning

> **Watch the animation**: ![GRPO Animation](../gifs/01-grpo.gif)

---

## One-Line Summary

GRPO eliminates the need for a Value Network (Critic) in PPO by generating multiple responses to the same question and using their relative group ranking as advantage estimates. It is the core training method used in **DeepSeek-R1**.

GROP 不需要 PPO 中的价值网络（Critic），通过对同一问题生成多个回答并比较组内相对优劣来估计优势。它是 **DeepSeek-R1** 的核心训练方法。

---

## Why Do We Need GRPO?

### The PPO Problem

Standard PPO requires training **two networks** of equal size:
- **Actor (Policy)**: generates responses
- **Critic (Value)**: evaluates each state

When both are 70B parameters, this **doubles memory** and introduces training instability.

标准 PPO 需要训练两个同样大小的网络，当都是 70B 参数时，显存翻倍且训练不稳定。

### GRPO's Core Insight

> Can we estimate advantage by comparing a question's own generated responses?

**Answer: No Critic needed.** The group's mean-reward acts as a baseline; group std normalizes signal strength.

用组内均值做基线、标准差归一化，不需要价值网络。

---

## Algorithm

```
┌───────────────────────────────────┐
│      GRPO: One Training Step      │
└───────────────────────────────────┘

    ┌──────────┐
    │ Question  │
    │    q      │
    └────┬─────┘
         │
         ▼
┌──────────────────────┐
│ Sample G responses   │  o_1 ... o_G ~ π_θ
│ 采样 G 个回答          │  from same policy
└─────────┬────────────┘
          ▼
┌──────────────────────┐
│ Score each response  │  r_1 ... r_G
│ 给每个回答打分        │  (reward model)
└─────────┬────────────┘
          ▼
┌──────────────────────────────────────┐
│  Advantage: A_i = (r_i - μ) / σ     │
│  组内优势 = (奖励-均值)/标准差        │
│  -- NO CRITIC NEEDED!               │
└────────────────────┬─────────────────┘
                     ▼
┌──────────────────────────────────────┐
│  PPO-clip update + KL regularization │
│  策略更新                              │
└──────────────────────────────────────┘
```

### The Math

```
Advantage:
  μ   = (1/G) · Σ r_j
  σ   = √((1/G) · Σ(r_j - μ)²)
  A_i = (r_i - μ) / σ

Loss:
  ratio = π_θ(o_i|q) / π_θ_old(o_i|q)
  L = E[ min(ratio·A_i, clip(ratio,1-ε,1+ε)·A_i) ]
      - β · D_KL(π_θ || π_ref)
```

### PPO vs GRPO

| Dimension | PPO | GRPO |
|-----------|-----|------|
| Critic | Required (same size as Actor) | Not needed |
| Advantage | From Critic value | From group ranking |
| Memory | 2x Actor | ~1.2x Actor |
| Stability | Critic errors hurt | More stable |
| Best for | General RL | Verifiable rewards |

---

## Code Implementation

```python
import torch
import torch.nn.functional as F

def grpo_loss(policy_logits, old_logits, rewards, beta,
              ref_logits=None, epsilon=0.2):
    """
    GRPO loss function

    Args:
        policy_logits: [G, V] current policy logits (current model)
        old_logits:    [G, V] old policy logits (when samples were drawn)
        rewards:       [G]   scalar reward per response
        beta:          float KL penalty coefficient (KL惩罚系数)
        ref_logits:    [G, V] reference policy logits (optional)
        epsilon:       float PPO clip parameter
    """
    log_probs     = F.log_softmax(policy_logits, dim=-1).sum(dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1).sum(dim=-1)

    # Group-normalized advantage | 组内归一化优势
    mean_r = rewards.mean()
    std_r  = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r

    # PPO-clip
    ratio   = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss_clip = torch.min(ratio * advantages, clipped * advantages)

    # KL penalty | KL 惩罚
    kl = 0.0
    if ref_logits is not None:
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_ref    = F.log_softmax(ref_logits, dim=-1)
        kl = F.kl_div(log_ref, log_policy,
                      reduction='batchmean', log_target=True)

    # Negate for minimization (we maximize reward)
    return -(loss_clip.mean() - beta * kl)
```

---

## Deep Dive

### 1. Is the Advantage Estimate Reliable?

GRPO replaces the Critic with **group-relative normalization**:

| In group | A_i | Effect |
|----------|-----|--------|
| Best response | > 0 | Increase probability |
| Worst response | < 0 | Decrease probability |
| Average | approx 0 | Near-zero update |

The std normalization provides **adaptive signal**: when all responses are bad (low variance), gradients amplify; when similar (high variance), they dampen.

### 2. When Does GRPO Underperform?

- **Noisy rewards**: unreliable reward models make rankings noisy
- **Need for bootstrapping**: GRPO is zero-sum within a group -- can't tell if ALL responses are good or ALL are bad
- **Long-horizon tasks**: designed for single-step generation, not multi-step environments

### 3. DeepSeek-R1's Key Improvements

- Rule-based + format rewards (not just answer correctness, but reasoning trace structure)
- Mixed with SFT data to prevent excessive distribution shift
- Large G (more rollouts per question) for better advantage estimation

---

## Further Reading

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) -- Original GRPO
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) -- GRPO at scale
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

_Prev: --  |  Next: [Day 02 - MoE](02-mixture-of-experts.md)_
