# GRPO - Group Relative Policy Optimization

> **日期**: 2026-04-03 | **难度**: 进阶 | **类别**: 强化学习 / LLM 对齐

---

## 一句话总结

GRPO 是一种不需要 Value Network（Critic）的 RL 算法，通过对同一问题生成多个回答并比较组内相对优劣来更新策略。它是 PPO 的简化版，也是 DeepSeek-R1 使用的核心训练方法。

---

## 为什么需要 GRPO?

### PPO 的问题

标准 PPO 需要训练两个网络：
- **Actor (策略网络)**: 生成响应
- **Critic (价值网络)**: 评估状态的好坏

Critic 和 Actor 一样大时（70B 模型），资源消耗翻倍。并且 Critic 难以训练，容易不稳定。

### GRPO 的核心洞察

**如果同一问题生成多个回答，直接用这些回答之间的相对分数作为优势估计，还需要 Critic 吗？**

答案是不需要。这就是 GRPO 的核心思想。

---

## 算法详解

### 流程图

```
                     ┌─────────────────────────────────────────┐
                     │          GRPO 训练一步                     │
                     └─────────────────────────────────────────┘

        ┌──────────┐
        │  问题 q   │
        └────┬─────┘
             │
             ▼
    ┌────────────────────┐
    │  采样 G 个回答       │  o_1, o_2, ..., o_G ~ π_θ
    │  (来自同一策略)      │  (相同的输入 q)
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  给每个回答打分       │  r_1, r_2, ..., r_G
    │  (奖励模型/规则)     │  (e.g. 答案正确性)
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  计算组内优势        │  A_i = (r_i - mean(r)) / std(r)
    │  (不需要 Critic!)   │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │  用 PPO-clip 更新   │  L = min(ratio·A, clip(ratio)·A)
    │  策略参数 θ          │  + KL 正则惩罚
    └────────────────────┘
```

### 公式

**优势计算 (Advantage):**

给定问题 q，生成 G 个输出 o_1, ..., o_G，对应的奖励为 r_1, ..., r_G：

```
A_i = (r_i - μ) / σ

其中:
  μ = (1/G) * Σ r_j    # 组内平均奖励
  σ = √((1/G) * Σ(r_j - μ)²)  # 组内标准差
```

**策略损失 (GRPO-Clip):**

```
L_GRPO(θ) = E[q, {o_i}] [ (1/G) * Σ min(
    (π_θ(o_i|q) / π_θ_old(o_i|q)) * A_i,
    clip(ratio, 1-ε, 1+ε) * A_i
) - β * D_KL(π_θ || π_ref) ]
```

其中 ratio = π_θ(o_i|q) / π_θ_old(o_i|q)

### 与 PPO 的关键区别

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic 网络 | 需要 (与 Actor 同级大小) | 不需要 |
| 优势估计 | 基于 Critic 的 value 预测 | 基于组内相对排名 |
| 显存占用 | 2x Actor | 1.2x Actor (只需存 G 个 rollout) |
| 训练稳定性 | Critic 训练影响稳定性 | 优势直接可算，更稳定 |
| 适用场景 | 通用 RL | 特别适合可验证奖励 (答案对错分明) |

---

## 代码实现 (核心)

```python
import torch
import torch.nn.functional as F

def grpo_loss(policy_logits, old_logits, rewards, beta, ref_logits=None, epsilon=0.2):
    """
    GRPO 损失函数
    
    Args:
        policy_logits: [G, V] 当前策略的 logit
        old_logits:    [G, V] 旧策略的 logit (生成这些样本时的策略)
        rewards:       [G]   每个样本的标量奖励
        beta:          float KL 惩罚系数
        ref_logits:    [G, V] 参考策略的 logit (可选)
        epsilon:       float PPO clip 参数
    """
    # 计算概率比
    log_probs = F.log_softmax(policy_logits, dim=-1).sum(dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1).sum(dim=-1)
    
    # 组内归一化优势
    mean_r = rewards.mean()
    std_r = rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r
    
    # PPO-clip
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    
    loss_clip = torch.min(ratio * advantages, clipped * advantages)
    
    # KL 惩罚
    kl = 0.0
    if ref_logits is not None:
        log_policy = F.log_softmax(policy_logits, dim=-1)
        log_ref = F.log_softmax(ref_logits, dim=-1)
        kl = F.kl_div(log_ref, log_policy, reduction='batchmean', log_target=True)
    
    # 损失取负 (因为要最大化)
    grpo_loss = -(loss_clip.mean() - beta * kl)
    return grpo_loss
```

---

## 深度讨论

### 1. GRPO 的优势估计是否靠谱？

GRPO 用 **组内均值归一化** 替代 Critic 的优势函数估计。这本质上是给每个回答相对于同组其他回答定位：

- 组内最好的回答: A_i > 0  →  概率增加
- 组内最差的回答: A_i < 0  →  概率降低
- 组内平均水平: A_i ≈ 0  →  几乎不变

**直觉**: 如果 G 足够大（比如 G=16），组内均值近似于 "基线水平"。标准差归一化则提供了自适应的学习信号 -- 当所有回答都很差时（低方差组），梯度会被放大；当所有回答接近时（高方差组），梯度会被抑制。

### 2. 什么时候 GRPO 不如 PPO?

- **奖励噪声大时**: 如果奖励模型的判别力差，组内排名本身就不可信，GRPO 的信号质量下降
- **需要 bootstrapping 的任务**: GRPO 的组内归一化是 "零和" 的，无法判断所有回答都好或都坏
- **长序列决策**: GRPO 在单步生成任务表现好，在需要多步决策的环境中效果较差

### 3. DeepSeek-R1 对 GRPO 的关键改进

- **规则奖励 + 格式奖励**: DeepSeek 不仅评估答案对错，还奖励模型输出推理过程的格式（thinking trace）
- **混合训练**: GRPO 与 SFT 数据混合训练，防止分布偏移过大
- **多次 rollout**: 每个问题采样更多候选，提升优势估计精度

---

## 扩展阅读

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (2024)](https://arxiv.org/abs/2402.03300) -- GRPO 首次提出
- [DeepSeek-R1: Incentivizing Reasoning Capability (2025)](https://arxiv.org/abs/2501.12948) -- GRPO 的大规模应用
- [PPO 原始论文](https://arxiv.org/abs/1707.06347)

---

_下一个教程: [Day 2 - MoE 架构深度解析](02-mixture-of-experts.md)_
