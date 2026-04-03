---
date: "2026-04-03"
difficulty: "高级"
category: "强化学习"
---

# 第 01 天：GRPO —— 组相对策略优化

> **观看动画演示**：![GRPO 动画](../gifs/01-grpo.gif)

---

## 一句话概要

GRPO 通过在同一个输入上生成多个回答并利用组内相对排名来估计优势值，从而消除了 PPO 中价值网络（Critic）的需求。它是 **DeepSeek-R1** 的核心训练算法。

---

## 为什么需要 GRPO？

### PPO 的问题

标准的近端策略优化（PPO）需要训练**两个同等大小**的网络：

- **策略网络（Actor）**：生成回答或动作
- **价值网络（Critic）**：评估每个状态的好坏

在 70B+ 参数量级的大语言模型时代，维护一个与策略网络同等大小的价值网络会使**显存占用翻倍**，并引入额外的训练不稳定性。价值网络本身需要从奖励信号中进行学习，如果训练不当，它会污染策略梯度，导致策略网络朝错误方向学习。

### GRPO 的核心思路

GRPO 提出了一个简单的问题：*我们能否通过比较同一输入生成的一组回答来估计优势值？*

答案是可以的。通过对同一个问题从当前策略中采样 G 个回答，计算它们的奖励，并相对于组内均值和标准差进行归一化，我们就得到了一个**完全不需要价值网络**的优势估计。组内均值充当基线，组内标准差提供自适应的归一化信号强度。

---

## 算法流程

```
==================================================================
          GRPO：一次训练步骤（无需价值网络！）
==================================================================

    ┌──────────────────┐
    │   输入问题 q      │
    └────────┬─────────┘
             │
             │  从当前策略采样 G 个回答
             ▼
    ┌──────────────────────────┐
    │  生成 G 个回答            │
    │  o_1, o_2, ..., o_G      │
    │  ~ π_θ(· | q)            │
    └──────────┬───────────────┘
               │
               │  用奖励模型评估每个回答
               ▼
    ┌──────────────────────────┐
    │  计算奖励                 │
    │  r_1, r_2, ..., r_G      │
    │  reward_model(o_i, q)   │
    └──────────┬───────────────┘
               │
               │  组内相对归一化
               ▼
    ┌──────────────────────────────────────────────────────┐
    │  组内归一化优势值（核心步骤）                         │
    │                                                      │
    │  μ  = (1/G) · Σ r_j                                  │
    │  σ  = √( (1/G) · Σ(r_j - μ)² ) + ε                  │
    │  A_i = (r_i - μ) / σ                                 │
    │                                                      │
    │  ★ 不需要价值网络 ★                                   │
    └──────────────────────┬───────────────────────────────┘
                           │
                           │  重要性采样比率
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  PPO 截断更新 + KL 正则化                            │
    │                                                      │
    │  ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)                   │
    │  L = E[ min(ρ·A, clip(ρ,1-ε,1+ε)·A) ]                │
    │      - β · D_KL(π_θ || π_ref)                        │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
                    更新 θ → θ_new
```

## 数学公式

### 优势估计（组内相对）

对于问题 q 的 G 个回答，计算标量奖励 r_1, ..., r_G：

```
μ   = (1/G) · Σ_{j=1}^{G} r_j                    -- 组内均值（基线）
σ   = √( (1/G) · Σ_{j=1}^{G} (r_j - μ)² ) + ε    -- 组内标准差
A_i = (r_i - μ) / σ                                -- 归一化优势值
```

- 组内均值 μ 替代了价值网络中的基线 V(s)
- 组内标准差 σ 归一化了优势信号：当回答差异大时放大梯度，回答相似时减小梯度
- 小常数 ε 防止除零

### 带 PPO 截断的策略目标函数

```
ρ_i(θ) = π_θ(o_i | q) / π_θ_old(o_i | q)           -- 重要性采样比率

L_GRPO(θ) = E[  min( ρ_i(θ) · A_i,
                      clip(ρ_i(θ), 1 - ε, 1 + ε) · A_i ) ]
            - β · D_KL( π_θ || π_ref )
```

其中：

- **A_i**：第 i 个回答的组内归一化优势值
- **ε**：截断参数（通常为 0.2），防止策略更新步长过大
- **β**：KL 惩罚系数，使当前策略不会偏离参考策略太远
- **D_KL**：当前策略与参考策略之间的 KL 散度

---

## PPO 与 GRPO 对比

| 维度 | PPO | GRPO |
|---|---|---|
| 价值网络 | 需要（与策略网络等大） | 不需要 |
| 优势值来源 | 价值函数 V(s) | 组内排名（均值 + 标准差归一化） |
| 显存占用 | 约 2 倍策略网络 | 约 1.2 倍策略网络 |
| 训练稳定性 | 受价值网络精度影响大 | 更稳定，无需辅助网络 |
| 最佳适用场景 | 通用强化学习任务 | 可验证奖励（数学、代码、问答） |
| 基线 | 学习得到的 V(s) | 组内均值 μ |
| 信号归一化 | 固定或学习 | 通过组内标准差 σ 自动缩放 |

---

## Python 代码实现

```python
import torch
import torch.nn.functional as F
import numpy as np


def group_normalize(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算组内归一化优势值。

    对每组回答，减去组内均值并除以组内标准差，
    从而在不使用价值网络的情况下获得优势估计。

    参数:
        rewards:   形状 (G,) 组内每个回答的标量奖励
        eps:       防止除零的小常数

    返回:
        advantages: 形状 (G,) 每个回答的归一化优势值
    """
    mean = rewards.mean(dim=0)
    std = rewards.std(dim=0) + eps
    advantages = (rewards - mean) / std
    return advantages


def ppo_clipped_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    计算 PPO 截断代理损失。

    通过截断重要性采样比率，防止策略更新步长过大。

    参数:
        policy_log_probs: 形状 (G,) 当前策略的对数概率
        old_log_probs:    形状 (G,) 旧策略的对数概率
        advantages:       形状 (G,) 组内归一化优势值
        epsilon:          截断阈值（默认 0.2）

    返回:
        loss_scalar: 平均截断损失（取反后被优化器最小化）
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surrogate_unclipped = ratio * advantages
    surrogate_clipped = clipped_ratio * advantages

    # 取最小值以实现保守更新
    loss = torch.min(surrogate_unclipped, surrogate_clipped)
    return loss.mean()


def kl_penalty(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """
    计算当前策略与参考策略之间的 KL 散度。

    此惩罚项防止策略过度偏离原始监督微调模型。

    参数:
        policy_logits: 形状 (G, 词表大小) 当前模型的 logits
        ref_logits:    形状 (G, 词表大小) 参考模型的 logits

    返回:
        kl_loss: 标量 KL 散度值
    """
    log_policy = F.log_softmax(policy_logits, dim=-1)
    log_ref = F.log_softmax(ref_logits, dim=-1)
    kl = F.kl_div(log_ref, log_policy, reduction="batchmean", log_target=True)
    return kl


def grpo_loss(
    policy_logits: torch.Tensor,
    old_logits: torch.Tensor,
    rewards: torch.Tensor,
    beta: float,
    ref_logits: torch.Tensor | None = None,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    完整的 GRPO 损失函数，结合组内归一化、PPO 截断和 KL 正则化。

    参数:
        policy_logits: 形状 (G, 词表大小) 当前策略的 logits
        old_logits:    形状 (G, 词表大小) 采样时的旧策略 logits
        rewards:       形状 (G,) 每个回答的标量奖励
        beta:          KL 惩罚系数
        ref_logits:    形状 (G, 词表大小) 参考策略的 logits（可选）
        epsilon:       PPO 截断参数（默认 0.2）

    返回:
        loss: 标量损失值（取反以便标准优化器进行最小化）
    """
    # --- 步骤 1：计算当前策略和旧策略下的对数概率 ---
    policy_log_probs = F.log_softmax(policy_logits, dim=-1).sum(dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1).sum(dim=-1)

    # --- 步骤 2：组内归一化优势值（无需价值网络！）---
    advantages = group_normalize(rewards)

    # --- 步骤 3：PPO 截断目标 ---
    clip_objective = ppo_clipped_loss(
        policy_log_probs, old_log_probs, advantages, epsilon
    )

    # --- 步骤 4：KL 正则化（可选但强烈推荐）---
    kl = torch.tensor(0.0, device=policy_logits.device)
    if ref_logits is not None:
        kl = kl_penalty(policy_logits, ref_logits)

    # --- 步骤 5：合并目标（取反以实现标准最小化）---
    total_loss = -(clip_objective - beta * kl)
    return total_loss


# ------------------------------------------------------------------
# 使用模拟数据验证实现的示例
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    G = 8        # 每组回答的数量
    V = 1000     # 词表大小（模拟）

    # 模拟的虚拟数据
    policy_logits = torch.randn(G, V)
    old_logits = torch.randn(G, V)
    ref_logits = torch.randn(G, V)
    rewards = torch.tensor([0.9, 0.3, 0.7, 0.5, 0.1, 0.8, 0.2, 0.6])

    beta = 0.01  # KL 惩罚系数

    loss = grpo_loss(
        policy_logits=policy_logits,
        old_logits=old_logits,
        rewards=rewards,
        beta=beta,
        ref_logits=ref_logits,
        epsilon=0.2,
    )

    print(f"GRPO 损失值: {loss.item():.6f}")

    # 梯度验证：反向传播可以正常执行
    loss.backward()
    print("反向传播成功 —— 实现具备可微性。")
```

---

## 深入探究

### 1. 为什么组内归一化优势值有效？

GRPO 用**采样基线**（组内均值）替代了学习到的价值函数 V(s)。这有两层意义：

首先，组内均值 μ 是当前策略下期望奖励的**无偏估计量**。在经典的策略梯度理论中，任何从奖励中减去的基线 b 都不会给梯度引入偏差，只要这个基线 b 不依赖于具体的动作。由于 μ 是基于整个组在分配优势值之前计算得到的，因此满足这一条件。

其次，除以组内标准差 σ 提供了**自动奖励缩放**功能。在训练初期，策略生成的回答多样性很高，奖励差异很大，此时 σ 较大，优势值被削弱——这防止了不稳定、过大的梯度更新。随着训练的推进，策略逐步改进，回答之间变得更加相似，σ 会缩小，优势值随之增大，从而提供更精细的学习信号。

| 组内回答质量情况 | σ 大小 | 对优势值的影响 |
|---|---|---|
| 所有回答相似 | 小 σ | 优势值大，信号强 |
| 回答差异很大 | 大 σ | 优势值被削弱，保守更新 |
| 全部都很差 | μ 低，小 σ | 强烈远离差回答 |
| 全部都好 | μ 高，小 σ | 强烈趋向好回答 |

### 2. 局限性与边界情况

GRPO 设计优雅，但也存在已知的失效模式：

**组内零和问题：** GRPO 的优势估计是相对的而非绝对的。如果 G 个回答全部都很差，它们的优势值之和仍然为零。组内均值会偏低，但最好的回答仍然会得到正的优势值并被强化——尽管这个回答客观上仍然很差。解决方案包括引入绝对奖励阈值，或混合监督微调数据。

**奖励噪声敏感性：** 如果奖励模型本身存在噪声，组内的排名可能不可靠。当两个回答之间的奖励差异小于奖励噪声时，GRPO 可能会将策略推向错误方向。增大组大小 G 有助于通过平均化来消除均值中的噪声，但无法修复两两比较中的排名错误。

**不适用于多步环境：** GRPO 的设计目标是单轮生成任务，即奖励在完成整个回答后即可获取。对于需要多步交互的顺序强化学习环境，GRPO 的组内归一化无法跨时间步传播优势值。在这些场景下，仍然需要使用价值网络或 GAE 风格的优势估计。

### 3. DeepSeek-R1 在 GRPO 之上的创新

DeepSeek-R1 以前所未有的规模应用了 GRPO，并引入了多项实用改进：

**基于规则的可验证奖励：** DeepSeek-R1 不依赖学习式的奖励模型，而是对数学和代码问题使用确定性的奖励信号：代码能否编译？能否通过单元测试？数学答案是否与标准答案一致？这在相应领域中完全消除了奖励模型的偏差。

**格式奖励：** 奖励信号还检查推理过程的结构属性：回答中是否包含清晰标注的"思考"部分和最终答案？推理过程是否结构清晰？这鼓励了模型涌现出推理能力，而无需显式的思维链监督。

**监督微调预热与大规模组：** DeepSeek-R1 首先用监督微调数据对策略进行预热训练，然后逐步过渡到纯 GRPO 训练。同时，它们在训练过程中使用了较大的组大小（G = 16 或更大），这为优势值计算提供了更可靠的均值和方差估计。

---

## 延伸阅读

- **DeepSeekMath**（GRPO 原始论文）：https://arxiv.org/abs/2402.03300
- **DeepSeek-R1**（GRPO 大规模应用）：https://arxiv.org/abs/2501.12948
- **PPO 论文**（近端策略优化）：https://arxiv.org/abs/1707.06347
- **GAE 论文**（广义优势估计）：https://arxiv.org/abs/1506.02438

---

_上一篇：无  |  下一篇：[第 02 天 —— 混合专家模型](02-mixture-of-experts.md)_
