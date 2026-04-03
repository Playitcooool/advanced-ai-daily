---
date: "2026-04-03"
difficulty: "Advanced"
category: "Reinforcement Learning"
---

# Day 01: GRPO -- Group Relative Policy Optimization

> **Watch the animation**: ![GRPO Animation](../gifs/01-grpo.gif)

---

## One-Line Summary

GRPO eliminates the need for a Value Network (Critic) in PPO by generating multiple responses to the same input and using their relative group ranking as advantage estimates. It is the core training algorithm used in **DeepSeek-R1**.

---

## Why Do We Need GRPO?

### The PPO Problem

Standard Proximal Policy Optimization (PPO) requires training **two networks** of equal size:

- **Actor (Policy)**: generates responses or actions
- **Critic (Value Network)**: evaluates the quality of each state

In the era of 70B+ parameter language models, maintaining a Critic of the same size as the Actor **doubles GPU memory usage** and introduces an additional source of training instability. The Critic itself must be learned from reward signals, and if it is poorly trained, it pollutes the policy gradient, causing the Actor to learn in the wrong direction.

### GRPO's Core Insight

GRPO asks a simple question: *Can we estimate advantage by comparing a group of responses generated for the same input?*

The answer is yes. By sampling G responses from the current policy for the same question, computing their rewards, and normalizing each reward relative to the group's mean and standard deviation, we obtain an advantage estimate **without any Critic network at all**. The group mean serves as the baseline, and the group standard deviation provides adaptive, normalized signal strength.

---

## Algorithm Walkthrough

```
==================================================================
              GRPO: One Training Step (No Critic!)
==================================================================

    ┌──────────────────┐
    │  Input Question  │
    │       q          │
    └────────┬─────────┘
             │
             │  Sample G responses from current policy
             ▼
    ┌──────────────────────────┐
    │  Generate G Responses    │
    │  o_1, o_2, ..., o_G      │
    │  ~ π_θ(· | q)            │
    └──────────┬───────────────┘
               │
               │  Evaluate each with reward model
               ▼
    ┌──────────────────────────┐
    │  Compute Rewards         │
    │  r_1, r_2, ..., r_G      │
    │  reward_model(o_i, q)    │
    └──────────┬───────────────┘
               │
               │  Group-relative normalization
               ▼
    ┌──────────────────────────────────────────────────────┐
    │  Group-Normalized Advantage  (KEY STEP)              │
    │                                                      │
    │  μ  = (1/G) · Σ r_j                                  │
    │  σ  = √( (1/G) · Σ(r_j - μ)² ) + ε                  │
    │  A_i = (r_i - μ) / σ                                 │
    │                                                      │
    │  ★ NO CRITIC NETWORK NEEDED ★                        │
    └──────────────────────┬───────────────────────────────┘
                           │
                           │  Importance sampling ratio
                           ▼
    ┌──────────────────────────────────────────────────────┐
    │  PPO Clip Update + KL Regularization                 │
    │                                                      │
    │  ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)                   │
    │  L = E[ min(ρ·A, clip(ρ,1-ε,1+ε)·A) ]                │
    │      - β · D_KL(π_θ || π_ref)                        │
    └──────────────────────┬───────────────────────────────┘
                           │
                           ▼
                    Update θ → θ_new
```

## Mathematical Formulation

### Advantage Estimation (Group-Relative)

Given G responses for question q, compute scalar rewards r_1, ..., r_G:

```
μ   = (1/G) · Σ_{j=1}^{G} r_j                    -- group mean (baseline)
σ   = √( (1/G) · Σ_{j=1}^{G} (r_j - μ)² ) + ε    -- group standard deviation
A_i = (r_i - μ) / σ                                -- normalized advantage
```

- The group mean μ replaces the Critic's baseline V(s)
- The group std σ normalizes the advantage signal: amplifies gradients when responses differ widely, dampens them when responses are similar
- The small epsilon prevents division by zero

### Policy Objective with PPO Clipping

```
ρ_i(θ) = π_θ(o_i | q) / π_θ_old(o_i | q)           -- importance sampling ratio

L_GRPO(θ) = E[  min( ρ_i(θ) · A_i,
                      clip(ρ_i(θ), 1 - ε, 1 + ε) · A_i ) ]
            - β · D_KL( π_θ || π_ref )
```

Where:

- **A_i**: group-normalized advantage for response i
- **ε**: clip parameter (typically 0.2) that prevents overly large policy updates
- **β**: KL penalty coefficient that keeps the policy close to the reference
- **D_KL**: KL divergence between the current policy and the reference policy

---

## PPO vs GRPO Comparison

| Dimension | PPO | GRPO |
|---|---|---|
| Critic Network | Required (same size as Actor) | Not needed |
| Advantage Source | Critic value function V(s) | Group ranking (mean + std normalization) |
| Memory Footprint | ~2× Actor model | ~1.2× Actor model |
| Training Stability | Sensitive to Critic accuracy | More stable, no auxiliary network |
| Best Use Case | General RL tasks | Verifiable rewards (math, code, QA) |
| Baseline | Learned V(s) | Group mean μ |
| Signal Normalization | Fixed or learned | Auto-scaling via group std σ |

---

## Python Code Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np


def group_normalize(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute group-normalized advantages.

    For each group of responses, subtract the group mean and divide
    by the group standard deviation to obtain advantage estimates
    without requiring a critic network.

    Args:
        rewards:   Shape (G,) scalar reward for each response in the group
        eps:       Small constant to prevent division by zero

    Returns:
        advantages: Shape (G,) normalized advantage for each response
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
    Compute the PPO-clip surrogate loss.

    Clamps the importance ratio to prevent excessively large policy updates.

    Args:
        policy_log_probs: Shape (G,) log-probability under current policy
        old_log_probs:    Shape (G,) log-probability under old policy
        advantages:       Shape (G,) group-normalized advantages
        epsilon:          Clip threshold (default 0.2)

    Returns:
        loss_scalar: Mean clipped loss (to be maximized before negation)
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)

    surrogate_unclipped = ratio * advantages
    surrogate_clipped = clipped_ratio * advantages

    # Take the minimum to create a conservative update
    loss = torch.min(surrogate_unclipped, surrogate_clipped)
    return loss.mean()


def kl_penalty(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between current policy and reference policy.

    This penalty prevents the policy from drifting too far from the
    original supervised-fine-tuned model.

    Args:
        policy_logits: Shape (G, vocab_size) current model logits
        ref_logits:    Shape (G, vocab_size) reference model logits

    Returns:
        kl_loss: Scalar KL divergence value
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
    Full GRPO loss function combining group normalization, PPO clipping,
    and KL regularization.

    Args:
        policy_logits: Shape (G, vocab_size) current policy logits
        old_logits:    Shape (G, vocab_size) logits from when samples were drawn
        rewards:       Shape (G,) scalar reward per response
        beta:          KL penalty coefficient
        ref_logits:    Shape (G, vocab_size) reference policy logits (optional)
        epsilon:       PPO clip parameter (default 0.2)

    Returns:
        loss: Scalar loss value (negated for minimization with standard optimizer)
    """
    # --- Step 1: Compute log-probabilities under current and old policy ---
    policy_log_probs = F.log_softmax(policy_logits, dim=-1).sum(dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1).sum(dim=-1)

    # --- Step 2: Group-normalized advantage (no critic!) ---
    advantages = group_normalize(rewards)

    # --- Step 3: PPO-clip objective ---
    clip_objective = ppo_clipped_loss(
        policy_log_probs, old_log_probs, advantages, epsilon
    )

    # --- Step 4: KL regularization (optional but recommended) ---
    kl = torch.tensor(0.0, device=policy_logits.device)
    if ref_logits is not None:
        kl = kl_penalty(policy_logits, ref_logits)

    # --- Step 5: Combine objectives (negate for standard minimization) ---
    total_loss = -(clip_objective - beta * kl)
    return total_loss


# ------------------------------------------------------------------
# Example usage with dummy data to verify the implementation
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    G = 8        # Number of responses per group
    V = 1000     # Vocabulary size (simulated)

    # Simulated dummy data
    policy_logits = torch.randn(G, V)
    old_logits = torch.randn(G, V)
    ref_logits = torch.randn(G, V)
    rewards = torch.tensor([0.9, 0.3, 0.7, 0.5, 0.1, 0.8, 0.2, 0.6])

    beta = 0.01  # KL penalty coefficient

    loss = grpo_loss(
        policy_logits=policy_logits,
        old_logits=old_logits,
        rewards=rewards,
        beta=beta,
        ref_logits=ref_logits,
        epsilon=0.2,
    )

    print(f"GRPO Loss: {loss.item():.6f}")

    # Gradient verification: backward pass works
    loss.backward()
    print("Backward pass successful -- implementation is differentiable.")
```

---

## Deep Dive

### 1. Why Does Group-Normalized Advantage Work?

GRPO replaces the learned value function V(s) with a **sampled baseline** (the group mean). This matters for two reasons:

First, the group mean μ is an **unbiased estimator** of the expected reward under the current policy. In classical policy gradient theory, any baseline b subtracted from the reward does not introduce bias into the gradient as long as b does not depend on the action. Since μ is computed from the entire group before assigning advantages, it satisfies this condition.

Second, dividing by the group standard deviation σ provides **automatic reward scaling**. In early training, the policy generates highly diverse responses with widely varying rewards, so σ is large and advantages are dampened -- this prevents unstable, oversized gradient steps. As training progresses and the policy improves, responses become more similar, σ shrinks, and advantages grow larger to provide finer-grained learning signal.

| Response quality in group | σ value | Effect on advantage |
|---|---|---|
| All responses similar | Small σ | Large advantages, strong signal |
| Responses very different | Large σ | Dampened advantages, cautious update |
| All equally poor | μ low, small σ | Strong push away from bad responses |
| All equally excellent | μ high, small σ | Strong push toward good responses |

### 2. Limitations and Edge Cases

GRPO is elegant but has known failure modes:

**Zero-sum within group:** GRPO's advantage estimate is relative, not absolute. If all G responses are terrible, they still sum to zero advantage. The group mean will be low but the best response gets a positive advantage and is reinforced despite being objectively poor. Solutions include incorporating an absolute reward threshold or mixing in supervised fine-tuning data.

**Reward noise sensitivity:** If the reward model is noisy, the ranking within a group may be unreliable. When the reward difference between two responses is smaller than the reward noise, GRPO may push the policy in the wrong direction. Using a larger group size G helps by averaging out noise in the mean, but does not fix pairwise ranking errors.

**Not suited for multi-step environments:** GRPO was designed for single-turn generation tasks where the reward is available after the full response. For sequential RL environments where reward comes at the end of many steps, GRPO's per-group normalization does not propagate advantage across time steps. A critic or GAE-style advantage is still needed in those settings.

### 3. DeepSeek-R1's Innovations on Top of GRPO

DeepSeek-R1 applied GRPO at an unprecedented scale and introduced several practical enhancements:

**Rule-based verifiable rewards:** Instead of relying solely on a learned reward model, DeepSeek-R1 used deterministic reward signals for math and code problems: Does the code compile? Does it pass the unit test? Does the math answer match the ground truth? This eliminates reward model bias entirely for these domains.

**Format-based rewards:** The reward also checked structural properties of the reasoning trace: Does the response include a clear "thought" section followed by a final answer? Is the reasoning process well-organized? This encourages the emergence of reasoning capabilities without explicit chain-of-thought supervision.

**Warm-start with SFT and large group sizes:** DeepSeek-R1 first warmed up the policy with supervised fine-tuning data, then gradually transitioned to pure GRPO. They also used large group sizes (G = 16 or more) during training, which provides more reliable mean and variance estimates for the advantage calculation.

---

## Further Reading

- **DeepSeekMath** (original GRPO paper): https://arxiv.org/abs/2402.03300
- **DeepSeek-R1** (GRPO at scale): https://arxiv.org/abs/2501.12948
- **Proximal Policy Optimization** (PPO paper): https://arxiv.org/abs/1707.06347
- **Generalized Advantage Estimation** (GAE paper): https://arxiv.org/abs/1506.02438

---

_Prev: None  |  Next: [Day 02 -- Mixture of Experts](02-mixture-of-experts.md)_
