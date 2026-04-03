---
date: "2026-04-03"
difficulty: "Advanced"
category: "Model Architecture"
---

# Day 02: Mixture of Experts (MoE) -- Scaling Models Without Scaling Cost

> **Watch the animation**: ![MoE Animation](../gifs/02-mixture-of-experts.gif)

---

## One-Line Summary

Mixture of Experts replaces every feed-forward network (FFN) layer with a set of K independent expert networks and a trainable Router that selects only the top-k experts per token, enabling models to scale to trillions of parameters while keeping per-token computation constant and roughly equal to a dense model.

---

## Why Do We Need MoE?

### The Compute Scaling Wall

Traditional dense transformer architecture scales both parameter count AND per-token compute together: doubling parameters means doubling multiply-accumulate operations for every token. A 300-billion-parameter dense model requires 300 billion FLOPs per token per layer -- this is prohibitively expensive for both training and inference.

### MoE's Core Insight

MoE asks: *Can we have trillions of total parameters but only activate a small fraction of them for each token?*

The answer is yes, by exploiting a fundamental observation: not every token needs every neuron, not every sentence needs every concept, and not every problem requires every skill. By maintaining a pool of specialized "expert" networks and routing each token to only the most relevant experts, a model can store vast knowledge across many experts while computing each token with only a tiny subset of the total parameters.

---

## Algorithm Walkthrough

```
==================================================================
              MoE Forward Pass -- Per Token
==================================================================

     ┌─────────────────────────────┐
     │     Input Token h           │
     │  Shape: (d_model,)          │
     └─────────────┬───────────────┘
                   │
                   │  h goes to router
                   ▼
     ┌─────────────────────────────────────────────┐
     │              Router Network                 │
     │                                             │
     │  gate_logits = h · W_router                 │
     │  Shape: (gate_logits_dim = num_experts)     │
     └─────────────┬───────────────────────────────┘
                   │
                   │  Top-K selection with noise & load balancing
                   ▼
     ┌────────────────────────────────────────────────────────────┐
     │              Top-K Routing + Load Balancing                │
     │                                                            │
     │  P = softmax(gate_logits)          -- routing probs        │
     │  Top-K indices: {e_1, e_2, ..., e_k}                       │
     │  Expert weights: w_1, w_2, ..., w_k                        │
     │                                                            │
     │  ┌──────────────────────────────┐                          │
     │  │  Load Balance Auxiliary Loss │                          │
     │  │                              │                          │
     │  │  f_i = fraction of tokens    │                          │
     │  │        routed to expert i    │                          │
     │  │  P_i = mean routing prob     │                          │
     │  │        for expert i          │                          │
     │  │                              │                          │
     │  │  L_aux = α · N · Σ f_i·P_i   │                          │
     │  │                              │                          │
     │  └──────────────────────────────┘                          │
     └─────────────┬──────────────────────────────────────────────┘
                   │
                   │  Dispatch tokens to selected experts
                   ▼
     ┌──────────────────────────────────────────┐
     │          Expert Computation              │
     │                                          │
     │  For each selected expert e_j:           │
     │    y_j = w_j · Expert_{e_j}(h)           │
     │                                          │
     │  Note: Each expert is a standalone FFN   │
     │  with hidden dimension d_ff              │
     └─────────────┬────────────────────────────┘
                   │
                   │  Combine weighted expert outputs
                   ▼
     ┌──────────────────────────────────────────┐
     │          Output Combination              │
     │                                          │
     │  y = Σ_{j=1}^{k} w_j · Expert_{e_j}(h)   │
     │                                          │
     │  ★ Only k experts activated per token     │
     │  ★ Total compute ≈ k/k_total of dense     │
     └─────────────┬────────────────────────────┘
                   ▼
            Output y (d_model,)
```

---

## Mathematical Formulation

### Router and Top-K Selection

For each input token h, the router computes selection scores across all N experts:

```
gate_logits = h · W_router                        -- linear projection to N scores
where W_router has shape (d_model, N)

P = softmax(gate_logits / temperature)             -- routing probabilities
P_i for i in {1, ..., N},  Σ P_i = 1

Top-K selection:
  Select indices T = top-k(P) = {e_1, e_2, ..., e_k}
  Normalize selected weights: w_j = P_{e_j} / Σ_{m=1}^{k} P_{e_m}
```

The temperature parameter controls routing "sharpness": lower temperature gives more confident routing (one expert dominates), higher temperature gives softer routing (multiple experts contribute more evenly).

### Expert Output Computation

```
y = Σ_{j=1}^{k} w_j · Expert_{e_j}(h)

where each Expert_i(h) = ReLU(h · W_gate_i + b_gate_i) · W_down_i

Total forward FLOPs per token ≈ k · (2 · d_model · d_ff)  -- NOT N · d_ff
```

### Load Balancing Auxiliary Loss

Without an explicit load balance loss, routing tends to degenerate: a few "popular" experts receive most tokens while others remain underutilized (expert collapse):

```
f_i = (# tokens routed to expert i) / (total tokens)     -- empirical load
P_i = mean(P_i over all tokens in the batch)              -- mean routing prob

L_aux = α · N · Σ_{i=1}^{N} f_i · P_i

where α is a weighting coefficient (typically 0.01)
```

The loss term f_i · P_i is minimized when both f_i and P_i are uniform (1/N). The N multiplier normalizes so the expected loss under uniform routing equals 1.

### Capacity Factor and Token Dropping

In practice, experts are allocated a fixed "capacity" to enable batched GPU computation:

```
capacity = capacity_factor · (total_tokens / num_experts)

if tokens_for_expert_i > capacity:
    -- drop excess tokens (they pass through as identity)
    -- or use overflow buffer
```

A capacity factor of 1.25 means each expert can handle 25% more than its average expected load. Excess tokens are either dropped or routed to a fallback.

---

## Dense vs MoE Comparison

| Dimension | Dense Transformer | MoE Transformer |
|---|---|---|
| Parameters per layer | d_model × d_ff | N × d_model × d_ff (N experts) |
| Active parameters per token | d_model × d_ff | k × d_model × d_ff (k experts) |
| Total parameters | Scales linearly with model size | Scales with N (number of experts) |
| FLOPs per token | Fixed, proportional to all params | k/N of total parameters |
| Memory footprint | Proportional to active params | Proportional to total params |
| Training stability | Well-understood | Sensitive to routing collapse |
| Communication (distributed) | Standard all-reduce | Expert parallelism + all-to-all |
| Real-world examples | LLaMA 3, Mistral 7B | Mixtral 8x7B, DeepSeek-V3 |
| Best trade-off | Simplicity, deployment ease | Massive parameter count at constant compute |

---

## Expert Collapse and Routing Dynamics

### What Is Expert Collapse?

Expert collapse occurs when the router degenerate into always selecting the same 1 or 2 experts (the "rich get richer" problem):

```
Initial state (healthy):
  Expert utilization: [12%, 11%, 13%, 12%, 11%, 12%, 13%, 14%]  ✓ Uniform

After collapse (degenerate):
  Expert utilization:  [95%, 3%, 0%,  1%,  0%,  0%,  1%,  0%]    ✗ Collapsed
```

### Causes and Prevention

| Cause | Mechanism | Prevention |
|---|---|---|
| Positive feedback loop | Popular experts get more gradient updates, become even more popular | Auxiliary load balance loss |
| Insufficient initialization | Random weights cause early routing bias to a few experts | Router initialization, temperature warmup |
| Limited capacity | Popular experts drop tokens, reinforcing routing away from them | Increase capacity factor, random routing |
| Expert specialization | One expert "steals" all easy cases, others atrophy | Noisy Top-K gating, random token forcing |

### Noisy Top-K Gating

Adding noise before the Top-K selection prevents premature convergence:

```
noisy_logits = gate_logits + ε · randn_like(gate_logits)
P = softmax(noisy_logits / temperature)
```

This is equivalent to a Gumbel-Softmax relaxation that maintains exploration among routing decisions during training.

---

## Mixtral 8x7B Architecture Case Study

```
Mixtral 8x7B:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Layers:           32 transformer blocks
  Experts per MoE:  8 (all used via top-2 routing)
  d_model:          4096
  d_ff per expert:  14336
  Active params:    ~13B per token (2 of 8 experts)
  Total params:     ~46.7B
  FLOPs/token:      ≈ 12-13B (dense 7B equivalent)

  Key detail: Uses gated ReLU experts and top-2 routing
  with capacity_factor = 1.0 (no capacity limit)

  Routing: Each token selects exactly 2 experts
  ─────────────────────────────────────────────────

Token h ──→ Router (W: 4096→8) ──→ Top-2 probs
                │
                ├─→ Expert 3: FFN(h) × 0.52
                ├─→ Expert 7: FFN(h) × 0.48
                │
                └─→ y = 0.52·E3(h) + 0.48·E7(h)
```

---

## Python Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Expert(nn.Module):
    """
    A single expert network -- a standard feed-forward block.

    In practice, experts are typically SwiGLU or gated ReLU FFNs
    identical to those used in dense transformers.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the expert FFN gate and down-projection.

        Args:
            x: Shape (batch * seq_len, d_model) input tokens

        Returns:
            output: Shape (batch * seq_len, d_model)
        """
        return self.w_down(F.relu(self.w_gate(x)))


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with Top-K routing, load balancing,
    and auxilliary loss support.

    This implements a simplified version of the Switch
    Transformer / Mixtral-style MoE layer.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
        noise_std: float = 0.0,
    ):
        """
        Initialize the MoE layer.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension per expert
            num_experts: Total number of expert networks
            top_k: Number of experts selected per token
            capacity_factor: Expert capacity multiplier (1.0 = exact average)
            aux_loss_weight: Weight for load balancing auxiliary loss (alpha)
            noise_std: Noise standard deviation for routing (for training)
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.noise_std = noise_std

        # Router: simple linear projection to num_experts scores
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Create independent experts
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Args:
            x: Shape (batch, seq_len, d_model) input tokens

        Returns:
            output: Shape (batch, seq_len, d_model) combined expert output
            aux_loss: Scalar auxiliary loss for load balancing
        """
        batch_size, seq_len, d_model = x.shape

        # Flatten to process all tokens uniformly
        flat_x = x.reshape(-1, d_model)  # (batch * seq_len, d_model)

        # --- Compute routing scores ---
        gate_logits = self.router(flat_x)  # (BT, num_experts)

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # Routing probabilities
        # Apply temperature scaling if needed
        routing_weights = F.softmax(gate_logits, dim=-1)  # (BT, num_experts)

        # --- Load balancing auxiliary loss ---
        aux_loss = self.compute_auxiliary_loss(routing_weights)

        # --- Top-K selection ---
        # Get top-k expert indices and their routing weights
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (BT, top_k)

        # Normalize selected weights so they sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        )

        # --- Dispatch and compute ---
        # Initialize output tensor
        output = torch.zeros_like(flat_x)  # (BT, d_model)

        # Process each expert separately
        # In production, this uses scatter/gather for efficiency
        for expert_idx in range(self.num_experts):
            # Find which tokens select this expert
            selected_mask = (top_k_indices == expert_idx)  # (BT, top_k)
            selected_positions = selected_mask.any(dim=-1)  # (BT,)

            if not selected_positions.any():
                continue

            # Collect tokens for this expert
            expert_input = flat_x[selected_positions]  # (n_tokens_i, d_model)

            # Also collect the weights for these tokens
            # We need to sum weights across multiple top-k positions
            expert_weights = (top_k_weights * selected_mask.float()).sum(
                dim=-1
            )  # (n_tokens_i,)

            # Compute expert output
            expert_output = self.experts[expert_idx](expert_input)

            # Weight by routing probability and scatter back
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            output[selected_positions] += expert_output

        # Reshape back to original dimensions
        output = output.reshape(batch_size, seq_len, d_model)

        return output, aux_loss

    def compute_auxiliary_loss(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the load balancing auxiliary loss.

        This encourages uniform token distribution across experts
        to prevent expert collapse.

        Args:
            routing_weights: Shape (num_tokens, num_experts) routing probs

        Returns:
            aux_loss: Scalar auxiliary loss value
        """
        num_tokens = routing_weights.size(0)
        N = self.num_experts

        # f_i: actual fraction of tokens routed to expert i (based on top-K)
        top_k_probs, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        expert_counts = torch.zeros(N, device=routing_weights.device)
        for i in range(N):
            expert_counts[i] = (top_k_indices == i).float().sum()
        f = expert_counts / (num_tokens * self.top_k)

        # P_i: mean routing probability for expert i
        P = routing_weights.mean(dim=0)  # (num_experts,)

        # Loss: N * sum(f_i * P_i)
        aux_loss = self.aux_loss_weight * N * torch.sum(f * P)

        return aux_loss


class MoETransformerBlock(nn.Module):
    """
    Simplified transformer block with MoE FFN layer.

    Combines attention with the MoE-based FFN to form
    a complete transformer layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int,
        d_ff: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.moe = MoELayer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            aux_loss_weight=aux_loss_weight,
            noise_std=0.1,  # Small noise for training exploration
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections and layernorm.

        Returns:
            output: Shape (batch, seq_len, d_model)
            aux_loss: Scalar MoE auxiliary loss
        """
        # Pre-LN residual attention
        attn_in = self.norm1(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        # Pre-LN residual MoE
        moe_in = self.norm2(x)
        moe_out, aux_loss = self.moe(moe_in)
        x = x + moe_out

        return x, aux_loss


# ------------------------------------------------------------------
# Example usage -- training a small MoE model
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Small mockup configuration
    batch_size = 4
    seq_len = 32
    d_model = 256
    n_heads = 4
    num_experts = 8
    d_ff = 512
    top_k = 2

    # Create a mockup MoE transformer block
    block = MoETransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        num_experts=num_experts,
        d_ff=d_ff,
        top_k=top_k,
        capacity_factor=1.25,
        aux_loss_weight=0.01,
    )

    print(f"MoE Configuration:")
    print(f"  Num experts: {num_experts}")
    print(f"  Top-K routing: {top_k}")
    print(f"  Params per expert: {sum(p.numel() for p in block.moe.experts[0].parameters())/1e6:.2f}M")
    print(f"  Total MoE params: {sum(p.numel() for p in block.moe.parameters())/1e6:.2f}M")
    print(f"  Active params per token: {sum(p.numel() for p in block.moe.experts[0].parameters())/1e6 * top_k:.2f}M")
    print()

    # Random input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"Input shape: {x.shape}")

    # Forward pass
    block.train()
    output, aux_loss = block(x)

    print(f"Output shape: {output.shape}")
    print(f"Auxiliary loss (load balance): {aux_loss.item():.6f}")
    print()

    # Check that we can backprop (including aux loss)
    total_loss = output.sum() + aux_loss
    total_loss.backward()
    print("Backward pass successful -- MoE layer is fully differentiable.")

    # Demonstrate load balance: check token distribution
    # Run a few batches to see if routing is uniform
    expert_counts = torch.zeros(num_experts)
    with torch.no_grad():
        for _ in range(100):
            x = torch.randn(batch_size * seq_len, d_model)
            gate_logits = block.moe.router(x)
            _, indices = torch.topk(gate_logits, top_k, dim=-1)
            for e in range(num_experts):
                expert_counts[e] += (indices == e).float().sum()

    expert_counts /= expert_counts.sum()
    pct = ", ".join(f"{c:.1%}" for c in expert_counts.tolist())
    print(f"Expert utilization (100 batches): {pct}")
    print(f"Std deviation: {expert_counts.std().item():.4f} "
          f"(lower = better balance)")
```

---

## Deep Dive

### 1. Why Doesn't Routing Collapse Always Happen?

The theoretical risk is that the optimizer discovers a "lazy" solution: route everything to one easy-to-train expert and stop learning. In practice, the auxiliary loss prevents this by directly penalizing non-uniform distributions. But the auxiliary loss alone is not enough -- the key insight is that different experts naturally specialize on different types of tokens during training because they receive different gradient signals:

```
Token types → Different experts specialize:
  Math tokens   → Expert 2, Expert 5 (learn arithmetic patterns)
  Code tokens   → Expert 1, Expert 3 (learn syntax structure)
  Prose tokens  → Expert 0, Expert 4, Expert 7 (learn language patterns)
  Mixed tokens  → Expert 6 (generalist / fallback)
```

This specialization emerges naturally because the router learns to match token representations with expert specializations, and each expert's weights are shaped by the specific subset of tokens it receives.

### 2. Expert Parallelism: Distributed Training at Scale

When N experts cannot fit on a single GPU, MoE introduces specialized distributed training patterns:

```
Expert Parallelism (EP):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─GPU 0───┐  ┌─GPU 1───┐  ┌─GPU 2───┐  ┌─GPU 3───┐
│Expert 0 │  │Expert 2 │  │Expert 4 │  │Expert 6 │
│Expert 1 │  │Expert 3 │  │Expert 5 │  │Expert 7 │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └──── all-to-all communication ─────────┘
     (tokens routed to correct GPU for expert processing)

  All-to-all cost grows O(N) with expert count.
  Communication overhead is the primary bottleneck in MoE training.
```

DeepSeek-V3 further optimized this with "DualPipe" overlapping computation and communication, and "Highway Routing" that allows tokens to skip the MoE entirely when no expert is needed.

### 3. MoE Capacity and Token Dropping

In Switch Transformers, a hard capacity limit is enforced per expert to enable static batch sizing. Tokens that overflow an expert's capacity are either:

- **Dropped**: They pass through the MoE layer unchanged and contribute zero auxiliary loss. This is simple but wastes the information in those tokens.
- **Overflow buffered**: Excess tokens are queued and processed in the next opportunity.

The capacity factor is a critical hyperparameter. Too low (1.0) risks excessive token dropping, reducing model quality. Too high (1.5+) wastes GPU memory and compute, defeating MoE's efficiency purpose.

### 4. Dense-Sparse Model Family: Same Architecture, Different Training Regimes

A key advantage of MoE: you can train a dense version of the same architecture by setting top_k = num_experts (all experts active). This allows seamless transfer between training budgets:

```
Training regime:
  Rich budget:  top_k = N (dense) → maximum quality, maximum compute
  Medium:       top_k = 2 to N/4 → good quality, moderate compute
  Tight:        top_k = 1 → sparse activation, lowest compute

Same model weights, same expert specialization,
just different routing sparsity at inference time.
```

### 5. DeepSeek-V3's Advanced MoE Features

DeepSeek-V3 (and V2) introduced several MoE innovations:

- **Shared experts**: In addition to the top-k routed experts, a small number of "shared experts" are activated for ALL tokens. These share the same weights across all routing decisions and capture general-purpose patterns that all tokens benefit from.

- **Multi-token prediction**: DeepSeek-V3's MoE layer also learns to predict multiple future tokens in parallel, effectively getting free compute for the additional predictions since the experts are already activated.

- **Fine-grained experts**: Instead of a few large experts, DeepSeek-V3 uses many smaller experts (256 experts), enabling finer specialization and better routing decisions.

---

## Further Reading

- **Switch Transformers** (Fedus et al., 2021): https://arxiv.org/abs/2101.03961
- **Mixtral of Experts** (Jiang et al., 2024): https://arxiv.org/abs/2401.04088
- **DeepSeek-V3 Technical Report**: https://arxiv.org/abs/2412.19437
- **GShard** (Lepikhin et al., 2021): https://arxiv.org/abs/2006.16668
- **Sparse Mixture of Experts** (survey): https://arxiv.org/abs/2209.00085

---

_Prev: [Day 01 -- GRPO](01-grpo.md)  |  Next: [Day 03 -- Speculative Decoding](03-speculative-decoding.md)_