---
date: "2026-04-03"
difficulty: "Advanced"
category: "Inference Acceleration"
---

# Day 03: Speculative Decoding -- Lossless Inference Speedup

> **Watch the animation**: ![Speculative Decoding Animation](../gifs/03-speculative-decoding.gif)

---

## One-Line Summary

Speculative decoding uses a small draft model to predict K candidate tokens in parallel, then uses the large target model to verify all K tokens in a single forward pass, accepting each token with probability min(1, p/q) -- achieving 2x to 4x inference speedup with mathematically guaranteed identical output distribution to the target model.

---

## Why Do We Need Speculative Decoding?

### The Autoregressive Bottleneck

Standard autoregressive generation produces exactly one token per forward pass of the model, regardless of how simple or predictable that token is:

```
Standard decoding:
  "The" → [forward pass: 10B FLOPs] → "cat"
  "cat"  → [forward pass: 10B FLOPs] → "sat"
  "sat"  → [forward pass: 10B FLOPs] → "on"
  "on"   → [forward pass: 10B FLOPs] → "the"

  4 tokens = 4 forward passes = 4 × 10B FLOPs = 40B FLOPs total
```

This is fundamentally memory-bandwidth bound, not compute-bound. The model's weights must be streamed from GPU memory for every single token, but the actual matrix multiplication on one token's representation is tiny. GPUs sit idle most of the time, waiting for weights to arrive.

### Speculative Decoding's Core Insight

Speculative decoding asks: *Can a cheap draft model guess what the target model will produce, and only pay for the target model's compute once to check all guesses at once?*

The answer is yes, and the key is a verification algorithm that guarantees the output distribution is exactly identical to what running the target model alone would produce:

```
Speculative decoding (draft K=3, target verifies):
  Draft (130M): "The" → "cat" → "sat" → "on"   [3 forward passes of small model]
  Target (10B):   "The" + 3 tokens → verify all in 1 pass
  Result: 3 tokens with 1 big forward pass instead of 3
  Speedup: ~2-3x with ZERO quality loss

  Tokens produced per target forward pass:
    Standard:  always 1
    SpecDec:   0.5 to K+1 (average ≈ 2-4 depending on draft quality)
```

---

## Algorithm Walkthrough

```
==================================================================
          Speculative Decoding: Draft + Verify Loop
==================================================================

Step 1: Generate K draft tokens autoregressively
────────────────────────────────────────────────

  ┌──────────────────┐
  │  Current context │──► Draft model generates x_1, x_2, ..., x_K
  │  x_1, ..., x_t   │    (K forward passes of SMALL model)
  └──────────────────┘


Step 2: Verify all K+1 candidates in ONE target forward pass
─────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────┐
  │  Target model inputs: x_1, ..., x_t, x_1, ..., x_K   │
  │                                                      │
  │  Target computes: q(x_{t+1}), q(x_{t+2}), ...        │
  │                    q(x_{t+K}), q(x_{t+K+1})         │
  │  (1 forward pass of LARGE model produces K+1 probs) │
  └─────────────┬────────────────────────────────────────┘
                │
                ▼


Step 3: Parallel verification with acceptance probability
─────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  For each position i = 1 to K:                              │
  │                                                             │
  │  Draft prob:  r_i = P_draft(x_i | x_1, ..., x_{i-1})       │
  │  Target prob: q_i = P_target(x_i | x_1, ..., x_{i-1})      │
  │                                                             │
  │  Acceptance: α_i = min(1, q_i / r_i)                        │
  │                                                             │
  │  Draw u ~ Uniform(0, 1)                                    │
  │  If u < α_i:                                               │
  │    ACCEPT x_i ✓                                             │
  │    Continue to next position                                │
  │  Else:                                                      │
  │    REJECT x_i ✗                                             │
  │    Replace with sample from adjusted distribution           │
  │    Break the verification loop                              │
  │                                                             │
  │  If ALL K accepted, sample 1 more from q(x_{t+K+1})         │
  └─────────────┬───────────────────────────────────────────────┘
                │
                ▼
     Append accepted tokens to output sequence
     Repeat from Step 1 with updated context


Step 4: Rejection sampling (corrected distribution)
────────────────────────────────────────────────────

  If position j is rejected:
    Compute adjusted distribution:
      p_adjusted(x) = max(0, q(x) - r(x)) / (1 - Σ_{accepted} acceptance_prob)

    Sample x_{t+j} ~ p_adjusted and append

  This correction is CRITICAL: it ensures the final
  distribution is EXACTLY q, not an approximation.
```

---

## Mathematical Formulation

### Acceptance Probability

For draft token x_i at position i, the draft model assigns probability r_i and the target model assigns probability q_i. The token x_i is accepted with probability:

```
α_i = min(1, q_i / r_i)
```

This formula has three cases:

```
Case 1: q_i > r_i  (target is more confident than draft)
  α_i = 1 → ALWAYS accept ✓
  The draft was actually conservative about a good token.

Case 2: q_i < r_i but q_i ≈ r_i  (target agrees with draft approximately)
  α_i ≈ 1 → Almost always accept ✓
  Small deviation, small risk of rejection.

Case 3: q_i < r_i significantly  (target disagrees with draft)
  α_i < 1 → May reject ✗
  The draft was over-confident about a wrong token.
```

### Why min(1, q/r) Preserves the Exact Distribution

The key theorem: the output of speculative decoding is mathematically identical in distribution to greedy/ancestral sampling from the target model alone.

```
Proof sketch:

P(accept x_i) = α_i = q_i / r_i   (when q_i ≤ r_i)

The overall probability of accepting sequence x_1, ..., x_k AND then
accepting x_{k+1} (via the final sample) is:

  ∏_{i=1}^{k} [r_i · (q_i/r_i)] · q_{k+1}  =  ∏_{i=1}^{k} q_i · q_{k+1}  =  ∏_{i=1}^{k+1} q_i

Which equals the target model's probability. ∎

When a token is rejected, the adjusted distribution p_adjusted is:

  p_adjusted(x) = max(0, q(x) - r(x)) / Z

where Z = 1 - Σ_{x: q(x)≥r(x)} r(x) · α(x) = Σ_x max(0, q(x) - r(x))

This is the "residual mass" of probability that the draft failed to
capture. Sampling from p_adjusted exactly recovers the remaining
probability mass needed for an exact match with q. ∎
```

### Expected Acceptance Rate and Speedup

The expected number of tokens accepted per round is:

```
E[accepted] = Σ_{i=1}^{K} α_i · Π_{j=1}^{i-1} α_j

If we approximate all α_i ≈ α (constant acceptance rate):
  E[accepted] ≈ (1 - α^K) / (1 - α)  for α < 1
  E[accepted] ≈ K  for α ≈ 1

Speedup factor ≈ E[accepted] + 1  (the +1 is the draft model's cost fraction)

Example calculations:
  α = 0.90,  K=4:  E[accepted] ≈ 4.0 → Speedup ≈ 4.1x
  α = 0.80,  K=4:  E[accepted] ≈ 2.9 → Speedup ≈ 3.1x
  α = 0.70,  K=4:  E[accepted] ≈ 2.2 → Speedup ≈ 2.5x
  α = 0.50,  K=4:  E[accepted] ≈ 1.0 → Speedup ≈ 1.4x
  α = 0.95,  K=8:  E[accepted] ≈ 6.7 → Speedup ≈ 6.9x
```

### Optimal K Selection

```
K_optimal depends on acceptance rate α:

  Low quality draft (α ≈ 0.5):  K = 1 to 3  (more tokens = more rejections)
  Medium quality (α ≈ 0.7):     K = 3 to 5  (balance)
  High quality draft (α ≈ 0.9): K = 5 to 10 (longer speculation pays off)

Rule of thumb: K ≈ -log(0.1) / -log(α)  (K that gives 90% chance of at least 1 acceptance)
  α = 0.7 → K ≈ 6.5
  α = 0.8 → K ≈ 10.3
  α = 0.9 → K ≈ 22
  α = 0.95 → K ≈ 45

In practice, K = 3 to 8 is a good range because:
  - Beyond K=8, diminishing returns (rejection probability compounds)
  - Larger K requires more target model KV-cache capacity
  - Communication overhead grows with K
```

---

## Method Comparison

| Dimension | Standard AR | Speculative Decoding | Medusa | Lookahead Decoding | EAGLE |
|---|---|---|---|---|---|
| Speedup | 1x (baseline) | 2-4x | 2-3x | 2-4x | 2-5x |
| Quality loss | None | None (exact) | Small (approximate) | None (exact) | None (exact) |
| Draft model needed | N/A | Yes (separate small model) | No (head-based) | No (n-gram cache) | No (feature-based) |
| Max accepted per step | 1 | K | K heads × depth | N-gram length | K (feature-drafted) |
| Memory overhead | Base model only | Base + draft model | K × vocab heads | N-gram lookup table | Feature layers |
| Training required | N/A | Separate draft model fine-tune | Train K heads on target | No training | Train feature layers |
| Best use case | General, no extra setup | When draft model available | Single-model deployment | Repetitive/templated text | High-accuracy inference |
| Theoretical guarantee | Exact target dist. | Exact target dist. | Approximate | Exact target dist. | Exact target dist. |

---

## Variants of Speculative Decoding

### 1. Classic Speculative Decoding (Chen et al.)

The original approach using a separate small draft model:

```
Architecture:
  Draft model: 125M params (e.g., distilled from target)
  Target model: 7B+ params (original)

  Draft generates K tokens autoregressively → Target verifies

  Pros: Simple, exact, well-studied
  Cons: Requires training/obtaining a draft model
```

### 2. Medusa: Multi-Head Speculative Decoding

Instead of a separate draft model, Medusa adds K additional "heads" to the target model:

```
         ┌─────────────┐
         │  Transformer │
         │   layers     │
         └──────┬───────┘
                │
           ┌────┼────┬────┬────────┐
           ▼    ▼    ▼    ▼        ▼
       Head_0 Head_1 Head_2 Head_3 Head_4
       (orig)  (t+1)  (t+2)  (t+3)  (t+4)

  Each head predicts a future token independently,
  bypassing autoregressive generation.

  Key trade-off:
    - No separate model needed (simpler deployment)
    - Each head is weaker than a full autoregressive model
    - Acceptance rate is lower than a dedicated draft model
    - Quality is approximate, not exact
```

### 3. Lookahead Decoding

Uses an n-gram cache to skip tokens that have appeared before:

```
  During generation, maintain a dictionary:
    prefix_ngram → likely_continuation

  When current context matches a cached prefix:
    Directly "look ahead" to the cached continuation
    Verify with target model

  Best for: Repetitive text, templates, code patterns
  Worst for: Novel, unpredictable text
```

### 4. EAGLE: Feature-Based Drafting

Instead of training a separate small model, EAGLE trains draft heads that predict future tokens from intermediate layer representations:

```
  EAGLE insight: Intermediate layers contain useful
  information about future tokens. A small network
  on top of intermediate features can draft accurately
  without a full separate model.

  Architecture:
    Target model forward pass → extract intermediate features
    Draft network (MLP + attention) on top of features → predict next tokens
    Target model verifies the draft

  Key advantages:
    - Higher acceptance rate than Medusa (uses rich intermediate features)
    - Lower memory than separate draft model
    - Exact distribution guarantee
```

---

## Python Code Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np


def speculative_verify(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int, torch.Tensor | None]:
    """
    Verify draft tokens against the target model's distribution.

    This implements the core verification algorithm of speculative
    decoding, which guarantees the output distribution is exactly
    identical to sampling from the target model alone.

    Args:
        draft_tokens: Shape (K,) the tokens generated by the draft model
        draft_probs:  Shape (K,) probability the draft assigned to its own predictions
        target_logits: Shape (K+1, vocab_size) logits from target model for
                       positions t+1 through t+K+1
        temperature:   Sampling temperature (default 1.0)

    Returns:
        accepted_tokens: Tokens accepted in this round
        n_accepted: Number of draft tokens accepted (0 to K)
        replacement_token: If rejection occurred, the token sampled
                          from the adjusted distribution (None if all accepted)
    """
    K = draft_tokens.size(0)

    # Get target probabilities for draft token positions
    # Note: target_logits[0] corresponds to position t+1, etc.
    target_probs = F.softmax(target_logits / temperature, dim=-1)  # (K+1, vocab)

    accepted_tokens = []
    n_accepted = 0
    replacement_token = None

    # --- Verify each draft token in order ---
    for i in range(K):
        r_i = draft_probs[i]           # Draft's probability for its prediction
        q_i = target_probs[i, draft_tokens[i]]  # Target's probability for draft's token

        # Acceptance probability
        alpha_i = min(1.0, q_i / r_i)

        if torch.rand(1).item() < alpha_i:
            # ACCEPT this token
            accepted_tokens.append(draft_tokens[i].item())
            n_accepted += 1
        else:
            # REJECT this token
            # Sample from adjusted (residual) distribution
            adjusted = torch.clamp(target_probs[i] - draft_probs[i].unsqueeze(-1).expand_as(target_probs[i]) * 0.0, min=0.0)

            # Actually compute: max(0, q(x) - r(x)) for all x in vocab
            # where r(x) = draft_prob if x == draft_token, else 0
            r_expanded = torch.zeros_like(target_probs[i])
            r_expanded[draft_tokens[i]] = r_i

            adjusted = torch.clamp(target_probs[i] - r_expanded, min=0.0)
            adjusted_sum = adjusted.sum()

            if adjusted_sum > 0:
                adjusted = adjusted / adjusted_sum
                replacement = torch.multinomial(adjusted, 1)
                replacement_token = replacement.item()
            else:
                # Edge case: just sample from target
                replacement = torch.multinomial(target_probs[i], 1)
                replacement_token = replacement.item()
            break

    return torch.tensor(accepted_tokens), n_accepted, replacement_token


def speculative_decoding_step(
    draft_model,
    target_model,
    context: torch.Tensor,
    max_draft_length: int = 4,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """
    Perform one step of speculative decoding.

    The draft model generates max_draft_length candidate tokens
    autoregressively, and the target model verifies them in one pass.

    Args:
        draft_model: Smaller model for drafting (must support forward + return logits)
        target_model: Large model for verification
        context: Shape (1, seq_len) current token sequence
        max_draft_length: Maximum number of draft tokens (K)
        temperature: Sampling temperature

    Returns:
        new_tokens: All tokens appended this round (shape varies: 0 to K+1)
        n_accepted: Number of draft tokens accepted
    """
    draft_tokens_list = []
    draft_probs_list = []

    current = context.clone()

    # --- Phase 1: Draft generation (autoregressive with small model) ---
    with torch.no_grad():
        for _ in range(max_draft_length):
            # Draft model forward pass (small model, fast)
            draft_logits = draft_model(current)  # (1, seq, vocab)
            next_token_logits = draft_logits[0, -1, :]
            next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)

            # Sample a token from draft distribution
            draft_token = torch.multinomial(next_token_probs, 1)
            draft_prob = next_token_probs[draft_token].item()

            draft_tokens_list.append(draft_token.item())
            draft_probs_list.append(draft_prob)

            # Append to draft input for next step
            current = torch.cat([current, draft_token.unsqueeze(0)], dim=-1)

    draft_tokens = torch.tensor(draft_tokens_list, device=context.device)
    draft_probs = torch.tensor(draft_probs_list, device=context.device)

    # --- Phase 2: Target verification (one big forward pass) ---
    with torch.no_grad():
        # Target model verifies all positions in one pass
        # Input: context + all K draft tokens
        target_input = torch.cat([context, draft_tokens.unsqueeze(0)], dim=-1)
        target_logits = target_model(target_input)  # (1, seq+K, vocab)

        # Extract logits for draft positions and one extra
        # We need logits at positions: len(context), len(context)+1, ..., len(context)+K
        verify_logits = target_logits[0, -max_draft_length - 1:, :]

    # --- Phase 3: Acceptance/rejection ---
    accepted_tokens, n_accepted, replacement = speculative_verify(
        draft_tokens, draft_probs, verify_logits, temperature
    )

    # Build output tokens
    if replacement is not None:
        # Rejection happened: append accepted + replacement
        new_tokens = torch.cat([accepted_tokens, torch.tensor([replacement], device=context.device)])
    else:
        # All accepted + one bonus token from target
        n_accepted = max_draft_length
        bonus_logits = verify_logits[-1, :]  # Position t+K+1
        bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
        bonus_token = torch.multinomial(bonus_probs, 1)
        new_tokens = torch.cat([accepted_tokens, bonus_token])

    return new_tokens, n_accepted


class MockModel:
    """
    A mock model for demonstrating speculative decoding.
    In production, these would be actual transformer models.
    """

    def __init__(self, vocab_size: int, hidden_size: int, is_large: bool = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.is_large = is_large
        # Simple linear projection to simulate vocabulary logits
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

        if is_large:
            print(f"  Target model: {self.vocab_size} vocab, {hidden_size} hidden (LARGE)")
        else:
            print(f"  Draft model:  {self.vocab_size} vocab, {hidden_size} hidden (small)")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Simulated forward pass - just projects token embeddings to logits.

        Args:
            token_ids: Shape (batch, seq) token IDs

        Returns:
            logits: Shape (batch, seq, vocab_size) simulated logits
        """
        batch, seq = token_ids.shape
        # Create deterministic but non-trivial logits based on token IDs
        embeddings = token_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).float()
        # Add some structure: later tokens influence later positions
        positional_bias = torch.arange(seq, device=token_ids.device).unsqueeze(0).unsqueeze(-1)
        features = embeddings + 0.01 * positional_bias
        logits = self.proj(features)
        return logits

    def __call__(self, x):
        return self.forward(x)


# ------------------------------------------------------------------
# Example usage: comparing standard vs speculative decoding
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = 1000
    context_tokens = torch.tensor([[1, 2, 3, 4, 5]])

    # Create mock models with different capacities
    print("Initializing models...")
    draft_model = MockModel(vocab_size, hidden_size=64, is_large=False)
    target_model = MockModel(vocab_size, hidden_size=256, is_large=True)
    print()

    K = 4
    n_rounds = 10

    # --- Run speculative decoding ---
    print(f"Running speculative decoding (K={K}, rounds={n_rounds})...")
    print("=" * 60)

    context = context_tokens.clone()
    all_spec_tokens = []
    total_draft = 0
    total_accepted = 0

    for round_idx in range(n_rounds):
        new_tokens, n_accepted = speculative_decoding_step(
            draft_model, target_model, context,
            max_draft_length=K, temperature=1.0
        )
        context = torch.cat([context, new_tokens.unsqueeze(0)], dim=-1)
        all_spec_tokens.extend(new_tokens.tolist())

        total_draft += K
        total_accepted += n_accepted

        print(f"  Round {round_idx+1:2d}: drafted {K}, adopted {n_accepted}+"
              f"{1 if new_tokens.shape[0] > n_accepted else 0} = "
              f"{new_tokens.shape[0]} tokens  "
              f"(tokens so far: {len(all_spec_tokens)})")

    accept_rate = total_accepted / total_draft
    tokens_per_target_pass = len(all_spec_tokens) / n_rounds

    print()
    print("Results:")
    print(f"  Total draft tokens: {total_draft}")
    print(f"  Total accepted:     {total_accepted}")
    print(f"  Acceptance rate:    {accept_rate:.1%}")
    print(f"  Total tokens generated: {len(all_spec_tokens)}")
    print(f"  Target model calls:   {n_rounds} "
          f"(standard would need {len(all_spec_tokens)} calls)")
    print(f"  Effective speedup:    {tokens_per_target_pass:.2f}x "
          f"(tokens per target forward pass)")
    print()

    # --- Compare: standard autoregressive (simulated) ---
    print("Comparison: standard AR would require:")
    print(f"  {len(all_spec_tokens)} target model forward passes")
    print(f"  vs. {n_rounds} target passes + {total_draft} draft passes")
    print(f"  (draft model is ~{256/64:.0f}x smaller/faster)")
    draft_cost = total_draft / (256/64)
    total_cost_equiv = n_rounds + draft_cost
    print(f"  Equivalent cost ratio: {n_rounds / total_cost_equiv:.2f}x theoretical speedup")
```

---

## Deep Dive

### 1. The Fundamental Theorem: Why Speculative Decoding is Lossless

This is the most important result in speculative decoding, and it is often misunderstood. The rejection sampling mechanism is not a heuristic -- it is an exact mathematical construction that guarantees the output distribution matches the target model perfectly.

```
Target distribution for next token: q(x)
Draft distribution for next token:  r(x)

Naive approach (WRONG):
  Just accept the draft token always.
  Result: output follows r(x), not q(x) → quality loss.

Better approach (WRONG):
  Accept draft token if q(x) > threshold.
  Result: output is a biased subset → quality loss.

Correct approach (EXACT):
  Accept with α(x) = min(1, q(x)/r(x)).
  If rejected, sample from p_adjusted(x) = max(0, q(x)-r(x)) / Z.

  Result: output follows q(x) EXACTLY.
  This is not an approximation -- it is a mathematical identity.
```

The proof is essentially this: at every step, the algorithm either accepts the draft token (contributing r(x) · q(x)/r(x) = q(x) to the output) or rejects it and samples from the residual distribution (contributing the remaining probability mass that sums to the correct q(x)). The decomposition q(x) = q(x) · 1 = q(x) · [r(x)/r(x)] works out because the rejection sampling exactly captures the gap between what the draft provides and what the target requires.

### 2. Draft Model Design: What Makes a Good Draft Model?

The draft model does NOT need to be as accurate as the target model. It needs to be:

1. **Fast**: Typically 3-10x faster than the target model. If the draft is too slow, the time spent generating K candidates exceeds the time saved by verifying them in one pass.

2. **Diverse**: The draft must produce plausible tokens, even if not the exact tokens the target would choose. High acceptance rate is more important than top-1 accuracy.

3. **Well-calibrated**: The draft's probability estimates should roughly track the target's. If the draft assigns 99% probability to a token that the target assigns 0.1%, the acceptance rate will be poor.

Common draft model choices:

```
Draft strategy          Acceptance rate  Setup complexity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Distilled small model   High (70-90%)    Moderate (SFT distillation)
Same model, fewer layers Medium (50-70%)  Low (just use subset)
n-gram cache            Variable (20-80%) None (just cache)
Medusa heads            Medium (40-60%)  Low (train heads only)
EAGLE features          High (60-85%)    Moderate (train feature net)
```

### 3. Temperature and Its Effect on Acceptance Rate

Temperature significantly affects the acceptance rate:

```
At low temperature (T = 0.1):
  Both models concentrate probability on the top token.
  If they agree on the top token → α ≈ 1 (always accept).
  If they disagree → α ≈ 0 (almost always reject).
  Binary outcome, high variance in speedup.

At high temperature (T = 2.0):
  Both distributions are flatter.
  q(x)/r(x) ratios are closer to 1 across tokens.
  More stable but lower acceptance of specific tokens.

Sweet spot: T = 0.5 to T = 1.0
  Enough concentration for meaningful predictions,
  but enough spread for reasonable acceptance rates.
```

### 4. Speculative Decoding in Practice: Engineering Challenges

**KV Cache management:** The target model's KV cache must be populated for K positions simultaneously, but some positions will be rejected. Efficient implementations handle this by:

```
  1. Fill KV cache for all K+1 positions
  2. Accept the first n_accepted positions
  3. Evict the rejected positions from KV cache
  4. The bonus token (position K+1) is always kept

  This requires careful cache management to avoid
  recomputing the accepted positions' KV states.
```

**Batched speculative decoding:** For serving multiple requests, draft tokens can be batched across requests, and the target model verifies all batches in parallel:

```
  Request 1: [draft tokens x3]
  Request 2: [draft tokens x5]
  Request 3: [draft tokens x2]
  ... padded to max K, verified in one target batch
```

**Asynchronous drafting:** Advanced implementations run the draft model continuously in the background as the target model processes previous rounds, hiding the draft's latency entirely.

### 5. When Should You NOT Use Speculative Decoding?

Speculative decoding is not universally beneficial:

```
❌ Low acceptance rate scenario:
   Draft model is very different from target (e.g., different domain)
   Acceptance rate drops below ~40% → speedup < 1.5x → not worth it
   The draft model's compute + communication overhead exceeds savings.

❌ Very short sequences:
   If generating only 5-10 tokens, overhead dominates.
   Speculative decoding shines at 100+ tokens.

❌ Compute-bound workloads:
   If the target model is already compute-saturated (large batch serving),
   memory bandwidth is not the bottleneck, so speculative decoding helps less.

❌ Non-autoregressive generation:
   Speculative decoding relies on the autoregressive structure.
   For masked LM or diffusion, different acceleration techniques are needed.
```

---

## Further Reading

- **Fast Inference from Transformers via Speculative Decoding** (Chen et al., 2023): https://arxiv.org/abs/2211.17192
- **Medusa: Simple LLM Inference Acceleration Framework** (Cai et al., 2024): https://arxiv.org/abs/2401.10774
- **Lookahead Decoding** (Fu et al., 2024): https://arxiv.org/abs/2402.02057
- **EAGLE: Speculative Sampling with Feature-Based Drafting** (Li et al., 2024): https://arxiv.org/abs/2406.16858
- **Speculative Decoding: A Survey** (comprehensive review): https://arxiv.org/abs/2409.15385

---

_Prev: [Day 02 -- Mixture of Experts](02-mixture-of-experts.md)  |  Next: [Day 04 -- Test-Time Compute](04-test-time-compute.md)_