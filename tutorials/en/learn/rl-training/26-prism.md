---
date: "2026-05-03"
difficulty: "Advanced"
category: "RL Training / Multimodal"
paper: "2604.28123"
---

# Day 26: PRISM — Closing the Gap Between SFT and RLVR

> **Watch the animation**: ![Animation](https://playitcooool.github.io/advanced-ai-daily/gifs/26-prism.gif)

## One-Line Summary

PRISM inserts a black-box on-policy distillation stage between SFT and RLVR, using a MoE discriminator to align the policy back to the supervision distribution before reinforcement learning — mitigating the distributional drift that compounds across post-training stages.

---

## Why This Matters

### The Standard Post-Training Recipe Has a Drift Problem

The prevailing post-training recipe for large multimodal models (LMMs) is:

1. **SFT** — supervised fine-tuning on curated demonstrations
2. **RLVR** — reinforcement learning with verifiable rewards

SFT teaches the model *what good responses look like* through imitation. RLVR then trains the model to *maximize reward signals* beyond simple imitation. Together they produce capable assistants.

But each stage shifts the model's distribution. SFT introduces drift away from the original pretrained model. Then RLVR trains on top of that drifted distribution, compounding the problem. In multimodal reasoning, perception errors and reasoning failures have *distinct* drift patterns that compound during RL.

### Why Drift Accumulates

- SFT distributions are curated and limited — they can't cover all failure modes
- RL explores on top of the SFT-fine-tuned policy, not the original model
- In multimodal settings, visual grounding errors and reasoning errors compound
- Each RL step compounds the distributional gap further from the original model's knowledge

### The Key Insight: Align Before You RL

Instead of letting drift compound, PRISM adds an explicit **distribution-alignment stage** between SFT and RLVR. This stage uses **on-policy distillation** to push the policy back toward the supervision distribution before RL begins — creating a cleaner foundation for RLVR to build on.

The result: RLVR starts from a better-aligned policy, so it learns more efficiently and avoids the compounding drift problem.

---

## Core Insight

### The Three-Stage Pipeline

```
SFT  →  PRISM Alignment  →  RLVR
```

**Stage 1: SFT** — Initialize with broad behavioral cloning from 1.26M public demonstrations.

**Stage 2: PRISM Alignment** — The core contribution. On-policy distillation that aligns the SFT-drifted policy back toward the supervision distribution without access to teacher logits.

**Stage 3: RLVR** — Reinforcement learning with verifiable rewards on the aligned policy.

### The Black-Box On-Policy Distillation Game

PRISM casts alignment as a **response-level adversarial game** between:

- **Policy** — the model being aligned
- **MoE Discriminator** — a mixture-of-experts with dedicated perception and reasoning experts

The discriminator provides **disentangled corrective signals** — it can separately identify whether the policy's error comes from visual grounding failures or from reasoning failures. This matters because these two failure modes compound in different ways during RL.

No teacher logits needed — the game is played at the response level, making it black-box and widely applicable.

### Why Not Just More SFT or Just Better RL?

- More SFT amplifies drift further, not corrects it
- Standard RL optimizes on the drifted distribution — the compounding problem continues
- PRISM specifically addresses the *distribution gap* before RL begins, not during or after
- The MoE discriminator gives structured, interpretable signals about *where* the drift originates

### Experimental Results

On Qwen3-VL across multiple RL algorithms (GRPO, DAPO, GSPO):

| Model | Accuracy Gain over SFT→RLVR |
|-------|---------------------------|
| Qwen3-VL 4B | +4.4 points |
| Qwen3-VL 8B | +6.0 points |

Significant gains across diverse multimodal benchmarks, confirming that closing the distribution gap early leads to more efficient RL.

---

## Quick Quiz

**Q1:** What is the core problem PRISM addresses?

<details>
<summary>Answer</summary>

**A:** SFT introduces distributional drift from the original pretrained model, and RLVR compounds this drift by training on top of the drifted policy. PRISM inserts an alignment stage before RLVR to close this gap.
</details>

**Q2:** Why does the MoE discriminator help?

<details>
<summary>Answer</summary>

**A:** It provides disentangled corrective signals — separate signals for perception errors vs. reasoning errors, so the alignment can target the specific type of drift rather than treating all errors uniformly.
</details>

**Q3:** Why is "black-box" distillation important?

<details>
<summary>Answer</summary>

**A:** It doesn't require access to teacher logits or internal activations — only response-level feedback. This makes it widely applicable even when the teacher model's internal states aren't available.
</details>

---

## How It Connects to Prior Days

- **Day 24 (Exploration Hacking)** — Both papers study failure modes in RL post-training; PRISM is one approach to make RL more robust against distribution drift
- **Day 25 (Synthetic Computers)** — Both target agent training; PRISM improves the post-training pipeline for multimodal agents
- **Day 10 (SRPO)** — Unified GRPO + self-distillation; PRISM similarly bridges two post-training stages but for multimodal RL

---

## Resource Links

- [Paper (arXiv 2604.28123)](https://arxiv.org/abs/2604.28123v1)
- [GitHub (XIAO4579/PRISM)](https://github.com/XIAO4579/PRISM)