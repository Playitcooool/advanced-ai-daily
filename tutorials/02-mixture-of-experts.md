# Day 02: MoE -- Mixture of Experts
# 第 02 天: MoE 混合专家架构

> **Date**: 2026-04-03 | **Difficulty**: Advanced | **Category**: LLM Architecture 模型架构

> **Watch the animation**: ![MoE Animation](../gifs/02-moe.gif)

---

## One-Line Summary

MoE lets different parts of a model (Experts) handle what they do best, with a gating network (Router) dynamically routing tokens. The key advantage: **inference only activates a fraction of parameters while training uses all of them.** Mixtral, DeepSeek-V3, and Qwen-MoE all use this architecture.

MoE 让模型的不同部分（Experts）处理各自擅长的问题，通过门控网络（Router）动态分配 Token。核心优势：**推理时只激活部分参数，训练时利用全部参数。**

---

## Dense vs MoE

### Dense FFN

```
Input =====>>> [ FFN: all ~7B params active ] >>> Output
                    全部参数参与计算
```

Every forward pass activates **all** parameters. Inference cost = parameter count.

每次前向传播所有参数都参与，推理成本 = 参数量。

### MoE FFN

```
                   ┌───┐
Input → [ Router │ Gating ] ─┬──→ [ Expert 1 ] ──┐
                   │           ├──→ [ Expert 2 ] ──│
                   │           ├──→ [ Expert 3 ]   │
                   │           ├──→ [ Expert 4 ]   │  → Merge → Output
                   │           ├──→ [ Expert 5 ]   │
                   │           └──→ [ Expert N ] ──┘
                   │
              Only Top-K activated
              只激活 Top-K 个 Expert
```

**Key**: 8 Experts, only 2 active at inference. 8x parameters, ~2x compute.

**关键**: 8 个 Expert，推理只激活 2 个。8 倍参数，约 2 倍计算成本。

---

## Core Components | 核心组件

### 1. Router / Gating Network | 路由器

```python
class MoERouter(torch.nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Learnable routing weights | 可学习的路由权重
        self.gate_w = torch.nn.Parameter(torch.randn(d_model, num_experts) * 0.01)

    def forward(self, x):
        # x: [seq_len, d_model]
        logits = x @ self.gate_w                     # [seq, num_experts]
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_scores, dim=-1)  # normalize weights
        aux_loss = self._load_balancing_loss(logits)       # prevent collapse
        return topk_weights, topk_indices, aux_loss

    def _load_balancing_loss(self, logits):
        \"\"\"
        Auxiliary loss: encourages uniform expert usage
        辅助损失: 强制均匀分配，防止路由坍缩
        Without this, the Router collapses to 1-2 experts.
        \"\"\"
        # Count how often each expert is selected
        # Compare against uniform distribution
        ...
```

### 2. MoE Layer

```python
class MoELayer(torch.nn.Module):
    def __init__(self, d_model, num_experts, hidden_dim, top_k=2):
        super().__init__()
        self.router = MoERouter(d_model, num_experts, top_k)
        # Each expert is a standard FFN | 每个 Expert 就是一个标准 FFN
        self.experts = torch.nn.ModuleList([
            FFN(d_model, hidden_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        weights, indices, aux = self.router(x)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for i in range(x.size(0)):
                expert_out = self.experts[indices[i, k]](x[i:i+1])
                output[i] += weights[i, k] * expert_out.squeeze()
        return output, aux
```

---

## Key Challenges | 关键挑战

### 1. Expert Collapse | 路由坍缩

Without auxiliary loss, the Router gravitates toward 1-2 experts.

**Solutions**: Aux loss (Mixtral), Noise Gating (Switch Transformer), Soft Gating (SoftMoE)

### 2. Communication Bottleneck | 通信瓶颈

The real bottleneck is **not compute but All-to-All communication** between GPUs:

```
GPU 0: tokens → GPU 1's Expert 3, GPU 2's Expert 5
GPU 1: tokens → GPU 0's Expert 1, GPU 3's Expert 7
GPU 2: tokens → GPU 0's Expert 2                ← All-to-All Collectives
GPU 3: tokens → GPU 1's Expert 4, GPU 2's Expert 8

         Network bandwidth is the real bottleneck
              网络带宽才是真正的瓶颈
```

DeepSeek-V3 solves this with grouped experts and Multi-Token Prediction to reduce cross-node traffic.

### 3. Training Instability | 训练不稳定

Top-K selection is a discrete operation -- gradients are hard to propagate.

**Solutions**: Straight-through estimation, Gumbel-Softmax relaxation, Dense-to-sparse initialization.

---

## Model Comparison | 模型对比

| Model | Params | Active | Experts | Top-K | Routing |
|-------|--------|--------|---------|-------|---------|
| Mixtral 8x7B | 46.7B | ~12.9B | 8 | 2 | Top-K + aux |
| Mixtral 8x22B | 141B | ~39B | 8 | 2 | Top-K + aux |
| DeepSeek-V3 | 671B | ~37B | 256 | ~6-8 | Aux-free + grouped |
| Switch Transformer | 1.6T | ~12B | 2048 | 1 | Single expert |

### Why can DeepSeek-V3 use 671B params but only activate 37B?

Many experts (256), narrow hidden dims, grouped routing + Multi-Token Prediction reduces communication overhead.

专家数量多 (256)、每个较窄、分组路由减少通信。

---

## Further Reading

- [DeepSeek-V2/V3](https://arxiv.org/abs/2405.04434)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [SoftMoE](https://arxiv.org/abs/2304.09891)

---

_Prev: [Day 01 - GRPO](01-grpo.md)  |  Next: [Day 03 - Speculative Decoding](03-speculative-decoding.md)_
_上一篇: [Day 01 - GRPO](01-grpo.md)  |  下一篇: [Day 03 - Speculative Decoding](03-speculative-decoding.md)_
