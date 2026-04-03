# MoE - Mixture of Experts 架构深度解析

> **日期**: 2026-04-03 | **难度**: 进阶 | **类别**: LLM 架构 / 高效推理

---

## 一句话总结

MoE 让模型的各个部分（Experts）只处理它们擅长的问题，通过一个门控网络（Router/Gating）动态分配 token 到不同的 Expert。核心优势：**推理时只激活部分参数，但训练时利用了全部参数**。Mixtral、DeepSeek-V3、Qwen-MoE 都基于此架构。

---

## 为什么需要 MoE?

### Dense vs Sparse

**Dense FFN (传统 Transformer):**

```
输入 ───→ [ FFN ] ───→ 输出
       (全部参数激活)
```

每次前向传播，所有参数参与计算。推理成本 = 参数量 × FLOPs。

**MoE FFN:**

```
                  ┌───┐
输入 ──→ [ Router/Gating ] ──┬──→ [ Expert 1 ] ──┐
                  │           ├──→ [ Expert 2 ] ──┤
                  │           ├──→ [ Expert 3 ]    │
                  │           ├──→ [ Expert 4 ]    │  合并 → 输出
                  │           ├──→ [ Expert 5 ]    │
                  │           ├──→ ...             │
                  │           └──→ [ Expert N ] ──┘
                  │
              只选 Top-K 个 Expert
```

**关键**: 8 个 Expert 的 MoE，推理时只激活 2 个。8 倍参数，只有 ~2 倍计算成本。

---

## 架构详解

### 门控机制 (Router/Gating Network)

```python
import torch
import torch.nn.functional as F

class MoERouter(torch.nn.Module):
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_weight = torch.nn.Parameter(torch.randn(d_model, num_experts) * 0.01)
    
    def forward(self, x):
        # x: [seq_len, d_model]
        # 计算每个 token 对每个 expert 的评分
        logits = x @ self.gate_weight  # [seq_len, num_experts]
        
        # Top-K 选择
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)  # 归一化权重
        
        # 辅助损失 (防止 expert 不平衡)
        aux_loss = self._load_balancing_loss(logits, topk_indices)
        
        return topk_weights, topk_indices, aux_loss
    
    def _load_balancing_loss(self, logits, topk_indices):
        """
        辅助损失: 鼓励所有 expert 被均匀使用
        不加入此损失 → Router 会偏向少数几个 expert (路由坍缩)
        """
        fracs = torch.zeros(self.num_experts, device=logits.device)
        for expert_idx in range(self.num_experts):
            fracs[expert_idx] = (topk_indices == expert_idx).sum().float()
        fracs = fracs / fracs.sum()
        # 鼓励均匀分布 (最小化与均匀分布的差距)
        return (fracs.pow(2).sum()) * self.num_experts
```

### Token 派发 (Dispatch & Combine)

```python
class MoELayer(torch.nn.Module):
    def __init__(self, d_model, num_experts, expert_hidden_dim, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = MoERouter(d_model, num_experts, top_k)
        # 每个 expert 就是一个标准 FFN
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_model, expert_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(expert_hidden_dim, d_model)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [seq_len, d_model]
        weights, indices, aux_loss = self.router(x)
        
        # 方法 1: 逐 expert 处理 (简单但慢)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]  # [seq_len]
            w = weights[..., k:k+1]       # [seq_len, 1]
            
            # 对每个 token, 用对应的 expert 处理并加权
            for i in range(x.size(0)):
                expert_output = self.experts[expert_idx[i]](x[i:i+1])
                if k == 0:
                    output[i] = w[i, 0] * expert_output
                else:
                    output[i] += w[i, 0] * expert_output
        
        return output, aux_loss
```

---

## 关键挑战

### 1. 路由坍缩 (Expert Collapse)

如果不用辅助损失，Router 会倾向于只依赖少数 1-2 个 expert。

**解决方案**:
- **Auxiliary Loss**: Mixtral 使用，强制均匀分配
- **Noise Gating**: 在 Router 输出上加随机扰动 (Switch Transformer)
- **Soft gating**: 不用 Top-K 硬选择，而用稀疏 Softmax (SoftMoE)

### 2. 通信瓶颈 (分布式训练)

MoE 的核心挑战不是计算，而是 **Expert 之间的 All-to-All 通信**。

```
GPU 0: tokens → 需要发往 GPU 1 的 Expert 3, GPU 2 的 Expert 5
GPU 1: tokens → 需要发往 GPU 0 的 Expert 1, GPU 3 的 Expert 7
GPU 2: tokens → 需要发往 GPU 0 的 Expert 2, GPU 3 的 Expert 6
GPU 3: tokens → 需要发往 GPU 1 的 Expert 4, GPU 2 的 Expert 8
                    ↓
          All-to-All Collectives
          (网络带宽是真正的瓶颈)
```

**DeepSeek-V3 的解决**: Multi-Token Prediction (MTP)，将专家分组，减少跨节点通信。

### 3. 训练不稳定

MoE 的 Router 梯度很难传播，因为 Top-K 选择是 discrete 操作。

**解决方案**:
- 直通过 (Straight-Through) 估计梯度
- Gumbel-Softmax 松弛
- 预训练一个 Dense 模型，然后 "Expert-split" 初始化

---

## MoE 的变体对比

| 模型 | 参数 | 激活参数 | Experts | Top-K | 路由策略 |
|------|------|---------|---------|-------|---------|
| Mixtral 8x7B | 46.7B | ~12.9B | 8 | 2 | Top-K + aux loss |
| Mixtral 8x22B | 141B | ~39B | 8 | 2 | Top-K + aux loss |
| DeepSeek-V3 | 671B | ~37B | 256 | ~6-8 | Aux-free + 分组路由 |
| Qwen1.5-MoE | 14B | ~2.7B | 4 | 2 | Top-K + aux loss |
| Switch Transformer | 1.6T | ~12B | 2048 | 1 | 仅选 1 个 |

### 为什么 DeepSeek-V3 可以用 671B 参数但只激活 37B?

**核心**: 专家数量多 (256) 且每个 Expert 较窄 (hidden_dim 较小)，通过分组路由 (Expert Parallelism) + Multi-Token Prediction 减少通信开销。256 个 Expert 分布在多个 GPU 上，每次前向传播只激活其中 6-8 个。

---

## 扩展阅读

- [DeepSeek-V2/V3 Architecture](https://arxiv.org/abs/2405.04434)
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [SoftMoE: Soft Mixture of Experts](https://arxiv.org/abs/2304.09891)

---

_上一个: [Day 1 - GRPO](01-grpo.md) | 下一个: [Day 3 - Speculative Decoding](03-speculative-decoding.md)_
