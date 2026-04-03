---
date: "2026-04-03"
difficulty: "高级"
category: "模型架构"
---

# 第 02 天：混合专家模型（MoE）—— 扩展模型规模而不扩展计算成本

> **观看动画演示**：![MoE 动画](../gifs/02-mixture-of-experts.gif)

---

## 一句话概要

混合专家模型（MoE）将每个前馈网络（FFN）层替换为一组 K 个独立的专家网络和一个可训练的路由器，路由器为每个 token 仅选择排名最高的 k 个专家，使得模型能够扩展到万亿级参数量，同时保持每个 token 的计算量恒定，且大致等同于稠密模型。

---

## 为什么需要 MoE？

### 计算扩展的瓶颈

传统的稠密 Transformer 扩展时，参数量和每个 token 的计算量同时增长：参数量翻倍意味着每个 token 每层的乘加运算也翻倍。一个 3000 亿参数的稠密模型每层每个 token 需要 3000 亿次浮点运算——这对训练和推理来说都过于昂贵。

### MoE 的核心思路

MoE 提出了一个问题：*我们能否拥有万亿级的总参数量，但仅为每个 token 激活其中一小部分？*

答案是可以的，其基本洞察是：并非每个 token 都需要所有神经元，并非每句话都需要所有概念，并非每个问题都需要所有技能。通过维护一组专业化的"专家"网络，并将每个 token 路由到最相关的少数专家，模型可以将海量知识存储在众多专家中，而计算每个 token 时只使用总参数的一小部分。

---

## 算法流程

```
==================================================================
              MoE 前向传播 —— 逐 Token
==================================================================

     ┌─────────────────────────────┐
     │     输入 Token h            │
     │  形状: (d_model,)           │
     └─────────────┬───────────────┘
                   │
                   │  h 送入路由器
                   ▼
     ┌─────────────────────────────────────────────┐
     │              路由器网络                      │
     │                                             │
     │  gate_logits = h · W_router                 │
     │  形状: (路由器维度 = num_experts)            │
     └─────────────┬───────────────────────────────┘
                   │
                   │  Top-K 选择 + 噪声 + 负载均衡
                   ▼
     ┌────────────────────────────────────────────────────────────┐
     │              Top-K 路由 + 负载均衡                          │
     │                                                            │
     │  P = softmax(gate_logits)          -- 路由概率            │
     │  Top-K 索引: {e_1, e_2, ..., e_k}                          │
     │  专家权重: w_1, w_2, ..., w_k                              │
     │                                                            │
     │  ┌──────────────────────────────┐                          │
     │  │  负载均衡辅助损失              │                          │
     │  │                              │                          │
     │  │  f_i = 路由到专家 i 的         │                          │
     │  │        token 占比             │                          │
     │  │  P_i = 专家 i 的平均           │                          │
     │  │        路由概率               │                          │
     │  │                              │                          │
     │  │  L_aux = α · N · Σ f_i·P_i   │                          │
     │  │                              │                          │
     │  └──────────────────────────────┘                          │
     └─────────────┬──────────────────────────────────────────────┘
                   │
                   │  分发 token 到选中的专家
                   ▼
     ┌──────────────────────────────────────────┐
     │          专家计算                         │
     │                                          │
     │  对每个选中的专家 e_j:                    │
     │    y_j = w_j · Expert_{e_j}(h)            │
     │                                          │
     │  注意: 每个专家是一个独立的 FFN            │
     │  具有隐藏维度 d_ff                        │
     └─────────────┬────────────────────────────┘
                   │
                   │  加权组合专家输出
                   ▼
     ┌──────────────────────────────────────────┐
     │          输出组合                         │
     │                                          │
     │  y = Σ_{j=1}^{k} w_j · Expert_{e_j}(h)    │
     │                                          │
     │  ★ 每个 token 仅激活 k 个专家              │
     │  ★ 总计算量 ≈ k/k_total of dense           │
     └─────────────┬────────────────────────────┘
                   ▼
              输出 y (d_model,)
```

---

## 数学公式

### 路由器与 Top-K 选择

对于每个输入 token h，路由器在所有 N 个专家上计算选择分数：

```
gate_logits = h · W_router                        -- 线性投影到 N 个分数
其中 W_router 的形状为 (d_model, N)

P = softmax(gate_logits / temperature)             -- 路由概率
P_i 对于 i 属于 {1, ..., N},  Σ P_i = 1

Top-K 选择:
  选择索引 T = top-k(P) = {e_1, e_2, ..., e_k}
  归一化选中的权重: w_j = P_{e_j} / Σ_{m=1}^{k} P_{e_m}
```

温度参数控制路由的"锐度"：较低的温度给出更确定的路由（一个专家占主导），较高的温度给出更柔和的路由（多个专家贡献更加均衡）。

### 专家输出计算

```
y = Σ_{j=1}^{k} w_j · Expert_{e_j}(h)

其中每个 Expert_i(h) = ReLU(h · W_gate_i + b_gate_i) · W_down_i

每个 token 的总前向浮点运算 ≈ k · (2 · d_model · d_ff)  -- 注意不是 N · d_ff
```

### 负载均衡辅助损失

如果没有显式的负载均衡损失，路由倾向于退化：少数"热门"专家接收大部分 token，而其他专家利用率不足（专家坍缩）：

```
f_i = (路由到专家 i 的 token 数) / (总 token 数)     -- 实际负载
P_i = (该 token 中专家 i 的平均路由概率)              -- 平均路由概率

L_aux = α · N · Σ_{i=1}^{N} f_i · P_i

其中 α 是权重系数（通常为 0.01）
```

当 f_i 和 P_i 都均匀（1/N）时，损失项 f_i · P_i 最小化。乘以 N 是为了归一化，使均匀路由下的期望损失等于 1。

### 容量因子与 Token 丢弃

在实践中，为专家分配固定的"容量"以支持批量 GPU 计算：

```
capacity = capacity_factor · (total_tokens / num_experts)

如果 专家 i 的 token 数 > capacity:
    -- 丢弃多余的 token（它们恒等通过）
    -- 或使用溢出缓冲区
```

1.25 的容量因子意味着每个专家可以处理超出平均预期负载 25% 的 token 数量。多余的 token 被丢弃或路由到回退机制。

---

## 稠密模型与 MoE 对比

| 维度 | 稠密 Transformer | MoE Transformer |
|---|---|---|
| 每层参数量 | d_model × d_ff | N × d_model × d_ff（N 个专家） |
| 每 token 激活参数量 | d_model × d_ff | k × d_model × d_ff（k 个专家） |
| 总参数量 | 与模型大小线性比例 | 与专家数量 N 成比例 |
| 每 token FLOPs | 固定，与所有参数成比例 | 总参数的 k/N |
| 显存占用 | 与激活参数成比例 | 与总参数成比例 |
| 训练稳定性 | 已被充分理解 | 对路由坍缩敏感 |
| 分布式通信 | 标准 all-reduce | 专家并行 + all-to-all |
| 实际案例 | LLaMA 3、Mistral 7B | Mixtral 8x7B、DeepSeek-V3 |
| 最佳权衡 | 简单性、部署便利性 | 以恒定计算容纳海量参数 |

---

## 专家坍缩与路由动态

### 什么是专家坍缩？

专家坍缩是指路由器退化为始终选择相同的 1 或 2 个专家的失效模式（"富者愈富"问题）：

```
初始状态（健康）:
  专家利用率: [12%, 11%, 13%, 12%, 11%, 12%, 13%, 14%]  ✓ 均匀

坍缩后状态（退化）:
  专家利用率:  [95%, 3%, 0%,  1%,  0%,  0%,  1%,  0%]    ✗ 坍缩
```

### 原因与预防

| 原因 | 机制 | 预防措施 |
|---|---|---|
| 正反馈循环 | 热门专家获得更多梯度更新，变得更受欢迎 | 辅助负载均衡损失 |
| 初始化不足 | 随机权重导致早期路由偏向少数专家 | 路由器初始化、温度预热 |
| 容量受限 | 热门专家丢弃 token，强化远离这些专家的路由 | 增大容量因子、随机路由 |
| 专家专业化 | 一个专家"抢走"所有简单样本，其他专家萎缩 | 噪声 Top-K 门控、随机 token 强制 |

### 噪声 Top-K 门控

在 Top-K 选择前添加噪声防止过早收敛：

```
noisy_logits = gate_logits + ε · randn_like(gate_logits)
P = softmax(noisy_logits / temperature)
```

这等价于 Gumbel-Softmax 松弛，在训练过程中维持路由决策中的探索性。

---

## Mixtral 8x7B 架构案例分析

```
Mixtral 8x7B:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  层数:             32 个 Transformer 块
  每层 MoE 专家数:   8（通过 top-2 路由全部可用）
  d_model:          4096
  每专家 d_ff:      14336
  每 token 激活参数:  约 130 亿（8 个专家中的 2 个）
  总参数:           约 467 亿
  每 token FLOPs:   约 120-130 亿（等同于稠密 7B 模型）

  关键细节: 使用门控 ReLU 专家和 top-2 路由
  并设 capacity_factor = 1.0（无容量限制）

  路由: 每个 token 恰好选择 2 个专家
  ─────────────────────────────────────────────────

Token h ──→ 路由器 (W: 4096→8) ──→ Top-2 概率
                │
                ├─→ 专家 3: FFN(h) × 0.52
                ├─→ 专家 7: FFN(h) × 0.48
                │
                └─→ y = 0.52·E3(h) + 0.48·E7(h)
```

---

## Python 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Expert(nn.Module):
    """
    单个专家网络 —— 一个标准的前馈网络模块。

    在实践中，专家通常是 SwiGLU 或门控 ReLU 的 FFN，
    与稠密 Transformer 中使用的相同。
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用专家 FFN 门控和降维投影。

        参数:
            x: 形状 (batch * seq_len, d_model) 的输入 token

        返回:
            output: 形状 (batch * seq_len, d_model) 的输出
        """
        return self.w_down(F.relu(self.w_gate(x)))


class MoELayer(nn.Module):
    """
    混合专家层，支持 Top-K 路由、负载均衡和辅助损失。

    这实现了 Switch Transformer / Mixtral 风格
    MoE 层的简化版本。
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
        初始化 MoE 层。

        参数:
            d_model: 模型维度
            d_ff: 每专家的前馈隐藏维度
            num_experts: 专家网络总数
            top_k: 每个 token 选择的专家数量
            capacity_factor: 专家容量倍增系数（1.0 = 精确平均）
            aux_loss_weight: 负载均衡辅助损失权重 (alpha)
            noise_std: 路由噪声标准差（用于训练）
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.noise_std = noise_std

        # 路由器：简单的线性投影到 num_experts 个分数
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # 创建独立的专家
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        MoE 层的前向传播。

        参数:
            x: 形状 (batch, seq_len, d_model) 的输入 token

        返回:
            output: 形状 (batch, seq_len, d_model) 的组合专家输出
            aux_loss: 标量辅助损失
        """
        batch_size, seq_len, d_model = x.shape

        # 展平以便统一处理所有 token
        flat_x = x.reshape(-1, d_model)  # (batch * seq_len, d_model)

        # --- 计算路由分数 ---
        gate_logits = self.router(flat_x)  # (BT, num_experts)

        # 训练期间添加噪声用于探索
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # 路由概率
        routing_weights = F.softmax(gate_logits, dim=-1)  # (BT, num_experts)

        # --- 负载均衡辅助损失 ---
        aux_loss = self.compute_auxiliary_loss(routing_weights)

        # --- Top-K 选择 ---
        # 获取 Top-K 专家索引及其路由权重
        top_k_weights, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )  # (BT, top_k)

        # 归一化选中的权重使它们和为 1
        top_k_weights = top_k_weights / top_k_weights.sum(
            dim=-1, keepdim=True
        )

        # --- 分发和计算 ---
        # 初始化输出张量
        output = torch.zeros_like(flat_x)  # (BT, d_model)

        # 分别处理每个专家
        # 在生产环境中，这使用 scatter/gather 以提高效率
        for expert_idx in range(self.num_experts):
            # 找出哪些 token 选择了此专家
            selected_mask = (top_k_indices == expert_idx)  # (BT, top_k)
            selected_positions = selected_mask.any(dim=-1)  # (BT,)

            if not selected_positions.any():
                continue

            # 收集此专家的 token
            expert_input = flat_x[selected_positions]  # (n_tokens_i, d_model)

            # 同时收集这些 token 的权重
            # 我们需要对多个 top-k 位置的权重求和
            expert_weights = (top_k_weights * selected_mask.float()).sum(
                dim=-1
            )  # (n_tokens_i,)

            # 计算专家输出
            expert_output = self.experts[expert_idx](expert_input)

            # 按路由概率加权并散射回原位
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            output[selected_positions] += expert_output

        # 重塑回原始维度
        output = output.reshape(batch_size, seq_len, d_model)

        return output, aux_loss

    def compute_auxiliary_loss(
        self, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负载均衡辅助损失。

        这鼓励 token 在专家之间均匀分布，
        以防止专家坍缩。

        参数:
            routing_weights: 形状 (num_tokens, num_experts) 路由概率

        返回:
            aux_loss: 标量辅助损失值
        """
        num_tokens = routing_weights.size(0)
        N = self.num_experts

        # f_i: 路由到专家 i 的实际 token 占比（基于 Top-K）
        top_k_probs, top_k_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        expert_counts = torch.zeros(N, device=routing_weights.device)
        for i in range(N):
            expert_counts[i] = (top_k_indices == i).float().sum()
        f = expert_counts / (num_tokens * self.top_k)

        # P_i: 专家 i 的平均路由概率
        P = routing_weights.mean(dim=0)  # (num_experts,)

        # 损失: N * sum(f_i * P_i)
        aux_loss = self.aux_loss_weight * N * torch.sum(f * P)

        return aux_loss


class MoETransformerBlock(nn.Module):
    """
    带有 MoE FFN 层的简化 Transformer 块。

    将注意力机制与基于 MoE 的 FFN 结合，
    形成完整的 Transformer 层。
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
            noise_std=0.1,  # 训练时的小噪声用于探索
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        带有残差连接和层归一化的前向传播。

        返回:
            output: 形状 (batch, seq_len, d_model)
            aux_loss: 标量 MoE 辅助损失
        """
        # Pre-LN 残差注意力
        attn_in = self.norm1(x)
        attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)
        x = x + attn_out

        # Pre-LN 残差 MoE
        moe_in = self.norm2(x)
        moe_out, aux_loss = self.moe(moe_in)
        x = x + moe_out

        return x, aux_loss


# ------------------------------------------------------------------
# 示例用法 —— 训练一个小型 MoE 模型
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # 小型模拟配置
    batch_size = 4
    seq_len = 32
    d_model = 256
    n_heads = 4
    num_experts = 8
    d_ff = 512
    top_k = 2

    # 创建一个模拟的 MoE Transformer 块
    block = MoETransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        num_experts=num_experts,
        d_ff=d_ff,
        top_k=top_k,
        capacity_factor=1.25,
        aux_loss_weight=0.01,
    )

    print(f"MoE 配置:")
    print(f"  专家数量: {num_experts}")
    print(f"  Top-K 路由: {top_k}")
    print(f"  每专家参数: {sum(p.numel() for p in block.moe.experts[0].parameters())/1e6:.2f}M")
    print(f"  MoE 总参数: {sum(p.numel() for p in block.moe.parameters())/1e6:.2f}M")
    print(f"  每 token 激活参数: {sum(p.numel() for p in block.moe.experts[0].parameters())/1e6 * top_k:.2f}M")
    print()

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"输入形状: {x.shape}")

    # 前向传播
    block.train()
    output, aux_loss = block(x)

    print(f"输出形状: {output.shape}")
    print(f"辅助损失（负载均衡）: {aux_loss.item():.6f}")
    print()

    # 检查是否可以反向传播（包括辅助损失）
    total_loss = output.sum() + aux_loss
    total_loss.backward()
    print("反向传播成功 —— MoE 层完全可微。")

    # 演示负载均衡: 检查 token 分布
    # 运行多个批次查看路由是否均匀
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
    print(f"专家利用率（100 个批次）: {pct}")
    print(f"标准差: {expert_counts.std().item():.4f} "
          f"（越低 = 平衡越好）")
```

---

## 深入探究

### 1. 为什么路由不会总是坍缩？

理论上的风险是：优化器发现一个"懒"解：将所有内容路由到一个易于训练的专家并停止学习。在实践中，辅助损失通过直接惩罚非均匀分布来防止这一点。但仅有辅助损失是不够的——关键洞察在于：不同的专家在训练过程中对不同类型的 token 自然形成专业化分工，因为它们接收到不同的梯度信号：

```
Token 类型 → 不同专家的专业化：
  数学 token   → 专家 2, 专家 5（学习算术模式）
  代码 token   → 专家 1, 专家 3（学习语法结构）
  散文 token   → 专家 0, 专家 4, 专家 7（学习语言模式）
  混合 token   → 专家 6（通用 / 回退）
```

这种专业化自然涌现，因为路由器学会了将 token 表示与专家的专业化相匹配，而每个专家的权重是由它所接收的特定 token 子集塑造的。

### 2. 专家并行：大规模分布式训练

当 N 个专家无法放入单个 GPU 时，MoE 引入了专门的分布式训练模式：

```
专家并行 (EP):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─GPU 0───┐  ┌─GPU 1───┐  ┌─GPU 2───┐  ┌─GPU 3───┐
│专家 0    │  │专家 2    │  │专家 4    │  │专家 6    │
│专家 1    │  │专家 3    │  │专家 5    │  │专家 7    │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └──── all-to-all 通信 ─────────────────┘
     （token 被路由到正确的 GPU 进行专家处理）

  All-to-all 成本随专家数量呈 O(N) 增长。
  通信开销是 MoE 训练的主要瓶颈。
```

DeepSeek-V3 进一步通过 "DualPipe" 优化，实现了计算与通信的重叠，以及"高速路由"，允许 token 在不需要任何专家时完全跳过 MoE 层。

### 3. MoE 容量与 Token 丢弃

在 Switch Transformers 中，对每个专家强制实施严格的容量限制以支持静态批次计算。溢出专家容量的 token 要么：

- **被丢弃**：它们不变地通过 MoE 层，并且不贡献辅助损失。这很简单但浪费了这些 token 中的信息。
- **溢出缓冲**：多余的 token 被排队并在下一个机会中处理。

容量因子是一个关键超参数。设置太低（1.0）有过多的 token 丢弃风险，降低模型质量。设置太高（1.5+）浪费 GPU 显存和计算，违背 MoE 的效率目的。

### 4. 稠密-稀疏模型家族：相同架构，不同训练策略

MoE 的一个关键优势：你可以通过设置 top_k = num_experts（所有专家激活）来训练相同架构的稠密版本。这允许在训练预算之间无缝转换：

```
训练策略:
  充裕预算:  top_k = N（稠密）→ 最高质量，最高计算
  中等:      top_k = 2 到 N/4 → 良好质量，适度计算
  紧张:      top_k = 1 → 稀疏激活，最低计算

相同的模型权重，相同的专家专业化，
仅在推理时使用不同的路由稀疏度。
```

### 5. DeepSeek-V3 的进阶 MoE 特性

DeepSeek-V3（及 V2）引入了多项 MoE 创新：

- **共享专家**：除了 top-k 路由的专家外，还有少量"共享专家"对所有 token 进行激活。它们在所有路由决策中共享相同的权重，并捕获所有 token 都受益的通用模式。

- **多 token 预测**：DeepSeek-V3 的 MoE 层还学习并行预测多个未来 token，实际上在专家已经激活的情况下免费获得了额外预测的计算力。

- **细粒度专家**：DeepSeek-V3 不使用少数大型专家，而是使用许多小型专家（256 个专家），实现更精细的专业化和更好的路由决策。

---

## 延伸阅读

- **Switch Transformers**（Fedus 等，2021）：https://arxiv.org/abs/2101.03961
- **Mixtral of Experts**（Jiang 等，2024）：https://arxiv.org/abs/2401.04088
- **DeepSeek-V3 Technical Report**：https://arxiv.org/abs/2412.19437
- **GShard**（Lepikhin 等，2021）：https://arxiv.org/abs/2006.16668
- **Sparse Mixture of Experts**（综述）：https://arxiv.org/abs/2209.00085

---

_上一篇：[第 01 天 —— GRPO](01-grpo.md)  |  下一篇：[第 03 天 —— 投机解码](03-speculative-decoding.md)_