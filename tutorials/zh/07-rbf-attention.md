# 第 07 天：超越点积——RBF 注意力机制

> **日期**：2026-04-03 | **难度**：高级 | **类别**：Transformer 架构创新

> **架构示意图**：![RBF 注意力架构](../diagrams/07-rbf-attention.png)

---

## 一句话概要

RBF（径向基函数）注意力将 Transformer 中标准的点积 `Q·K^T` 替换为基于距离的核函数 `exp(-||Q - K||^2 / (2σ^2))`。这从根本上改变了 token 之间相似度的度量方式：**从角度对齐变为空间距离**。

---

## 为什么需要替代点积？

### 点积注意力的局限性

标准 Transformer 注意力的计算公式为：

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

核心操作 `Q·K^T` 是一个**内积**（点积）：

- 当 Q 和 K 方向一致时（夹角小），内积较大
- 当 Q 和 K 正交或反向时，内积较小

内积关注的是向量方向的相似度，与向量长度关系较小。

**问题所在**：点积相似度存在两个已知的局限：

1. **范数敏感性**：范数较大的 query 会主导 softmax 输出，与语义无关
2. **线性决策边界**：`Q·K = 0` 定义了一个超平面，无法捕获复杂的非线性关系

### RBF 注意力的解决方案

```
RBF_Attention(Q, K, V) = softmax( -||Q - K||^2 / (2σ^2) ) · V
```

其中 `||Q - K||^2` 是 Q 与每个 K 之间的**欧几里得距离平方**：

```
||Q - K||^2 = Σ (Q_i - K_i)^2
             = ||Q||^2 - 2(Q·K) + ||K||^2
```

当 Q 和 K 距离较近时，分数较大（接近 1）。

当 Q 和 K 距离较远时，分数呈指数级衰减。

---

## 数学对比

### 点积对比 RBF

```
点积注意力:                     RBF 注意力:
  sim(q, k) = q·k                sim(q, k) = exp(-||q - k||^2 / (2σ^2))

特性:                            特性:
  - 对 q 和 k 呈线性              - 非线性（指数）
  - 无界范围 (-∞, +∞)            - 有界范围 (0, 1)
  - 方向相似度                    - 空间邻近度
  - 对 ||q||·||k|| 敏感          - 对 ||q - k|| 敏感
```

### 带宽参数 σ 的作用

参数 σ 控制"注意力半径"的大小：

```
σ 较大 → exp(-小距离 / 大σ^2) ≈ 1：
  所有 token 看起来都相似 → 注意力趋于均匀（类似平均池化）

σ 较小 → exp(-距离 / 小σ^2) 急剧衰减：
  只有非常接近的 token 才相似 → 注意力高度稀疏

σ ≈ hidden_dim / √2（甜蜜点）：
  在聚焦与分散之间取得平衡
```

### 数学关系：RBF 可以表达点积

```
exp(-||q - k||^2 / (2σ^2))
= exp(-||q||^2/(2σ^2)) · exp(-||k||^2/(2σ^2)) · exp(q·k/σ^2)

如果预先归一化使 ||q||^2 = ||k||^2 = 常数：
= C · exp(q·k/σ^2)

当 σ 较小时：exp(q·k/σ^2) ≈ 1 + q·k/σ^2 ≈ 点积注意力！
```

RBF 注意力是点积注意力的**严格推广**。在点积有效时，它能近似点积的行为；但它还能捕获点积无法表达的非线性关系。

---

## 代码实现

```python
import torch
import torch.nn.functional as F
import math

class RBFAttention(torch.nn.Module):
    """
    径向基函数注意力
    将 Q·K^T 替换为 exp(-||Q - K||^2 / (2σ^2))
    """
    def __init__(self, d_model, n_heads, sigma=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        # 每个头可学习的带宽参数 σ
        if sigma is not None:
            self.sigma = torch.nn.Parameter(torch.ones(n_heads) * sigma)
        else:
            # 默认初始化：σ = sqrt(head_dim)
            self.sigma = torch.nn.Parameter(torch.ones(n_heads) * math.sqrt(self.head_dim))

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 重排张量维度：[batch, heads, seq, dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 计算平方欧几里得距离
        # ||q - k||^2 = ||q||^2 - 2*q·k + ||k||^2
        q_sq = (Q ** 2).sum(dim=-1, keepdim=True)   # [B, H, S, 1]
        k_sq = (K ** 2).sum(dim=-1, keepdim=True).transpose(-1, -2)  # [B, H, 1, S]
        dot = Q @ K.transpose(-1, -2)                # [B, H, S, S]

        dist_sq = q_sq - 2 * dot + k_sq              # [B, H, S, S]

        # RBF 核函数：exp(-||q-k||^2 / (2*σ^2))
        # 每个头独立 σ，通过广播机制匹配维度
        sigma = self.sigma.view(1, self.n_heads, 1, 1)
        scores = -dist_sq / (2 * sigma ** 2)         # [B, H, S, S]

        # 注意力掩码（例如因果掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = (attn_weights @ V).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(output), attn_weights

class HybridAttention(torch.nn.Module):
    """
    混合注意力：结合点积和 RBF 注意力

    score = α * dot_product + (1 - α) * rbf

    混合系数 α 每个头独立学习，让模型自动选择最优混合比例。
    """
    def __init__(self, d_model, n_heads, sigma=None):
        super().__init__()
        self.rbf_attn = RBFAttention(d_model, n_heads, sigma)
        self.alpha = torch.nn.Parameter(torch.ones(n_heads) * 0.5)

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 点积分数（标准注意力）
        dot_scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        # RBF 分数（复用距离计算逻辑）
        q_sq = (Q ** 2).sum(-1, keepdim=True)
        k_sq = (K ** 2).sum(-1, keepdim=True).transpose(-1, -2)
        rbf_scores = -((q_sq - 2 * torch.matmul(Q, K.transpose(-1, -2)) + k_sq) /
                        (2 * self.rbf_attn.sigma.view(1, self.n_heads, 1, 1) ** 2))

        # 混合策略：通过 sigmoid 确保 α 在 0 到 1 之间
        alpha = torch.sigmoid(self.alpha).view(1, self.n_heads, 1, 1)
        scores = alpha * dot_scores + (1 - alpha) * rbf_scores

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = (attn @ V).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(output)
```

---

## 深入分析

### RBF 注意力在哪些场景下更有效？

| 场景               | 点积   | RBF    | 原因说明                                                         |
|--------------------|:------:|:------:|-----------------------------------------------------------------|
| 语义文本相似度       | ★★★★   | ★★★★★  | RBF 更好地捕获语义空间中的"邻近性"                               |
| 长距离依赖           | ★★★★   | ★★     | 点积的无界范围有助于连接远距离 token                              |
| 数值/表格数据        | ★★     | ★★★★★  | 欧几里得距离是结构化数据的自然度量                               |
| 代码补全             | ★★★    | ★★★★   | 代码同时包含方向性（意图）和邻近性（变量）信号                    |
| 多模态对齐           | ★★★    | ★★★★★  | 不同模态在距离比角度更具信息量的空间中                           |

### σ 的学习动态

```
训练过程中的 σ 变化规律：
  第 1-10 轮：   σ 较大 → 注意力接近均匀 → 模型稳定训练
  第 10-50 轮：  σ 缩小 → 注意力变得更具选择性 → 专业分工开始
  第 50 轮以后： 不同的头收敛到不同的 σ 值
                （一些头像点积注意力，另一些高度局部化）
```

这与多头标准注意力的行为类似：某些头学习局部关注，另一些学习全局关注。但 RBF 通过 σ 使这种分工变得显式可解释，而非隐藏在 Q 和 K 的投影权重中。

### 潜在问题

1. **计算开销略大**：计算 `||Q-K||^2` 需要 `q_sq + k_sq - 2*dot`，比单纯的点积多了一步计算（不过与 softmax 相比可以忽略不计）。

2. **σ 初始化敏感**：如果初始化时 σ 过小，注意力退化为 one-hot 分布；如果过大，则退化为均匀平均。

3. **与已有权重的兼容性**：无法直接将预训练的 Transformer 转换为 RBF 注意力——Q 和 K 投影需要重新微调。

---

## 扩展阅读

- [RBF Attention for Improved Similarity in Transformers](https://www.reddit.com/r/MachineLearning/comments/1s9cdq0/) —— 原始 Reddit 讨论
- [Gaussian Attention: A RBF Approach to Self-Attention](https://arxiv.org/) —— 相关工作
- [Performers: Rethinking Attention with Positive Orthogonal Random Features](https://arxiv.org/abs/2009.14794) —— 替代注意力核函数
- [cosformer: Rethinking Softmax in Attention](https://arxiv.org/abs/2202.08791) —— 非点积注意力核函数

---

_上一篇：[Day 06 - 量化](06-quantization.md)  |  下一篇：[Day 08 - 记忆与 KV Cache 优化](08-memory-kv-cache.md)_
