---
date: "2026-04-03"
difficulty: "高级"
category: "推理加速"
---

# 第 03 天：投机解码（Speculative Decoding）—— 无损推理加速

> **观看动画演示**：![投机解码动画](../gifs/03-speculative-decoding.gif)

---

## 一句话概要

投机解码使用小型草稿模型并行预测 K 个候选 token，然后利用大型目标模型在一次前向传播中验证所有 K 个 token，以 min(1, p/q) 的概率接受每个 token——在数学上保证与目标模型输出分布完全相同的前提下，实现 2 至 4 倍的推理加速。

---

## 为什么需要投机解码？

### 自回归生成的瓶颈

标准的自回归生成每次前向传播仅输出一个 token，无论该 token 是简单可预测还是困难：

```
标准解码：
  "猫" → [前向传播：100 亿 FLOPs] → "坐"
  "坐"  → [前向传播：100 亿 FLOPs] → "在"
  "在"  → [前向传播：100 亿 FLOPs] → "垫"
  "垫"  → [前向传播：100 亿 FLOPs] → "子"

  4 个 token = 4 次前向传播 = 4 × 100 亿 FLOPs = 总计 400 亿 FLOPs
```

这本质上是内存带宽受限，而非计算受限。每生成一个 token，模型权重都必须从 GPU 显存中流式传输，但对单个 token 表示进行实际矩阵乘法的计算量非常小。GPU 大部分时间处于空闲状态，等待权重数据到达。

### 投机解码的核心思路

投机解码提出了一个问题：*能否让一个廉价的草稿模型猜测目标模型将产生什么，然后仅对目标模型进行一次计算来同时检查所有猜测？*

答案是可以的，其关键在于一个验证算法，保证输出分布在数学上等同于单独运行目标模型所生成的分布：

```
投机解码（草稿 K=3，目标模型验证）：
  草稿模型（1.3亿参数）："猫" → "坐" → "在" → "垫"   [小模型 3 次前向传播]
  目标模型（100亿参数）：   "猫" + 3 个 token → 一次传播验证全部
  结果：3 个 token 仅用 1 次大模型前向传播代替 3 次
  加速比：约 2-3 倍，零质量损失

  每次目标前向传播产生的 token 数量：
    标准方式：  始终 1 个
    投机解码：  0.5 到 K+1 个（平均约 2-4 个，取决于草稿质量）
```

---

## 算法流程

```
==================================================================
          投机解码：草稿 + 验证 循环
==================================================================

步骤 1：自回归生成 K 个草稿 token
────────────────────────────────────────────────

  ┌──────────────────┐
  │  当前上下文       │──► 草稿模型生成 x_1, x_2, ..., x_K
  │  x_1, ..., x_t   │    （小模型 K 次前向传播）
  └──────────────────┘


步骤 2：在一次目标前向传播中验证所有 K+1 个候选
─────────────────────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────┐
  │  目标模型输入：x_1, ..., x_t, x_1, ..., x_K            │
  │                                                      │
  │  目标模型计算：q(x_{t+1}), q(x_{t+2}), ...            │
  │                   q(x_{t+K}), q(x_{t+K+1})           │
  │  （大模型 1 次前向传播，输出 K+1 个概率分布）        │
  └─────────────┬────────────────────────────────────────┘
                │
                ▼


步骤 3：并行验证，接受概率判定
─────────────────────────────────────────────────────────

  ┌─────────────────────────────────────────────────────────────┐
  │  对每个位置 i = 1 到 K:                                     │
  │                                                             │
  │  草稿概率:  r_i = P_draft(x_i | x_1, ..., x_{i-1})         │
  │  目标概率: q_i = P_target(x_i | x_1, ..., x_{i-1})         │
  │                                                             │
  │  接受率: α_i = min(1, q_i / r_i)                            │
  │                                                             │
  │  抽样 u ~ 均匀分布(0, 1)                                   │
  │  若 u < α_i:                                               │
  │    接受 x_i ✓                                              │
  │    继续验证下一位置                                        │
  │  否则:                                                      │
  │    拒绝 x_i ✗                                              │
  │    从修正分布中抽样替代 token                               │
  │    中断验证循环                                            │
  │                                                             │
  │  若全部 K 个都接受：从 q(x_{t+K+1}) 中额外抽样 1 个         │
  └─────────────┬───────────────────────────────────────────────┘
                │
                ▼
     将接受的 token 追加到输出序列末尾
     使用更新后的上下文从步骤 1 开始重复


步骤 4：拒绝采样（修正分布）
────────────────────────────────────────────────────

  若位置 j 被拒绝：
    计算修正分布：
      p_adjusted(x) = max(0, q(x) - r(x)) / (1 - Σ_{已接受} 接受概率)

    抽样 x_{t+j} ~ p_adjusted 并追加

  此修正确保最终的分布**精确等于** q，而非近似。
```

---

## 数学公式

### 接受概率

对于位置 i 的草稿 token x_i，草稿模型对其分配概率 r_i，目标模型分配概率 q_i。该 token x_i 被接受的概率为：

```
α_i = min(1, q_i / r_i)
```

此公式包含三种情况：

```
情况 1: q_i > r_i（目标模型比草稿模型更有信心）
  α_i = 1 → 始终接受 ✓
  草稿模型对该优秀 token 实际上过于保守了。

情况 2: q_i < r_i 但 q_i ≈ r_i（目标模型大致同意草稿）
  α_i ≈ 1 → 几乎总是接受 ✓
  偏差很小，被拒绝的风险很小。

情况 3: q_i 明显小于 r_i（目标模型不同意草稿）
  α_i < 1 → 可能被拒绝 ✗
  草稿模型对一个错误 token 过于自信了。
```

### 为什么 min(1, q/r) 能精确保留分布

核心定理：投机解码的输出在分布上与单独从目标模型进行贪婪/祖先采样**完全相同**。

```
证明概要：

P(接受 x_i) = α_i = q_i / r_i   （当 q_i ≤ r_i 时）

接受序列 x_1, ..., x_k 然后接受 x_{k+1}（通过最终抽样）的总概率为：

  ∏_{i=1}^{k} [r_i · (q_i/r_i)] · q_{k+1}  =  ∏_{i=1}^{k} q_i · q_{k+1}  =  ∏_{i=1}^{k+1} q_i

这等于目标模型的概率。∎

当一个 token 被拒绝时，修正分布 p_adjusted 为：

  p_adjusted(x) = max(0, q(x) - r(x)) / Z

其中 Z = 1 - Σ_{x: q(x)≥r(x)} r(x) · α(x) = Σ_x max(0, q(x) - r(x))

这就是草稿未能捕获的"剩余概率质量"。从 p_adjusted 中抽样
恰好补足了与 q 精确匹配所需的剩余概率质量。∎
```

### 期望接受率与加速比

每轮接受 token 的期望数量为：

```
E[接受数] = Σ_{i=1}^{K} α_i · Π_{j=1}^{i-1} α_j

如果近似假设所有 α_i ≈ α（常数接受率）：
  E[接受数] ≈ (1 - α^K) / (1 - α)  （α < 1 时）
  E[接受数] ≈ K  （α ≈ 1 时）

加速比 ≈ E[接受数] + 1  （+1 是草稿模型成本的分数部分）

计算示例：
  α = 0.90,  K=4:  E[接受数] ≈ 4.0 → 加速比 ≈ 4.1x
  α = 0.80,  K=4:  E[接受数] ≈ 2.9 → 加速比 ≈ 3.1x
  α = 0.70,  K=4:  E[接受数] ≈ 2.2 → 加速比 ≈ 2.5x
  α = 0.50,  K=4:  E[接受数] ≈ 1.0 → 加速比 ≈ 1.4x
  α = 0.95,  K=8:  E[接受数] ≈ 6.7 → 加速比 ≈ 6.9x
```

### 最优 K 值选择

```
最优 K 取决于接受率 α：

  低质量草稿（α ≈ 0.5）： K = 1 到 3  （更多 token = 更多拒绝）
  中等质量（α ≈ 0.7）：     K = 3 到 5  （平衡点）
  高质量草稿（α ≈ 0.9）：  K = 5 到 10（更长的投机才有意义）

经验法则：K ≈ -log(0.1) / -log(α)
  （使得至少 1 次接受概率达到 90% 的 K 值）
  α = 0.7 → K ≈ 6.5
  α = 0.8 → K ≈ 10.3
  α = 0.9 → K ≈ 22
  α = 0.95 → K ≈ 45

实践中，K = 3 到 8 是一个合理范围，因为：
  - 超过 K=8 后收益递减（拒绝概率复合增加）
  - 更大的 K 需要更多的目标模型 KV 缓存容量
  - 通信开销随 K 增大
```

---

## 方法对比

| 维度 | 标准自回归 | 投机解码 | Medusa | 前瞻解码 | EAGLE |
|---|---|---|---|---|---|
| 加速比 | 1x（基准） | 2-4x | 2-3x | 2-4x | 2-5x |
| 质量损失 | 无 | 无（精确） | 小（近似） | 无（精确） | 无（精确） |
| 是否需要草稿模型 | 不适用 | 是（独立小模型） | 否（基于多头） | 否（n-gram 缓存） | 否（基于特征） |
| 每步最大接受数 | 1 | K | K 头 × 深度 | N-gram 长度 | K（特征草稿） |
| 显存开销 | 仅基础模型 | 基础模型 + 草稿模型 | K × 词表大小 的头部 | N-gram 查找表 | 特征层 |
| 是否需要训练 | 不适用 | 需微调草稿模型 | 在目标模型上训练 K 个头 | 无需训练 | 训练特征层 |
| 最佳场景 | 通用，无需额外设置 | 有草稿模型可用 | 单模型部署 | 重复性/模板化文本 | 高精度推理 |
| 理论保证 | 精确分布 | 精确分布 | 近似 | 精确分布 | 精确分布 |

---

## 投机解码的变体

### 1. 经典投机解码（Chen 等）

使用独立小型草稿模型的原始方法：

```
架构：
  草稿模型：1.25 亿参数（例如从目标模型蒸馏而来）
  目标模型：70 亿+ 参数（原始模型）

  草稿自回归生成 K 个 token → 目标模型验证

  优点：简单、精确、已有充分研究
  缺点：需要训练/获取一个草稿模型
```

### 2. Medusa：多头投机解码

不采用独立的草稿模型，Medusa 在目标模型上额外添加 K 个"头部"：

```
        ┌─────────────┐
        │ Transformer  │
        │    层        │
        └──────┬───────┘
               │
          ┌────┼────┬────┬────────┐
          ▼    ▼    ▼    ▼        ▼
      Head_0 Head_1 Head_2 Head_3 Head_4
      （原始） （t+1） （t+2） （t+3） （t+4）

  每个头独立预测一个未来 token，
  绕过自回归生成。

  关键权衡：
    - 不需要独立模型（部署更简单）
    - 每个头部比完整的自回归模型弱
    - 接受率低于专用草稿模型
    - 质量是近似的，非精确保证
```

### 3. 前瞻解码（Lookahead Decoding）

使用 n-gram 缓存来跳过曾经出现过的 token：

```
  生成过程中维护一个字典：
    前缀 n-gram → 可能的续写

  当当前上下文与缓存中的前缀匹配时：
    直接"前瞻"到缓存中的续写
    由目标模型验证

  最佳用于：重复性文本、模板、代码模式
  最不适用于：全新的、不可预测的文本
```

### 4. EAGLE：基于特征的草稿

EAGLE 不训练单独的小模型，而是训练从中间层表示预测未来 token 的草稿头部：

```
  EAGLE 的洞察：中间层包含着对未来 token 有用的信息。
  在中间特征之上构建一个小网络即可精确起草，
  无需完整的独立模型。

  架构：
    目标模型前向传播 → 提取中间层特征
    草稿网络（MLP + 注意力）基于特征 → 预测后续 token
    目标模型验证草稿

  关键优势：
    - 接受率高于 Medusa（利用丰富的中间特征）
    - 显存低于独立草稿模型
    - 精确分布保证
```

---

## Python 代码实现

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
    根据目标模型的分布验证草稿 token。

    这实现了投机解码的核心验证算法，
    保证输出分布在数学上与单独从目标模型抽样完全相同。

    参数:
        draft_tokens: 形状 (K,) 草稿模型生成的 token
        draft_probs:  形状 (K,) 草稿模型对自身预测的分配概率
        target_logits: 形状 (K+1, 词表大小) 目标模型的 logits，
                       对应位置 t+1 到 t+K+1
        temperature:   采样温度（默认 1.0）

    返回:
        accepted_tokens: 本轮接受的 token
        n_accepted: 接受的草稿 token 数量（0 到 K）
        replacement_token: 若发生拒绝，从修正分布中抽样的替换 token
                          （若全部接受则为 None）
    """
    K = draft_tokens.size(0)

    # 获取草稿 token 位置的目标概率
    # 注意：target_logits[0] 对应位置 t+1，以此类推
    target_probs = F.softmax(target_logits / temperature, dim=-1)  # (K+1, 词表大小)

    accepted_tokens = []
    n_accepted = 0
    replacement_token = None

    # --- 依次验证每个草稿 token ---
    for i in range(K):
        r_i = draft_probs[i]           # 草稿模型对其预测的概率
        q_i = target_probs[i, draft_tokens[i]]  # 目标模型对草稿 token 的概率

        # 接受概率
        alpha_i = min(1.0, q_i / r_i)

        if torch.rand(1).item() < alpha_i:
            # 接受该 token
            accepted_tokens.append(draft_tokens[i].item())
            n_accepted += 1
        else:
            # 拒绝该 token
            # 从修正（残差）分布中抽样

            # 计算：max(0, q(x) - r(x))，对词表中所有 x
            # 其中 r(x) = 如果 x 等于草稿 token 则取草稿概率，否则为 0
            r_expanded = torch.zeros_like(target_probs[i])
            r_expanded[draft_tokens[i]] = r_i

            adjusted = torch.clamp(target_probs[i] - r_expanded, min=0.0)
            adjusted_sum = adjusted.sum()

            if adjusted_sum > 0:
                adjusted = adjusted / adjusted_sum
                replacement = torch.multinomial(adjusted, 1)
                replacement_token = replacement.item()
            else:
                # 边界情况：直接从目标模型抽样
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
    执行一步投机解码。

    草稿模型自回归生成 max_draft_length 个候选 token，
    然后目标模型在一次前向传播中验证它们。

    参数:
        draft_model: 较小的草稿模型（须支持前向传播并返回 logits）
        target_model: 大型目标模型
        context: 形状 (1, seq_len) 当前 token 序列
        max_draft_length: 最大草稿 token 数量（K）
        temperature: 采样温度

    返回:
        new_tokens: 本轮追加的所有 token（形状可变：0 到 K+1）
        n_accepted: 接受的草稿 token 数量
    """
    draft_tokens_list = []
    draft_probs_list = []

    current = context.clone()

    # --- 阶段 1：草稿生成（小模型自回归）---
    with torch.no_grad():
        for _ in range(max_draft_length):
            # 草稿模型前向传播（小模型，快速）
            draft_logits = draft_model(current)  # (1, seq, 词表大小)
            next_token_logits = draft_logits[0, -1, :]
            next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)

            # 从草稿分布中抽样一个 token
            draft_token = torch.multinomial(next_token_probs, 1)
            draft_prob = next_token_probs[draft_token].item()

            draft_tokens_list.append(draft_token.item())
            draft_probs_list.append(draft_prob)

            # 追加到草稿输入以便下一步生成
            current = torch.cat([current, draft_token.unsqueeze(0)], dim=-1)

    draft_tokens = torch.tensor(draft_tokens_list, device=context.device)
    draft_probs = torch.tensor(draft_probs_list, device=context.device)

    # --- 阶段 2：目标模型验证（一次大模型前向传播）---
    with torch.no_grad():
        # 目标模型一次传播验证所有位置
        # 输入：上下文 + 所有 K 个草稿 token
        target_input = torch.cat([context, draft_tokens.unsqueeze(0)], dim=-1)
        target_logits = target_model(target_input)  # (1, seq+K, 词表大小)

        # 提取草稿位置和额外一个位置的 logits
        # 我们需要位置 len(context), len(context)+1, ..., len(context)+K 的 logits
        verify_logits = target_logits[0, -max_draft_length - 1:, :]

    # --- 阶段 3：接受/拒绝 ---
    accepted_tokens, n_accepted, replacement = speculative_verify(
        draft_tokens, draft_probs, verify_logits, temperature
    )

    # 构建输出 token
    if replacement is not None:
        # 发生拒绝：追加已接受 + 替换 token
        new_tokens = torch.cat([accepted_tokens, torch.tensor([replacement], device=context.device)])
    else:
        # 全部接受 + 从目标模型抽取一个额外 token
        n_accepted = max_draft_length
        bonus_logits = verify_logits[-1, :]  # 位置 t+K+1
        bonus_probs = F.softmax(bonus_logits / temperature, dim=-1)
        bonus_token = torch.multinomial(bonus_probs, 1)
        new_tokens = torch.cat([accepted_tokens, bonus_token])

    return new_tokens, n_accepted


class MockModel:
    """
    用于演示投机解码的模拟模型。
    在实际应用中，这些应该是真实的 Transformer 模型。
    """

    def __init__(self, vocab_size: int, hidden_size: int, is_large: bool = True):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.is_large = is_large
        # 简单的线性投影来模拟词表 logits
        self.proj = torch.nn.Linear(hidden_size, vocab_size)

        if is_large:
            print(f"  目标模型：词表 {self.vocab_size}，隐藏层 {hidden_size}（大模型）")
        else:
            print(f"  草稿模型：词表 {self.vocab_size}，隐藏层 {hidden_size}（小模型）")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        模拟前向传播 —— 仅将 token 嵌入投影到 logits。

        参数:
            token_ids: 形状 (batch, seq) token ID

        返回:
            logits: 形状 (batch, seq, vocab_size) 模拟 logits
        """
        batch, seq = token_ids.shape
        # 创建确定但非平凡的基于 token ID 的 logits
        embeddings = token_ids.unsqueeze(-1).expand(-1, -1, self.hidden_size).float()
        # 加入位置偏置：后续 token 影响后续位置
        positional_bias = torch.arange(seq, device=token_ids.device).unsqueeze(0).unsqueeze(-1)
        features = embeddings + 0.01 * positional_bias
        logits = self.proj(features)
        return logits

    def __call__(self, x):
        return self.forward(x)


# ------------------------------------------------------------------
# 示例用法：对比标准解码与投机解码
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    vocab_size = 1000
    context_tokens = torch.tensor([[1, 2, 3, 4, 5]])

    # 创建具有不同容量的模拟模型
    print("初始化模型...")
    draft_model = MockModel(vocab_size, hidden_size=64, is_large=False)
    target_model = MockModel(vocab_size, hidden_size=256, is_large=True)
    print()

    K = 4
    n_rounds = 10

    # --- 运行投机解码 ---
    print(f"运行投机解码（K={K}，轮数={n_rounds}）...")
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

        bonus_count = 1 if new_tokens.shape[0] > n_accepted else 0
        print(f"  第 {round_idx+1:2d} 轮：起草 {K}，接受 {n_accepted}+"
              f"{bonus_count} = {new_tokens.shape[0]} 个 token  "
              f"（已生成 token 数：{len(all_spec_tokens)}）")

    accept_rate = total_accepted / total_draft
    tokens_per_target_pass = len(all_spec_tokens) / n_rounds

    print()
    print("结果：")
    print(f"  总起草 token 数：{total_draft}")
    print(f"  总接受 token 数：{total_accepted}")
    print(f"  接受率：{accept_rate:.1%}")
    print(f"  总生成的 token 数：{len(all_spec_tokens)}")
    print(f"  目标模型调用次数：  {n_rounds} "
          f"（标准方式需要 {len(all_spec_tokens)} 次调用）")
    print(f"  有效加速比：{tokens_per_target_pass:.2f}x "
          f"（每次目标前向传播对应的 token 数）")
    print()

    # --- 对比：标准自回归（模拟）---
    print("对比：标准自回归将需要：")
    print(f"  {len(all_spec_tokens)} 次目标模型前向传播")
    print(f"  对比：投机解码需 {n_rounds} 次目标传播 + {total_draft} 次草稿传播")
    print(f"  （草稿模型约为目标模型的 {256/64:.0f} 分之一大小/速度更快）")
    draft_cost = total_draft / (256/64)
    total_cost_equiv = n_rounds + draft_cost
    print(f"  等效成本比：理论加速比 {n_rounds / total_cost_equiv:.2f}x")
```

---

## 深入探究

### 1. 基础定理：为什么投机解码是无损的

这是投机解码中最重要的结果，且常常被误解。拒绝采样机制不是启发式方法，而是一个精确的数学构造，保证输出分布与目标模型完全匹配。

```
目标模型的下一个 token 分布：q(x)
草稿模型的下一个 token 分布：r(x)

朴素方法（错误）：
  总是接受草稿 token。
  结果：输出服从 r(x)，而非 q(x) → 质量损失。

更好的方法（仍然错误）：
  如果 q(x) > 阈值则接受草稿 token。
  结果：输出是有偏的子集 → 质量损失。

正确方法（精确）：
  以 α(x) = min(1, q(x)/r(x)) 接受。
  若被拒绝，则从 p_adjusted(x) = max(0, q(x)-r(x)) / Z 中抽样。

  结果：输出**精确地**服从 q(x)。
  这不是近似——这是一个数学恒等式。
```

证明的核心在于：每一步中，算法要么接受草稿 token（对输出贡献 r(x) · q(x)/r(x) = q(x)），要么拒绝它并从残差分布中抽样（贡献恰好补足剩余的概率质量，使其总和等于正确的 q(x)）。分解 q(x) = q(x) · 1 = q(x) · [r(x)/r(x)] 的巧妙之处在于，拒绝采样精确地捕获了草稿提供的与目标需要的之间的差距。

### 2. 草稿模型设计：什么才是好的草稿模型？

草稿模型**不需要**和目标模型一样准确。它需要：

1. **快速**：通常比目标模型快 3-10 倍。如果草稿太慢，生成 K 个候选的时间会超过一次传播验证它们所节省的时间。

2. **多样性**：草稿必须产生合理的 token，即使它们不是目标模型会选择的确切 token。高接受率比 top-1 准确率更重要。

3. **良好校准**：草稿的概率估计应大致与目标模型的跟踪一致。如果草稿给某个 token 分配 99% 的概率而目标模型仅分配 0.1%，接受率会很低。

常见草稿模型选择：

```
草稿策略              接受率           部署复杂度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
蒸馏小模型            高（70-90%）     中等（需要监督微调蒸馏）
相同模型，更少层数     中等（50-70%）   低（仅使用子集即可）
n-gram 缓存           可变（20-80%）   无需训练（直接缓存）
Medusa 头部           中等（40-60%）   低（仅训练头部）
EAGLE 特征            高（60-85%）     中等（训练特征网络）
```

### 3. 温度对接受率的影响

温度会显著影响接受率：

```
低温度（T = 0.1）时：
  两个模型都将概率集中在最高排名的 token 上。
  如果它们对最高排名 token 一致 → α ≈ 1（总是接受）。
  如果它们不一致 → α ≈ 0（几乎总是拒绝）。
  二元结果，加速比方差很大。

高温度（T = 2.0）时：
  两个分布都更平坦。
  各 token 间的 q(x)/r(x) 比值更接近于 1。
  更稳定但特定 token 的接受率较低。

最佳区间：T = 0.5 到 T = 1.0
  足够的集中度以产生有意义的预测，
  同时足够的分散性以维持合理的接受率。
```

### 4. 投机解码的工程挑战

**KV 缓存管理：** 目标模型的 KV 缓存必须同时填充 K 个位置，但其中一些位置可能会被拒绝。高效的实现通过以下方式处理：

```
  1. 对所有 K+1 个位置填充 KV 缓存
  2. 保留前 n_accepted 个位置
  3. 从 KV 缓存中驱逐被拒绝的位置
  4. 额外 bonus token（位置 K+1）始终保留

  这需要仔细的缓存管理，以避免
  对已接受位置的 KV 状态重新计算。
```

**批处理投机解码：** 在为多个请求提供服务时，草稿 token 可以跨请求批处理，目标模型并行验证所有批次：

```
  请求 1: [草稿 token ×3]
  请求 2: [草稿 token ×5]
  请求 3: [草稿 token ×2]
  ... 填充至最大 K，在一次目标批次中验证
```

**异步草稿：** 更先进的实现让草稿模型在后台持续运行，而目标模型处理前一轮的结果，从而完全隐藏草稿的延迟。

### 5. 什么时候不应使用投机解码？

投机解码并非普遍有益：

```
❌ 低接受率场景：
   草稿模型与目标模型差异很大（例如不同领域）
   接受率降至约 40% 以下 → 加速比 < 1.5x → 不值得使用
   草稿模型的计算和通信开销超过了节省。

❌ 非常短的序列：
   如果仅生成 5-10 个 token，开销占主导地位。
   投机解码在 100+ token 的场景中表现最佳。

❌ 计算密集型工作负载：
   如果目标模型已经计算饱和（大批量服务），
   内存带宽不是瓶颈，因此投机解码帮助较小。

❌ 非自回归生成：
   投机解码依赖于自回归结构。
   对于掩码语言模型或扩散模型，需要使用不同的加速技术。
```

---

## 延伸阅读

- **Fast Inference from Transformers via Speculative Decoding**（Chen 等，2023）：https://arxiv.org/abs/2211.17192
- **Medusa: Simple LLM Inference Acceleration Framework**（Cai 等，2024）：https://arxiv.org/abs/2401.10774
- **Lookahead Decoding**（Fu 等，2024）：https://arxiv.org/abs/2402.02057
- **EAGLE: Speculative Sampling with Feature-Based Drafting**（Li 等，2024）：https://arxiv.org/abs/2406.16858
- **Speculative Decoding: A Survey**（综合综述）：https://arxiv.org/abs/2409.15385

---

_上一篇：[第 02 天 —— 混合专家模型](02-mixture-of-experts.md)  |  下一篇：[第 04 天 —— 测试时计算](04-test-time-compute.md)_