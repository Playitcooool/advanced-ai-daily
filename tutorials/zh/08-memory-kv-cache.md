# 第 08 天：记忆机制与 KV Cache 优化

> **日期**：2026-04-03 | **难度**：高级 | **类别**：推理优化

> **架构示意图**：![KV Cache 架构](../diagrams/08-memory-kv-cache.png)

---

## 一句话概要

在自回归生成过程中，KV 缓存（Key-Value Cache）存储所有历史 token 的键值对状态，并随序列长度线性增长。对于长上下文场景，KV 缓存的显存开销甚至**超过模型权重本身**。优化 KV 缓存是长上下文推理的关键瓶颈。

---

## 内存瓶颈在哪里

### KV 缓存的增长规律

```
标准自回归解码流程：
  第一步：    处理 prompt → 缓存 100 个 token 的 KV   → 缓存: 100 × d_kv
  第二步：    生成 token 101                        → 缓存: 101 × d_kv
  第三步：    生成 token 102                        → 缓存: 102 × d_kv
  ...
  第 1000 步：生成 token 1099                       → 缓存: 1099 × d_kv

KV 缓存显存 = 2 × 层数 × 头数 × 头维度 × 序列长度 × 2（FP16 每个元素）
             = O(seq_len)  ← 线性增长
```

对于 70B 参数模型搭配 128K 上下文长度，KV 缓存可超过 **50 GB**——接近模型权重大小的三分之一。

### 内存分配全景

```
┌──────────────────────────────────────────────────┐
│        内存分布（70B 模型，128K 上下文）           │
├──────────────────────────────────────────────────┤
│  模型权重 (FP16)          │ 140 GB  ████████████  │
│  KV 缓存 (FP16)           │  96 GB  ████████      │ ← 长上下文下的主要消耗！
│  激活值                   │   8 GB  ▌             │
│  优化器状态                │ 280 GB  (仅训练阶段)  │
└──────────────────────────────────────────────────┘
```

---

## 优化策略

### 策略一：KV 缓存量化

对缓存的 K 和 V 张量进行低精度量化，无需重新训练模型：

```python
class QuantizedKVCache:
    """
    带 INT8 量化的 KV 缓存

    将 KV 缓存内存减少 50%，质量损失可忽略不计。
    """
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.cache = {
            'k': torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=torch.int8),
            'v': torch.zeros(n_layers, max_seq_len, n_heads, head_dim, dtype=torch.int8),
        }
        self.scales_k = torch.ones(n_layers, n_heads, 1, 1)  # [层, 头, 1, 1]
        self.scales_v = torch.ones(n_layers, n_heads, 1, 1)
        self.length = 0

    def append(self, layer_id, k_new, v_new):
        """将新 token 的 KV 追加到缓存（同时进行量化）"""
        k_new = k_new.detach()
        v_new = v_new.detach()

        # 按头计算量化缩放系数
        self.scales_k[layer_id] = torch.abs(k_new).max(dim=-2, keepdim=True)[0] / 127.0
        self.scales_v[layer_id] = torch.abs(v_new).max(dim=-2, keepdim=True)[0] / 127.0

        # 执行量化
        self.cache['k'][layer_id, self.length] = torch.round(
            k_new / self.scales_k[layer_id]).clamp(-128, 127).to(torch.int8)
        self.cache['v'][layer_id, self.length] = torch.round(
            v_new / self.scales_v[layer_id]).clamp(-128, 127).to(torch.int8)
        self.length += 1

    def get(self, layer_id):
        """获取完整的 KV 缓存（反量化后）"""
        k = self.cache['k'][layer_id, :self.length].to(torch.float32) * self.scales_k[layer_id]
        v = self.cache['v'][layer_id, :self.length].to(torch.float32) * self.scales_v[layer_id]
        return k, v
```

### 策略二：选择性淘汰

不要将所有 token 平等存储。保留重要的，淘汰不重要的：

```python
class SelectiveKVCache:
    """
    仅保留注意力贡献较高的 token

    策略：
      1. 在生成过程中计算每个 token 的重要性分数
      2. 保留重要性最高的 top-k 个 token
      3. 淘汰低重要性的 token
    """
    def __init__(self, max_tokens, eviction_threshold=0.85):
        self.max_tokens = max_tokens
        self.threshold = eviction_threshold
        self.cache = {}  # layer_id -> (k, v) 元组
        self.importance = {}  # layer_id -> 重要性分数

    def update(self, layer_id, k, v, attention_weights):
        """带选择性淘汰的 KV 缓存更新"""
        current_len = k.size(1)  # 序列长度维度

        if current_len <= self.max_tokens:
            self.cache[layer_id] = (k, v)
            # 追踪重要性：记录每个 token 收到的最大注意力权重
            if layer_id not in self.importance:
                self.importance[layer_id] = torch.zeros(current_len, device=k.device)
            self.importance[layer_id] = torch.max(
                self.importance[layer_id],
                attention_weights.max(dim=1)[0]
            )
        else:
            # 淘汰最不重要的 token
            _, keep_indices = torch.topk(self.importance[layer_id], self.max_tokens)
            k_kept = k[:, keep_indices]
            v_kept = v[:, keep_indices]
            self.cache[layer_id] = (k_kept, v_kept)
            self.importance[layer_id] = self.importance[layer_id][keep_indices]
```

### 策略三：PagedAttention（vLLM）

vLLM 的核心创新：像操作系统页表一样管理 KV 缓存内存。

```
传统方法：
  为最大序列长度预分配连续内存 → 造成大量内存浪费

PagedAttention：
  将 KV 缓存拆分为固定大小的"页"（类似操作系统的 4KB 页面）
  物理内存可以不连续 → 彻底消除内存碎片
  通过逻辑到物理的页表映射序列位置到具体的内存块
```

```python
class PagedAttentionKVCache:
    """
    类 PagedAttention 风格的 KV 缓存管理

    类似于操作系统的虚拟内存机制：
      - 逻辑 KV 块映射到物理内存页
      - 无需预分配
      - 几乎消除碎片造成的内存浪费
    """
    def __init__(self, page_size=16, max_num_pages=10000, head_dim=128, n_heads=32):
        self.page_size = page_size
        self.head_dim = head_dim
        self.n_heads = n_heads

        # 物理内存池（非连续）
        self.physical_k = torch.zeros(max_num_pages, page_size, n_heads, head_dim)
        self.physical_v = torch.zeros(max_num_pages, page_size, n_heads, head_dim)
        self.free_pages = list(range(max_num_pages))

        # 每个请求的页表
        self.page_tables = {}  # request_id -> [页索引0, 页索引1, ...]

    def allocate_page(self, request_id):
        """为请求分配一个新的物理页"""
        page_idx = self.free_pages.pop(0)
        if request_id not in self.page_tables:
            self.page_tables[request_id] = []
        self.page_tables[request_id].append(page_idx)
        return page_idx

    def get_kv_for_position(self, request_id, pos, layer_id):
        """通过页表查找获取指定位置的 KV 值"""
        page_idx = self.page_tables[request_id][pos // self.page_size]
        pos_in_page = pos % self.page_size
        k = self.physical_k[page_idx, pos_in_page:pos_in_page+1]
        v = self.physical_v[page_idx, pos_in_page:pos_in_page+1]
        return k, v
```

### 策略四：混合精度 KV 缓存

较新的 token 在注意力计算中更重要，因此以更高精度存储：

```
[最近 256 个 token]       → FP16  （完整精度）
[第 256-4096 个 token]    → INT8  （质量良好）
[第 4096 个及以后]         → INT4  （对于旧上下文可接受）
```

```python
class MixedPrecisionKVCache:
    """
    分层 KV 缓存：新 token 使用高精度，旧 token 使用低精度
    """
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len,
                 tier_config=None):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.current_len = 0

        if tier_config is None:
            # 默认三层配置
            tier_config = [
                (256, torch.float16),   # 最近：FP16
                (4096, torch.int8),     # 中期：INT8
                (max_seq_len, torch.int8),  # 早期：INT4（示例用 INT8 替代）
            ]
        self.tiers = tier_config

    def store(self, layer_id, k, v):
        """根据位置将 KV 值存入对应的精度层级"""
        batch_size, seq_len = k.shape[0], k.shape[1]
        self.current_len += seq_len
        # 根据 token 的位置路由到对应的精度层级
        # （具体实现细节省略）
```

---

## TurboQuant KV 跳过优化

TurboQuant 对于 KV 缓存的核心发现：**约 90% 的 KV 反量化计算可以被跳过**。

### 原理分析

在自回归解码过程中：

- 只有**最后一个 token** 需要与所有历史 token 重新计算注意力
- 对于更早的 token，它们的注意力模式在相邻解码步骤之间变化非常缓慢
- 因此可以**跳过对大部分历史 KV 条目的反量化和重新评分**

### 算法实现

```python
def turbo_kv_decode(last_k, last_v, kv_cache_quantized,
                    threshold=0.1, window_size=64):
    """
    TurboQuant KV 跳过优化（解码阶段）

    参数:
        last_k: [1, heads, dim] 最新的 key
        last_v: [1, heads, dim] 最新的 value
        kv_cache_quantized: 量化后的 KV 缓存
        threshold: 相似度阈值，低于此阈值的 skip
        window_size: 始终完整计算的最近 token 数量
    """
    seq_len = kv_cache_quantized.shape[0]

    # 滑动窗口内的最近 token 始终完整计算注意力
    if seq_len <= window_size:
        return full_attention(last_k, kv_cache_quantized)

    # 对于较旧的 token，检查注意力模式是否发生了显著变化
    # 仅在变化超过阈值时才进行反量化和重新评分
    # 这避免了在贡献稳定的历史 token 上执行冗余的反量化操作
```

**效果**：在 32K 上下文下实现 **22.8% 的解码速度提升**，质量损失几乎为零。

---

## 各方法对比

| 方法                 | 内存节省    | 精度损失   | 需重新训练 | 复杂度   |
|----------------------|:-----------:|:----------:|:----------:|:--------:|
| INT8 KV 缓存         | 50%         | <0.1%      | 否         | 低       |
| INT4 KV 缓存         | 75%         | ~0.3%      | 否         | 低       |
| 选择性淘汰           | 60-80%      | 0.5-1%     | 否         | 中       |
| PagedAttention       | <5%*        | 0%         | 否         | 中       |
| 混合精度             | 40-60%      | ~0.2%      | 否         | 中       |
| Turbo KV 跳过        | 0%（加速）  | <0.1%      | 否         | 高       |

*PagedAttention 主要解决内存碎片问题，而非减少原始数据大小。它是一种内存管理优化。

---

## 扩展阅读

- [TurboQuant: Accelerating LLM Inference](https://www.reddit.com/r/LocalLLaMA/comments/1s62g5v/) —— TurboQuant 详解
- [PagedAttention in vLLM](https://vllm.ai/) —— vLLM 的核心 KV 缓存优化技术
- [KVCache-Transformer: A Survey](https://arxiv.org/) —— 全面综述
- [StreamingLLM](https://arxiv.org/abs/2309.17453) —— 注意力锚点与缓存淘汰机制
- [GEAR](https://arxiv.org/abs/2403.05527) —— KV 缓存的低秩与稀疏近似

---

_上一篇：[Day 07 - RBF 注意力](07-rbf-attention.md)  |  下一篇：--_
