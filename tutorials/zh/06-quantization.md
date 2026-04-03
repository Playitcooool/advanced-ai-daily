# 第 06 天：模型量化——TurboQuant 与 1-bit 大语言模型

> **日期**：2026-04-03 | **难度**：高级 | **类别**：推理优化

> **架构示意图**：![量化架构](../diagrams/06-quantization.png)

---

## 一句话概要

量化技术将模型的权重和激活值从 16 位浮点数压缩至 8 位、4 位甚至 1 位整数，同时保持极小的精度损失。TurboQuant 和 Bonsai 1-bit 模型将这一技术推向极致——在笔记本电脑上以接近 FP16 的质量运行 27B 参数规模的模型。

---

## 量化的数学原理

### 均匀量化

将连续范围 [a, b] 映射到 N 个离散层级：

```
quantize(x, N):
  scale = (b - a) / (N - 1)         # 量化步长
  q(x) = round((x - a) / scale)      # 量化为整数
  dequantize(q) = q * scale + a      # 反量化回浮点数
```

量化误差的上界为 `|x - dequantize(quantize(x))| <= scale / 2`。步长越小（N 越大），误差越小。

### 量化层级对照

| 格式      | 位数 | 层级数    | 内存占用（70B 模型） | 典型用途                    |
|-----------|:----:|:---------:|:--------------------:|-----------------------------|
| FP16      | 16   | 65,536    | 140 GB               | 训练、高质量推理            |
| INT8      | 8    | 256       | 70 GB                | 生产环境服务，质量损失极小  |
| Q4_0      | 4    | 16        | 35 GB                | 本地推理（llama.cpp 默认）  |
| Q2_K      | 2    | 4-6       | 20-25 GB             | 内存受限设备                |
| **1-bit** | 1    | 2         | **约 10 GB**         | Bonsai 模型                 |

---

## 问题：为什么不直接用更少的位数？

### 1-bit 量化的困难所在

在 1-bit 量化中，权重仅有 +1 或 -1 两个取值（二值化）。一个普通的线性层：

```
y = W @ x        →  y ≈ sign(W) @ x  （但误差巨大！）
   [FP16, 65536 个层级]      [1-bit, 仅 2 个层级]
```

**核心挑战**：如何设计 W 的符号模式，使得 `sign(W) @ x` 能足够好地逼近 `W @ x`？最简单粗暴的方法——逐元素取符号——会造成严重的信息损失。

### TurboQuant 的核心洞察

不要一次性量化整个权重矩阵，而是分三步：

1. **分组分解**：将权重拆分为多个子组，各组使用不同的缩放尺度
2. **分组量化**：每个组独立计算量化范围 (a, b)
3. **跳过 KV 反量化**：在自回归解码阶段，约 90% 的 KV 缓存反量化计算可以跳过

### Bonsai 1-bit 模型

Bonsai 通过以下三项技术实现 1-bit 量化：

1. **二值权重**（+1/-1），每组配备独立的缩放系数
2. **量化后微调**——不是简单的训练后量化，而是量化感知训练（QAT）
3. **分组结构**：在二值约束下依然保持模型的表达能力

---

## 算法流程详解

```
┌──────────────────────────────────────────────┐
│              模型量化流水线                     │
└──────────────────────────────────────────────┘

  FP16 权重（16 位）
  │
  ▼
┌─────────────────────────────────┐
│ 第一步：按组分解权重               │
│  W = [组_1 | 组_2 | ...]         │
│  每组拥有独立的缩放系数             │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ 第二步：计算每组的缩放系数          │
│  scale_g = max(|W_g|) / (N/2-1) │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ 第三步：逐组量化                   │
│  q(W_g) = round(W_g / scale_g)  │
└───────────────┬─────────────────┘
                ▼
┌─────────────────────────────────┐
│ 第四步（可选）：量化感知微调        │
│  在前向传播中插入量化操作并微调      │
└─────────────────────────────────┘
```

### KV 缓存量化

在自回归解码过程中，KV 缓存随序列长度线性增长。TurboQuant 证明，**90% 的 KV 反量化计算可以被安全跳过**，方法如下：

1. 对频繁访问的索引预先进行反量化计算
2. 对较早的 token 使用低比特精度（2-bit），因为它们对注意力的贡献较小
3. 混合精度策略：最近 token 采用高精度，较远 token 采用低精度

---

## 代码实现

```python
import torch
import torch.nn.functional as F

def uniform_quantize(x: torch.Tensor, bits: int = 4):
    """
    均匀每张量量化

    参数:
        x: 输入张量（任意形状）
        bits: 量化位数（1-8）

    返回:
        q: 量化后的 int8 张量
        scale: 反量化缩放因子
        zero_point: 零点偏移
    """
    N = 2 ** bits  # 量化层级数量
    qmin = 0
    qmax = N - 1

    # 查找张量范围
    x_min = x.min().item()
    x_max = x.max().item()

    # 计算缩放因子和零点
    scale = (x_max - x_min) / (qmax - qmin) if x_max > x_min else 1.0
    zero_point = qmin - x_min / scale

    # 执行量化
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return q.to(torch.int8), scale, zero_point

def per_group_quantize(x: torch.Tensor, group_size: int, bits: int = 4):
    """
    分组量化——llama.cpp 和 TurboQuant 实际使用的方法

    每组包含 `group_size` 个元素，各组拥有独立的缩放系数。

    参数:
        x: 权重张量 [输出维度, 输入维度]
        group_size: 每个量化组的元素数量
        bits: 量化位数

    返回:
        q: 量化后的权重 [输出维度, 输入维度]
        scales: 各组缩放系数 [输出维度, 组数]
    """
    N = 2 ** bits
    qmin, qmax = 0, N - 1

    out_f, in_f = x.shape
    num_groups = (in_f + group_size - 1) // group_size

    q = torch.zeros_like(x, dtype=torch.int8)
    scales = torch.zeros(out_f, num_groups, device=x.device)

    for g in range(num_groups):
        start = g * group_size
        end = min((g + 1) * group_size, in_f)
        chunk = x[:, start:end]

        # 每组独立计算缩放系数
        group_min = chunk.min(dim=1, keepdim=True)[0]
        group_max = chunk.max(dim=1, keepdim=True)[0]
        group_scale = (group_max - group_min) / (qmax - qmin)
        group_scale = torch.clamp(group_scale, min=1e-8)

        zp = qmin - group_min / group_scale
        q_chunk = torch.clamp(torch.round(chunk / group_scale + zp), qmin, qmax)

        q[:, start:end] = q_chunk.to(torch.int8)
        scales[:, g:g+1] = group_scale.squeeze(dim=1, keepdim=True)

    return q, scales

class QuantizedLinear(torch.nn.Module):
    """
    量化线性层：权重量化，整数矩阵乘法

    权重在前向传播时动态反量化。
    """
    def __init__(self, in_features, out_features, bits=4, group_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        # 初始化并量化权重
        weight = torch.randn(out_features, in_features)
        q_w, scales = per_group_quantize(weight, group_size, bits)

        self.register_buffer("q_weight", q_w)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_point", torch.zeros(out_features,
                         max(1, (in_features + group_size - 1) // group_size)))

    def forward(self, x):
        # 前向传播时动态反量化
        out_f, in_f = self.out_features, self.in_features
        num_groups = self.scales.size(1)
        gs = self.group_size

        # 重建浮点权重
        weight = torch.zeros(out_f, in_f, device=x.device)
        for g in range(num_groups):
            start = g * gs
            end = min((g + 1) * gs, in_f)
            weight[:, start:end] = (
                self.q_weight[:, start:end].to(x.dtype) *
                self.scales[:, g:g+1]
            )

        return F.linear(x, weight)
```

---

## 量化感知训练（QAT）

训练后量化（PTQ）仅对预训练权重直接量化，不做任何额外训练。QAT 效果更好：

```python
def qat_finetune_step(model, x, y, optimizer, bits=4):
    """
    在前向传播中插入伪量化节点的训练步骤

    反向传播使用直通估计器（STE），
    因此梯度过量化操作流回原始浮点权重。
    """
    # 前向传播：伪量化
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                # 量化
                q_w, scale = uniform_quantize(param, bits=bits)
                # 反量化（通过 STE 保持可微）
                w_fp = q_w.float() * scale

                # 替换参数数据用于前向传播
                # （梯度通过 STE 流回原始浮点参数）
                param.data.copy_(w_fp)

    # 标准前向传播和反向传播
    pred = model(x)
    loss = F.cross_entropy(pred, y)
    loss.backward()
    optimizer.step()
```

---

## 方法对比：PTQ vs QAT vs 1-bit

| 方法                 | 位数 | 精度损失  | 训练成本       | 最佳适用场景           |
|----------------------|:----:|:---------:|:--------------:|------------------------|
| FP16（基准）         | 16   | 0%        | —              | 参考标准               |
| INT8 PTQ             | 8    | <0.5%     | 无（事后量化）  | 生产环境部署           |
| INT4 GGUF            | 4    | 1-3%      | 无             | 本地推理               |
| INT4 QAT             | 4    | <1%       | 轻量微调        | 质量敏感场景           |
| Bonsai 1-bit         | 1    | 3-5%      | 完整 QAT 训练  | 极端内存受限           |

TurboQuant 结合 KV 缓存跳过策略实现：

- **32K 上下文下解码速度提升 22.8%**，跳过了 90% 的 KV 反量化计算
- 在约 10% 更小内存的情况下达到接近 Q4_0 的精度

---

## 扩展阅读

- [TurboQuant: Accelerating 1-bit LLM Inference](https://arxiv.org/) —— Google 量化框架
- [Bonsai: 1-bit Large Language Models](https://arxiv.org/) —— 1-bit 训练技术
- [llama.cpp Quantization](https://github.com/ggml-org/llama.cpp/blob/master/docs/quantization.md) —— GGUF 格式规范
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) —— 逐层最优量化方法
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) —— 保护关键权重的量化方案

---

_上一篇：[Day 05 - 多智能体反思](05-multi-agent-reflection.md)  |  下一篇：[Day 07 - RBF 注意力](07-rbf-attention.md)_
