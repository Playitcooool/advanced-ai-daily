# Advanced AI Daily | AI 前沿日报

> Daily tutorials on cutting-edge AI | 每日更新的前沿 AI 教程
> LLM Architectures · Agent Systems · RL & Inference · Mathematical Foundations
> 大模型架构 · 智能体系统 · 强化学习与推理 · 数学基础

---

## 🇬🇧 English Tutorials

| # | Topic | Category | Time | Visual |
|:--|:------|:---------|:----:|:------:|
| 01 | [GRPO — Group Relative Policy Optimization](tutorials/en/01-grpo.md) | RL | ~15 min | Mermaid + GIF |
| 02 | [MoE — Mixture of Experts](tutorials/en/02-mixture-of-experts.md) | Architecture | ~20 min | Mermaid + GIF |
| 03 | [Speculative Decoding](tutorials/en/03-speculative-decoding.md) | Inference | ~15 min | Mermaid + GIF |
| 04 | [Test-Time Compute Scaling](tutorials/en/04-test-time-compute.md) | Inference | ~20 min | Mermaid + GIF |
| 05 | [Multi-Agent Reflection](tutorials/en/05-multi-agent-reflection.md) | Agent | ~20 min | Mermaid + GIF |
| 06 | [Quantization — TurboQuant & 1-bit LLMs](tutorials/en/06-quantization.md) | Inference | ~15 min | Mermaid + PNG |
| 07 | [RBF Attention](tutorials/en/07-rbf-attention.md) | Architecture | ~15 min | Mermaid + PNG |
| 08 | [Memory & KV Cache](tutorials/en/08-memory-kv-cache.md) | Inference | ~20 min | Mermaid + PNG |
| 09 | [Simple Self-Distillation (SSD)](tutorials/en/09-self-distillation.md) | Post-Training | ~20 min | Mermaid + GIF |
| 10 | [SRPO — Unifying GRPO & Self-Distillation](tutorials/en/10-sample-routing.md) | Post-Training | ~15 min | Mermaid + GIF |
| 11 | [Gradient Boosted Attention](tutorials/en/11-gradient-boosted-attention.md) | Attention | ~15 min | Mermaid + GIF |
| 12 | [Early Stopping via Confidence Dynamics](tutorials/en/12-early-stopping.md) | Reasoning | ~15 min | Mermaid + GIF |
| 13 | [Pluralistic Alignment](tutorials/en/13-pluralistic-alignment.md) | Alignment | ~15 min | Mermaid + GIF |
| 14 | [SUPERNOVA — Natural Instruction RL](tutorials/en/14-supernova.md) | RL | ~15 min | Mermaid + GIF |
| 15 | [HDPO — Meta-Cognitive Tool Use](tutorials/en/15-hdpo.md) | Agent | ~15 min | Mermaid + GIF |

## 🇨🇳 Chinese Tutorials

| # | Topic | Category | Time | Visual |
|:--|:------|:---------|:----:|:------:|
| 01 | [GRPO — 组相对策略优化](tutorials/zh/01-grpo.md) | 强化学习 | ~15 min | Mermaid + GIF |
| 02 | [MoE — 混合专家架构](tutorials/zh/02-mixture-of-experts.md) | 模型架构 | ~20 min | Mermaid + GIF |
| 03 | [投机解码](tutorials/zh/03-speculative-decoding.md) | 推理优化 | ~15 min | Mermaid + GIF |
| 04 | [推理时计算扩展](tutorials/zh/04-test-time-compute.md) | 推理优化 | ~20 min | Mermaid + GIF |
| 05 | [多智能体反思系统](tutorials/zh/05-multi-agent-reflection.md) | 智能体 | ~20 min | Mermaid + GIF |
| 06 | [模型量化 — TurboQuant 与 1-bit 大模型](tutorials/zh/06-quantization.md) | 推理优化 | ~15 min | Mermaid + PNG |
| 07 | [RBF 注意力 — 超越点积](tutorials/zh/07-rbf-attention.md) | 模型架构 | ~15 min | Mermaid + PNG |
| 08 | [记忆机制与 KV Cache 优化](tutorials/zh/08-memory-kv-cache.md) | 推理优化 | ~20 min | Mermaid + PNG |
| 09 | [简单自蒸馏 (SSD)](tutorials/zh/09-self-distillation.md) | 后训练 | ~20 min | Mermaid + GIF |
| 10 | [SRPO — 统一 GRPO 与自蒸馏](tutorials/zh/10-sample-routing.md) | 后训练 | ~15 min | Mermaid + GIF |
| 11 | [梯度提升注意力](tutorials/zh/11-gradient-boosted-attention.md) | 注意力机制 | ~15 min | Mermaid + GIF |
| 12 | [基于置信度动态的提前停止](tutorials/zh/12-early-stopping.md) | 推理模型 | ~15 min | Mermaid + GIF |
| 13 | [多元对齐：捕捉多样化人类偏好](tutorials/zh/13-pluralistic-alignment.md) | 对齐 | ~15 min | Mermaid + GIF |
| 14 | [SUPERNOVA — 自然指令强化学习](tutorials/zh/14-supernova.md) | 强化学习 | ~15 min | Mermaid + GIF |
| 15 | [HDPO — 元认知工具使用](tutorials/zh/15-hdpo.md) | 智能体 | ~15 min | Mermaid + GIF |

---

## What's Inside Each Tutorial | 每篇教程包含

Every tutorial (EN + ZH) includes:

- **YAML Frontmatter**: metadata, tags, prerequisites, reading time
- **Quick Reference**: core formula + one-liner code at a glance
- **Mermaid Diagrams**: interactive architecture flowcharts (rendered by GitHub)
- **LaTeX Math**: $$ equation blocks with term-by-term explanation
- **Runnable Python**: complete, typed code with `__main__` examples
- **Common Misconceptions**: what people get wrong about this concept
- **Exercises**: hands-on problems with collapsible answers
- **Verified References**: only real arXiv papers, no fabricated links

---

## Project Structure

```
advanced-ai-daily/
├── tutorials/
│   ├── en/               # Pure English tutorials (v4.0 format)
│   └── zh/               # Pure Chinese tutorials (v4.0 format)
├── gifs/                 # Animated GIFs (GitHub-native)
├── diagrams/             # Architecture diagrams (PNG fallback)
└── references/
    ├── verified-papers.json       # 25 verified arXiv papers
    └── keyword-database.json      # Topic coverage tracking
```

## Design Principles

- **Language Separation**: No mixed-language paragraphs. Each tutorial is 100% one language.
- **Depth First**: No ML 101. Straight to the frontier.
- **Visual First**: Every concept has Mermaid diagrams + GIF or PNG.
- **Code Included**: Complete, runnable Python with type hints.
- **Verified References**: Only real, confirmed arXiv papers — never fabricated.
- **Community-Driven**: Sourced from arXiv + Reddit (r/LocalLLaMA, r/MachineLearning, etc.).

## Quick Start

1. Pick a language and topic above
2. Open the tutorial — Quick Reference at top gives you the formula in 30s
3. Read the full Deep Dive section with code examples
4. Try the exercises at the bottom (answers are collapsible)
5. Say **"更新教程"** or **"今天更新什么"** to get a new daily tutorial

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

---

_Licensed under MIT · [GitHub Repo](https://github.com/Playitcooool/advanced-ai-daily)_
_30 tutorial files · 15 topics × 2 languages · v4.0 format_
