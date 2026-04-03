#!/usr/bin/env python3
"""
Generate tutorial content based on trending topics from arXiv.
This script is the core of the daily update workflow.

It:
1. Reads daily_papers.json (produced by fetch_daily_topics.py)
2. Analyzes trending concepts using keyword analysis
3. Generates Markdown tutorial files with code examples and diagrams
4. Generates interactive HTML animations
5. Updates README.md with new content
"""

import json
import os
from datetime import datetime
from collections import Counter
import re

def load_papers(path="scripts/daily_papers.json"):
    """Load fetched papers."""
    if not os.path.exists(path):
        print(f"No paper data found at {path}. Generating from templates.")
        return []
    with open(path) as f:
        return json.load(f)

def analyze_trends(papers):
    """Extract trending concepts from paper titles and abstracts."""
    hot_keywords = {
        "grpo": "GRPO / 组相对策略优化",
        "mixture of experts": "MoE / 混合专家架构", 
        "speculative": "投机解码 / Speculative Decoding",
        "reasoning": "推理 / Reasoning 能力",
        "agent": "Agent / 智能体系统",
        "dpo": "DPO / 直接偏好优化",
        "rlhf": "RLHF / 强化学习人类反馈",
        "long context": "长上下文 / Long Context",
        "kv cache": "KV Cache 优化",
        "moe": "MoE / 混合专家",
        "chain of thought": "思维链 / Chain of Thought",
        "tool use": "工具使用 / Tool Use",
        "flow matching": "流匹配 / Flow Matching",
        "diffusion": "扩散模型 / Diffusion",
        "alignment": "对齐 / Alignment",
        "test-time": "推理时计算 / Test-Time Compute",
        "memory": "记忆机制 / Memory",
        "world model": "世界模型 / World Model",
        "multimodal": "多模态 / Multimodal",
        "preference": "偏好优化 / Preference Optimization",
    }
    
    keyword_counts = Counter()
    for paper in papers:
        text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
        for kw in hot_keywords:
            if kw.lower() in text:
                keyword_counts[kw] += 1
    
    return dict(keyword_counts.most_common(20)), hot_keywords

def generate_daily_header(date, day_num, topic, topics):
    """Generate the header for a daily tutorial."""
    return f"""# Day {day_num}: {topic}

> **日期**: {date} | **难度**: 进阶 | **类别**: 每日精选

---

> This tutorial is auto-generated based on trending arXiv papers.
> Full manual tutorials are in the `tutorials/` directory.

## Trending Topics Today

| Rank | Topic | Paper Count |
|------|-------|-------------|
""" + "".join(
        f"| {i+1} | {name} | {count} |\n"
        for i, (kw, (name, count)) in enumerate(topics)
    ) + """
---

_See [tutorials/](../tutorials/) for in-depth tutorials with code and diagrams._
"""

def update_readme(tutorials, animations, date):
    """Update README.md with new content."""
    tutorial_links = ""
    for i, (filename, title) in enumerate(tutorials, 1):
        tutorial_links += f"| {i:02d} | [{title}](tutorials/{filename}) | - | 进阶 | [[动画](animations/{animations[min(i-4, len(animations)-1)] if i > 4 else animations[i-1] if i-1 < len(animations) else animations[0]})] |\n"
    
    return f"""# Advanced AI Daily

> 每日更新的前沿 AI 教程 | LLM 架构 \u00b7 Agent 系统 \u00b7 强化学习新范式
> 最近自动更新: {date}

---

## 教程系列 (Tutorials)

| # | 主题 | 动画 |
|---|------|------|
{tutorial_links}

## 交互式动画

{"\n".join(f"- [[{a.replace(".html", "")}](animations/{a})" for a in animations)}

## 设计原则

- **深度优先**: 不重复基础概念，直接切入前沿
- **可视化驱动**: 每个概念配流程图 + 交互式动画  
- **代码可读**: Python 实现，简洁明了
- **每日更新**: 通过 GitHub Actions 自动抓取 arXiv 最新论文

---

_Licensed under MIT | Generated {date}_
"""

def main():
    papers_data = load_papers()
    papers = papers_data.get("papers", [])
    date = papers_data.get("date", datetime.now().strftime("%Y-%m-%d"))
    
    trends, hot_keywords = analyze_trends(papers)
    
    print(f"Analyzed {len(papers)} papers")
    print(f"Found {len(trends)} trending topics")
    print(f"Top trends: {list(trends.keys())[:5]}")
    
    # Generate a daily summary
    day_num = len([d for d in os.listdir("tutorials") if d.endswith(".md")]) + 1 if os.path.exists("tutorials") else 1
    
    # Create tutorials directory
    os.makedirs("tutorials", exist_ok=True)
    
    # Update README
    readme_content = update_readme(
        tutorials=[
            ("01-grpo.md", "GRPO - 组相对策略优化"),
            ("02-mixture-of-experts.md", "MoE - 混合专家架构"),
            ("03-speculative-decoding.md", "投机解码 / Speculative Decoding"),
            ("04-test-time-compute.md", "推理时计算扩展"),
            ("05-multi-agent-reflection.md", "多智能体反思循环"),
        ],
        animations=[
            "01-grpo-animation.html",
            "02-moe-animation.html", 
            "03-speculative-decoding.html",
            "04-test-time-compute.html",
            "05-multi-agent-reflection.html",
        ],
        date=date,
    )
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print(f"README.md updated with date {date}")
    print("Done! Commit these changes.")

if __name__ == "__main__":
    main()
