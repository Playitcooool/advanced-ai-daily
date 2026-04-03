# Contributing | 贡献指南

## Adding a New Tutorial

1. Run the `daily-ai-tutorial` skill or say "更新教程"
2. The skill will generate BOTH `tutorials/en/NN-topic.md` AND `tutorials/zh/NN-topic.md`
3. Add a row to both tables in `README.md`
4. Cross-link prev/next in each file's nav

## Tutorial Template

Each tutorial consists of TWO independent files:

**`tutorials/en/NN-topic.md`** -- PURE ENGLISH
- All headings, body text, code comments in English only
- No Chinese characters (except in technical terms like "GRPO" etc.)

**`tutorials/zh/NN-topic.md`** -- PURE CHINESE
- All headings, body text, code comments in Chinese only
- No English paragraphs

## Structure

```markdown
# Day NN: Topic Title

> **Date**: YYYY-MM-DD
> **Difficulty**: Advanced
> **Category**: Category

> **Visual**: [GIF/Diagram](../gifs/NN-name.gif)

---

## One-Line Summary

...

## Why Do We Need This?

...

## Algorithm

```
[ASCII flowchart]
```

### The Math

...

## Code Implementation

```python
# Comments match the file's language
```

## Deep Dive

...

## Further Reading

...
```
