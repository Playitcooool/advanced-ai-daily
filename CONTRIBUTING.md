# Contributing | 贡献指南

## Adding a New Tutorial

1. Create `tutorials/NN-title.md` following the bilingual template
2. Generate a GIF animation in `gifs/NN-name.gif` using `scripts/gen_gif.py`
3. Add a row to the table in `README.md`
4. Cross-link: link to "prev" and "next" at the bottom

## Tutorial Template | 教程模板

```markdown
# Day NN: Title EN
# 第 NN 天: Title CN

> **Date**: YYYY-MM-DD  **Difficulty**: Advanced
> **Watch**: ![Animation](../gifs/NN-name.gif)

## One-Line Summary | 一句话总结
EN summary... CN 总结...

## Algorithm | 算法
(diagram + math)

## Code | 代码
(python with bilingual comments)

## Deep Dive | 深度讨论
(critical analysis)

## Further Reading | 扩展阅读
(links to papers)
```

## Generating GIFs | 生成动画

```bash
cd scripts
python gen_gif.py --topic grpo
python gen_gif.py --all  # generate all
```

GIFs are saved to `gifs/` directory and referenced in tutorials via relative paths.

## Development | 开发

```bash
cd advanced-ai-daily
git add .
git commit -m "feat: add tutorial NN - Topic EN/CN"
git push
```
