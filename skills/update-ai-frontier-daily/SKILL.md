---
name: update-ai-frontier-daily
description: Update the user's `/Volumes/Samsung/Projects/advanced-ai-daily` repository when they ask for today's AI frontier update, especially requests like `请你更新今日份的ai前沿`, `更新今日份的ai前沿`, `今天更新什么`, or `update today's AI frontier`. Research a current frontier topic, create the visual asset first, write bilingual EN/ZH tutorials in the repo's category structure, update `README.md`, and commit or push when the request clearly implies publishing the daily update.
---

# Update AI Frontier Daily

## Overview

Use this skill for the user's daily `advanced-ai-daily` publishing workflow. The target repo is `/Volumes/Samsung/Projects/advanced-ai-daily`, not the current working directory.

Before editing, inspect the repo's current state and mirror its latest conventions instead of inventing a new format. Prefer existing scripts in `scripts/`, existing tutorial structure, and existing category folders.

## When This Skill Should Trigger

Trigger on requests such as:

- `请你更新今日份的ai前沿`
- `更新今日份的ai前沿`
- `今天更新什么`
- `更新教程`
- `update today's AI frontier`
- `publish today's advanced-ai-daily update`

Do not use this skill for simple summaries, code review, or unrelated repos.

## Workflow

### 1. Build Context From the Local Repo

Work inside `/Volumes/Samsung/Projects/advanced-ai-daily`.

- Read [references/repo-layout.md](references/repo-layout.md).
- Read `README.md` and inspect the newest tutorial files under `tutorials/en/` and `tutorials/zh/`.
- Determine the next tutorial number from the highest existing numbered tutorial.
- Reuse the existing category layout and writing style.
- Prefer the repo's existing scripts when they help:
  - `scripts/fetch_daily_topics.py`
  - `scripts/generate_tutorial.py`

### 2. Research Today's Topic

This workflow depends on current information, so verify with live sources instead of relying on memory.

- Use multiple current sources when available.
- Required primary sources for topic selection: arXiv, Hugging Face, and Reddit.
- Topic selection rule: do not finalize a daily topic unless all three primary sources support it.
- Topic preference rule: prefer foundational concepts that explain today's discussion hotspots, especially core math, architecture, training or inference algorithms, and systems ideas.
- Good examples of preferred topic shape: RAG, linear attention, speculative decoding, Mixture of Experts routing, KV cache management, preference optimization, agent planning loops.
- Avoid picking a topic only because a single new paper, model, benchmark, or product is trending if the writeup would be too tied to that one artifact.
- Exception: a concrete paper/model/benchmark can be selected when it introduces or crystallizes a concept that is likely to become durable knowledge, reusable beyond that single release, and teachable as a general mechanism.
- When both are available, prefer "hot concept explained" over "hot artifact summarized."
- arXiv requirement: verify the paper or project exists and is current enough to plausibly qualify as today's frontier topic.
- Hugging Face requirement: verify there is a matching Papers page, repo, model card, or similarly direct Hugging Face signal tied to the same topic.
- Reddit requirement: verify there is direct discussion or clear community traction for the same topic, preferably on r/LocalLLaMA, r/MachineLearning, or another clearly relevant subreddit.
- If Reddit evidence is weak, indirect, or missing, do not pick that topic yet; keep screening until you find a topic that satisfies arXiv + Hugging Face + Reddit together.
- Fallback sources such as Hacker News, GitHub trending/new repos, and the repo's existing paper references may help rank candidates, but they do not replace any of the three required primary sources above.
- Prefer topics that are both current and teachable.
- Prefer topics that remain useful weeks or months later as reusable mental models.
- Prefer topics that connect naturally to previous tutorials.
- Verify that any cited paper or repo actually exists before writing.
- In the final writeup or progress update, briefly state how the chosen topic satisfied arXiv + Hugging Face + Reddit so the selection is auditable the next day as well.

If network access is blocked by the sandbox, request escalation and continue.

### 3. Create the Visual First

Before writing tutorial markdown, generate the tutorial visual asset.

- Prefer a GIF under `gifs/` for concepts that benefit from animation.
- Use English-only labels inside figures to avoid missing glyphs.
- Verify the asset exists on disk before referencing it in markdown.
- Reuse existing animation patterns from the repo and from [references/rendering-pitfalls.md](references/rendering-pitfalls.md).
- Treat animation clarity as a hard requirement. If labels overlap, arrows cross text, or too many regions compete in the same frame, redesign the storyboard instead of only shrinking the font.

The markdown should point to an asset that already exists.

### 4. Write the Tutorials

Create both language versions:

- `tutorials/en/<category>/NN-slug.md`
- `tutorials/zh/<category>/NN-slug.md`

Requirements:

- Keep EN and ZH structure aligned.
- Match the repo's existing tutorial style and section ordering.
- Include real references only.
- Include runnable Python where appropriate.
- Include Mermaid and math only if they can render safely on GitHub.

When choosing a category, use the existing repo taxonomy instead of making a new one unless the content clearly does not fit any current category.

### 5. Update the Index

Update `README.md` so the new tutorial appears in both the English and Chinese sections.

- Add the new row in the correct category table.
- Use the correct relative tutorial path.
- Update summary counts at the bottom if needed.

### 6. Verify Before Publishing

Perform local verification before any commit or push.

- Confirm the new files exist in both languages.
- Confirm the visual asset exists and the tutorial references the correct path.
- Inspect the GIF for readability: no overlapping labels, no text on top of arrows or bars, and no overcrowded frame with several active ideas competing at once.
- Re-read the new markdown for broken Mermaid or KaTeX patterns.
- Check for obvious numbering, path, or category mistakes.

Load [references/rendering-pitfalls.md](references/rendering-pitfalls.md) whenever you add Mermaid, KaTeX, or GIF content.

### 7. Commit and Push

If the user's request clearly implies the normal daily publishing flow, commit the relevant changes in `advanced-ai-daily` and push them. Otherwise stop after local edits and verification.

Rules:

- Do not touch unrelated dirty worktree changes.
- Stage only the files for today's update.
- Use a descriptive commit message such as `Add Day NN: Topic (EN+ZH + GIF)`.
- If push succeeds and web verification is possible, verify the rendered GitHub Pages result for the new tutorial.

## References

- [references/repo-layout.md](references/repo-layout.md): current local repo structure and conventions
- [references/rendering-pitfalls.md](references/rendering-pitfalls.md): Hermes-derived GIF, Mermaid, KaTeX, and publishing pitfalls
