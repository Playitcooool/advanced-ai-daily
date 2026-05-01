# advanced-ai-daily

Daily AI frontier tutorials — LLM architecture, agent systems, RL/inference, math foundations.

**Repo**: `/Volumes/Samsung/Projects/advanced-ai-daily`

## Workflow

Daily update (Day N → Day N+1):

1. **Research**: Run `python3 scripts/fetch_daily_topics.py` → check `scripts/daily_papers.json`
   - Filter: only cs.LG, cs.CL, cs.AI papers; ignore cs.LG-only medical/agricultural papers
   - Prefer papers with arXiv ID in 26XX format (2026)
   - Prefer topics that connect to existing categories
2. **Pick topic**: Must have clear arXiv signal + community relevance. If network blocked, use daily_papers.json only and note the limitation.
3. **Create GIF first**: Write `gifs/generate_day{NN}.py`, run with `.venv/bin/python gifs/generate_day{NN}.py`
   - Dark theme (#08111f bg), English labels only, one clear motion per animation
   - Check recent GIFs to avoid repeating the same animation logic
4. **Write tutorials**: EN + ZH markdown in `tutorials/en/work/` + `tutorials/zh/work/` (or act/learn)
   - Add Quick Quiz section before final `---`
   - Bucket: Work=inference/architecture/memory/routing/attention, Act=agent/multimodal, Learn=rl-training/distillation/alignment
5. **Self-review markdown** (required before commit):
   - [ ] Mermaid diagram renders without broken syntax
   - [ ] All LaTeX math has matching `$$` open/close pairs
   - [ ] Python code has correct imports and runs without error
   - [ ] EN and ZH files have same section structure
   - [ ] No mixed language in same file
   - [ ] GIF path in markdown matches actual file on disk
   - [ ] All URLs are real and reach the paper
   - [ ] Quick Quiz section is present
6. **Generate OG image**: `.venv/bin/python scripts/generate_og_images.py --day NN` → saves `assets/og/og-dayNN.png`
7. **Update README.md**: Add row to both EN and ZH tables under correct bucket, update topic count
8. **Commit and push**: If push fails, commit locally and tell the user

## 3-Bucket Structure

| Bucket | Contents | Examples |
|--------|----------|----------|
| **Work** | Inference, architecture, memory, routing, attention | Speculative Decoding, MoE, KV Cache |
| **Act** | Agents, tools, multimodal | Multi-Agent Reflection, Parallel Tool Calling |
| **Learn** | RL training, distillation, alignment | GRPO, Self-Distillation, Pluralistic Alignment |

## Conventions

- Next tutorial number = highest existing number + 1
- GIF path: `gifs/NN-short-name.gif`
- Tutorial path: `tutorials/en/work/<subcategory>/NN-slug.md`
- Language-separated: never mix EN/ZH in same file
- OG images: `assets/og/og-dayNN.png`
- GIF viewer: `gifs/viewer.html?gif=<path>`

## Topic Selection

**Good candidates**: RL training methods, agent systems, inference optimization, MoE/routing, attention mechanisms, distillation, alignment, model architecture

**Avoid**: Narrow domain ML (medical, agriculture, hardware-specific) unless it illustrates a general mechanism

**Required**: arXiv paper with current ID; must be confirmable via arXiv API or daily_papers.json

## Scripts

- `scripts/fetch_daily_topics.py` — fetch latest arXiv papers from cs.LG, cs.CL, cs.AI
- `scripts/daily_papers.json` — output of fetch script
- `scripts/generate_og_images.py` — generate OG preview images (`--all` or `--day NN`)
- `scripts/add_quiz.py` — add Quick Quiz to tutorials (`--all` or `--day NN`)
- `.github/workflows/quality-checks.yml` — CI lint (Mermaid, LaTeX pairs, language separation)

## Key Paths

```
tutorials/en/work/         # inference/, architecture/, moe/, attention/, memory/, quantization/, routing/
tutorials/en/act/          # agent/, multimodal/
tutorials/en/learn/        # rl-training/, distillation/, alignment/
tutorials/zh/work/
tutorials/zh/act/
tutorials/zh/learn/
gifs/                      # *.gif, viewer.html
assets/og/                 # og-dayNN.png
references/keyword-database.json
README.md
CLAUDE.md
```
