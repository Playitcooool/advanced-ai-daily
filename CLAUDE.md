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
4. **Write tutorials**: EN + ZH markdown in `tutorials/en/` + `tutorials/zh/` same category
5. **Self-review markdown** (required before commit):
   - [ ] Mermaid diagram renders without broken syntax
   - [ ] All LaTeX math has matching `$$` open/close pairs
   - [ ] Python code has correct imports and runs without error
   - [ ] EN and ZH files have same section structure
   - [ ] No mixed language in same file
   - [ ] GIF path in markdown matches actual file on disk
   - [ ] All URLs are real and reach the paper
6. **Update README.md**: Add row to both EN and ZH tables, update topic count
7. **Commit and push**: If push fails, commit locally and tell the user

## Conventions

- Next tutorial number = highest existing number + 1
- GIF path: `gifs/NN-short-name.gif`
- Tutorial path: `tutorials/en/<category>/NN-short-name.md`
- Language-separated: never mix EN/ZH in same file

## Topic Selection

**Good candidates**: RL training methods, agent systems, inference optimization, MoE/routing, attention mechanisms, distillation, alignment, model architecture

**Avoid**: Narrow domain ML (medical, agriculture, hardware-specific) unless it illustrates a general mechanism

**Required**: arXiv paper with current ID; must be confirmable via arXiv API or daily_papers.json

## Scripts

- `scripts/fetch_daily_topics.py` — fetch latest arXiv papers from cs.LG, cs.CL, cs.AI
- `scripts/daily_papers.json` — output of fetch script
- `references/keyword-database.json` — tracks which topics are already covered

## Key Paths

```
tutorials/en/<category>/
tutorials/zh/<category>/
gifs/
references/keyword-database.json
README.md
```
