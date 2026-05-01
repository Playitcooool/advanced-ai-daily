# advanced-ai-daily

Daily AI frontier tutorials — LLM architecture, agent systems, RL/inference, math foundations.

**Repo**: `/Volumes/Samsung/Projects/advanced-ai-daily`

## Workflow

Daily update (Day N → Day N+1):

1. **Research**: Run `python3 scripts/fetch_daily_topics.py` → check `scripts/daily_papers.json`
2. **Pick topic**: Must satisfy arXiv + Hugging Face + Reddit (3 required sources)
3. **Create GIF first**: Write `gifs/generate_day{NN}.py`, run with `.venv/bin/python gifs/generate_day{NN}.py`
4. **Write tutorials**: EN + ZH markdown in `tutorials/en/` + `tutorials/zh/` same category
5. **Update README.md**: Add row to both EN and ZH tables, update topic count
6. **Update verified-papers.json**: Add new paper entry
7. **Commit and push**

## Conventions

- Next tutorial number = highest existing number + 1
- GIF path: `gifs/NN-short-name.gif` (also embedded in repo)
- Tutorial path: `tutorials/en/<category>/NN-short-name.md`
- GIF animation: dark theme (#08111f bg), English labels only, one clear motion per animation
- Use `.venv/bin/python` for matplotlib scripts
- Language-separated: never mix EN/ZH in same file

## Scripts

- `scripts/fetch_daily_topics.py` — fetch latest arXiv papers
- `scripts/daily_papers.json` — output of fetch script
- `scripts/generate_tutorial.py` — optional tutorial generator

## Key Paths

```
tutorials/en/<category>/
tutorials/zh/<category>/
gifs/
references/verified-papers.json
references/keyword-database.json
README.md
```
