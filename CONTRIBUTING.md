# Contributing to Advanced AI Daily

## Adding a New Tutorial

1. Create `tutorials/NN-title.md` following existing templates
2. Create `animations/NN-title-animation.html` with interactive visualization
3. Add a row to the table in `README.md`
4. Cross-link: link to "previous" and "next" tutorials at the bottom

## Tutorial Format

Each tutorial should:
- Start with a one-sentence summary
- Include a diagram/flowchart (ASCII or image)
- Provide Python code examples with detailed comments
- Have a "深度讨论" section with critical analysis
- End with references to papers and related tutorials

## Development Workflow

```bash
cd advanced-ai-daily
git add .
git commit -m "feat: add tutorial NN - Topic"
git push
```

## Daily Update

The GitHub Actions workflow runs daily at midnight UTC to:
1. Fetch latest papers from arXiv
2. Analyze trending topics
3. Update README.md

To trigger manually: go to Actions tab → "Daily Tutorial Update" → Run workflow
