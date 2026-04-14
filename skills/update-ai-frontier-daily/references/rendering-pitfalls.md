# Rendering Pitfalls

Use this checklist whenever a tutorial includes Mermaid, KaTeX, or a GIF.

## GIFs

- Use English-only labels inside frames to avoid missing glyphs
- Keep labels short enough to avoid collisions
- Avoid placing arrows on top of text
- If several ideas compete in one frame, split the storyboard instead of shrinking everything
- Confirm the final asset exists on disk before linking it from markdown

## Mermaid

- Prefer simple GitHub-safe flowcharts
- Avoid overly dense graphs with many crossing edges
- Keep node labels concise
- Re-read the raw markdown for obvious syntax mistakes before publishing

## Math

- Use GitHub-safe block math syntax
- Prefer a few core equations over large unreadable derivations
- Explain each symbol in surrounding text
