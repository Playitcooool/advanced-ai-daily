# Repo Layout

`advanced-ai-daily` stores bilingual tutorials under `tutorials/en/` and `tutorials/zh/`, organized by category. New daily content should preserve the existing category structure instead of creating ad hoc top-level folders.

## Key Paths

- `README.md`: category index for both languages
- `tutorials/en/<category>/`: English tutorials
- `tutorials/zh/<category>/`: Chinese tutorials
- `gifs/`: animated visual assets and generation scripts
- `scripts/`: helper scripts for topic fetching and generation
- `references/`: paper metadata and repo-specific references

## Conventions

- Tutorial filenames use `NN-slug.md`
- English and Chinese files should mirror each other
- Visual assets should exist before markdown references them
- README rows must be updated in both language sections
- Topic numbering should advance from the current highest day number
