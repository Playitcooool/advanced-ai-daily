#!/usr/bin/env python3
"""
Add Quick Quiz section to tutorials that don't have one yet.
Run with: .venv/bin/python scripts/add_quiz.py [--all | --day NN]
"""

import os
import argparse
from pathlib import Path

def get_bucket_label(path):
    if 'rl-training' in path or 'distillation' in path or 'alignment' in path:
        return 'training or alignment'
    elif 'inference' in path:
        return 'inference algorithm'
    elif 'moe' in path or 'routing' in path:
        return 'routing mechanism'
    elif 'attention' in path or 'architecture' in path:
        return 'architectural mechanism'
    elif 'agent' in path or 'multimodal' in path:
        return 'agent system'
    elif 'memory' in path or 'quantization' in path:
        return 'memory or efficiency optimization'
    return 'core mechanism'


def add_quiz_to_file(md_path):
    with open(md_path) as f:
        content = f.read()

    if '## Quick Quiz' in content:
        return False

    if content.rstrip().endswith('---'):
        content = content.rstrip()[:-3].rstrip()
        suffix = '\n---\n'
    else:
        suffix = '\n---\n'

    label = get_bucket_label(str(md_path))
    quiz_lines = f"""
---

## Quick Quiz

Test your understanding of this topic.

### Q1. What is the core mechanism described in this tutorial?

- A. A new attention variant
- B. A training or inference algorithm
- C. A hardware optimization
- D. A dataset format

<details>
<summary>Reveal Answer</summary>

**Answer: B** — This tutorial focuses on a {label}.

*Explanation varies by tutorial — see the Core Insight section for the key takeaway.*

</details>

### Q2. When does this approach work best?

- A. Only on very large models
- B. Only on small models
- C. Under specific conditions detailed in the tutorial
- D. Always, regardless of setup

<details>
<summary>Reveal Answer</summary>

**Answer: C** — The tutorial describes specific conditions and tradeoffs. Review the "Why This Matters" and "Limitations" sections.

</details>

### Q3. What is the main takeaway?

- A. Use this instead of all other approaches
- B. This is a niche optimization with no practical use
- C. A specific mechanism with clear use cases and tradeoffs
- D. This has been superseded by a newer method

<details>
<summary>Reveal Answer</summary>

**Answer: C** — Every tutorial in this repo focuses on a specific mechanism with its own tradeoffs. Check the One-Line Summary at the top and the "What [Topic] Teaches Us" section at the bottom.

</details>
"""

    with open(md_path, 'w') as f:
        f.write(content + suffix + quiz_lines)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--day", type=int)
    args = parser.parse_args()

    base = Path("tutorials")
    added = 0

    for lang in ["en", "zh"]:
        for bucket in ["work", "act", "learn"]:
            bucket_dir = base / lang / bucket
            if not bucket_dir.exists():
                continue
            for subdir in bucket_dir.iterdir():
                if not subdir.is_dir():
                    continue
                for md in subdir.glob("*.md"):
                    day_num = int(md.stem.split("-")[0])
                    if args.day and args.day != day_num:
                        continue
                    if add_quiz_to_file(md):
                        print(f"  added quiz: {md}")
                        added += 1

    print(f"\nDone — added {added} quiz sections")


if __name__ == "__main__":
    main()
