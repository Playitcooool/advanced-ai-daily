#!/usr/bin/env python3
"""
Generate OG (social preview) images for tutorial pages.
Each image shows: topic name, category, day number, and a color-coded banner.
Run with: .venv/bin/python scripts/generate_og_images.py [--all | --day NN]
"""

import os
import sys
import argparse
from pathlib import Path

BACKGROUND = "#0d1117"
CARD_BG = "#161b22"
TEXT = "#e6edf3"
MUTED = "#8b949e"
ACCENT_WORK = "#5db2ff"
ACCENT_ACT = "#37d39d"
ACCENT_LEARN = "#ffb454"
BORDER = "#30363d"


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def get_colors(bucket):
    if bucket == "Work":
        return hex_to_rgb(ACCENT_WORK)
    elif bucket == "Act":
        return hex_to_rgb(ACCENT_ACT)
    return hex_to_rgb(ACCENT_LEARN)


def generate_og(title, day, bucket, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        print("matplotlib not available, skipping OG image generation")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Card background
    card = FancyBboxPatch(
        (0.05, 0.1), 0.9, 0.8,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=CARD_BG, edgecolor=BORDER, linewidth=1.5,
    )
    ax.add_patch(card)

    # Accent bar
    r, g, b = get_colors(bucket)
    accent = FancyBboxPatch(
        (0.05, 0.85), 0.9, 0.05,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=(r/255, g/255, b/255, 0.9),
    )
    ax.add_patch(accent)

    # Day badge
    day_text = f"Day {day:02d}"
    ax.text(0.12, 0.76, day_text, color=MUTED, fontsize=13, fontweight="bold", va="top")

    # Bucket label
    ax.text(0.88, 0.76, bucket, color=(r/255, g/255, b/255), fontsize=12,
           fontweight="bold", va="top", ha="right")

    # Title
    ax.text(0.12, 0.68, title, color=TEXT, fontsize=20, fontweight="bold",
           va="top", ha="left")

    # Footer
    ax.text(0.12, 0.18, "Advanced AI Daily", color=MUTED, fontsize=11)
    ax.text(0.88, 0.18, "playitcooool.github.io/advanced-ai-daily",
           color=MUTED, fontsize=10, ha="right")

    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches='tight',
               facecolor=BACKGROUND, edgecolor='none')
    plt.close(fig)
    print(f"  saved {output_path}")


FIG_SIZE = (12, 6.3)
FIG_DPI = 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--day", type=int)
    args = parser.parse_args()

    output_dir = Path("assets/og")
    output_dir.mkdir(parents=True, exist_ok=True)

    tutorials = [
        ("GRPO — Group Relative Policy Optimization", 1, "Learn"),
        ("MoE — Mixture of Experts", 2, "Work"),
        ("Speculative Decoding", 3, "Work"),
        ("Test-Time Compute Scaling", 4, "Work"),
        ("Multi-Agent Reflection", 5, "Act"),
        ("Quantization — TurboQuant & 1-bit LLMs", 6, "Work"),
        ("RBF Attention", 7, "Work"),
        ("Memory & KV Cache", 8, "Work"),
        ("Simple Self-Distillation (SSD)", 9, "Learn"),
        ("SRPO — Unifying GRPO & Self-Distillation", 10, "Work"),
        ("Gradient Boosted Attention", 11, "Work"),
        ("Early Stopping via Confidence Dynamics", 12, "Work"),
        ("Pluralistic Alignment", 13, "Learn"),
        ("SUPERNOVA — Natural Instruction RL", 14, "Work"),
        ("HDPO — Meta-Cognitive Tool Use", 15, "Act"),
        ("Routing Distraction", 16, "Work"),
        ("ClawBench — Everyday Online Tasks for Agents", 17, "Act"),
        ("Qianfan-OCR — Layout-as-Thought for Document Parsing", 18, "Act"),
        ("Looped Language Models", 19, "Work"),
        ("Adaptive Reasoning Budgets", 20, "Work"),
        ("Parallel Tool Calling", 21, "Act"),
        ("Parallel Drafting", 22, "Work"),
        ("Select to Think", 23, "Work"),
        ("Exploration Hacking", 24, "Learn"),
        ("Synthetic Computers at Scale", 25, "Act"),
        ("PRISM — Pre-alignment via On-policy Distillation", 26, "Learn"),
        ("LightKV — Lightweight KV Cache for LVLMs", 27, "Work"),
        ("SpecKV — Adaptive Speculative Decoding", 28, "Work"),
        ("OpenSeeker-v2 — Frontier Search Agents via Simple SFT", 29, "Act"),
    ]

    for title, day, bucket in tutorials:
        out = output_dir / f"og-day{day:02d}.png"
        if args.all or args.day == day:
            generate_og(title, day, bucket, out)
