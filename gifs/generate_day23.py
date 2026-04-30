import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


WIDTH = 8
HEIGHT = 5
FRAMES = 48
BG = "#08111f"
PANEL = "#101a2b"
TEXT = "#e6edf7"
MUTED = "#8ba2c7"
BLUE = "#5db2ff"
GREEN = "#37d39d"
ORANGE = "#ffb454"
RED = "#ff6b6b"
PURPLE = "#b388ff"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def stage(frame: int, start: int, end: int) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / float(end - start)


def rounded_box(ax, x, y, w, h, color, alpha=1.0, radius=0.03, lw=1.2):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=color,
        edgecolor=(1, 1, 1, 0.08),
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x1, y1, x2, y2, color, alpha=1.0, width=1.8):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            lw=width,
            color=color,
            alpha=alpha,
            shrinkA=6,
            shrinkB=8,
        ),
    )


def draw_topk_panel(ax, progress: float):
    """Show SLM top-K candidates containing LLM's preferred token."""
    alpha = clamp(progress * 1.4)
    rounded_box(ax, 0.05, 0.10, 0.38, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.88, "SLM Top-K Candidates", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.84, "token distribution over K proposals", color=MUTED, fontsize=9.2, alpha=alpha)

    # Candidate bars
    candidates = [
        ("token_7", 0.31, RED),
        ("token_3", 0.24, BLUE),
        ("token_9", 0.18, BLUE),
        ("token_1", 0.12, MUTED),
        ("token_5", 0.09, MUTED),
    ]
    labels = ["#1", "#2", "#3", "#4", "#5"]

    for idx, (label, prob, color) in enumerate(candidates):
        y_base = 0.68 - idx * 0.11
        bar_alpha = clamp(progress * 2.2 - idx * 0.22)
        if bar_alpha <= 0:
            continue
        rounded_box(ax, 0.10, y_base, 0.10, 0.06, "#16243b", alpha=0.9 * bar_alpha, radius=0.02, lw=0.8)
        rounded_box(ax, 0.10, y_base, 0.10 * prob * 5, 0.06, color, alpha=0.80 * bar_alpha, radius=0.02, lw=0.0)
        ax.text(0.21, y_base + 0.03, label, color=TEXT, fontsize=9.5, fontweight="bold", va="center", alpha=bar_alpha)
        ax.text(0.07, y_base + 0.03, labels[idx], color=MUTED, fontsize=9, va="center", alpha=bar_alpha)

    # Highlight
    if progress > 0.3:
        hl_alpha = clamp(progress * 1.6 - 0.5) * 0.9
        rounded_box(ax, 0.09, 0.56, 0.22, 0.09, "#1a3a5c", alpha=hl_alpha * 0.7, radius=0.03, lw=1.8)
        ax.text(0.20, 0.605, "LLM choice inside", color=GREEN, fontsize=9.5, fontweight="bold", ha="center", va="center", alpha=hl_alpha)


def draw_selection_panel(ax, progress: float):
    """Show S2T-LOCAL reranking the top-K into the correct choice."""
    alpha = clamp(progress * 1.4 - 0.08)
    rounded_box(ax, 0.47, 0.10, 0.48, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.49, 0.88, "S2T-LOCAL Reranker", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.49, 0.84, "distilled selection logic without LLM", color=MUTED, fontsize=9.2, alpha=alpha)

    # Selection circles
    local = clamp(progress * 2.0 - 0.2)
    for idx, (x, color, label) in enumerate([
        (0.54, RED, "rejected"),
        (0.64, BLUE, "rejected"),
        (0.74, GREEN, "selected"),
        (0.84, BLUE, "rejected"),
        (0.94, BLUE, "rejected"),
    ]):
        circle_alpha = clamp(progress * 2.4 - idx * 0.14) * 0.85
        if circle_alpha <= 0:
            continue
        pulse = 0.88 + 0.12 * math.sin(local * math.pi * 2 - idx * 0.4)
        ax.add_patch(patches.Circle((x, 0.68), 0.045, color=color, alpha=circle_alpha * pulse))
        ax.text(x, 0.68, label[0], color=("#1a1a1a" if color == GREEN else TEXT), fontsize=8.5, fontweight="bold", ha="center", va="center", alpha=circle_alpha)

    # Arrow from candidates to selection
    if local > 0:
        arrow(ax, 0.24, 0.68, 0.50, 0.68, MUTED, alpha=local * 0.7, width=1.4)

    # Output token
    out_local = clamp(progress * 2.4 - 0.5)
    if out_local > 0:
        rounded_box(ax, 0.62, 0.42, 0.20, 0.12, "#17314a", alpha=0.92 * out_local, radius=0.03, lw=0.8)
        ax.text(0.72, 0.48, "token_3", color=GREEN, fontsize=11.5, fontweight="bold", ha="center", va="center", alpha=out_local)
        arrow(ax, 0.74, 0.55, 0.74, 0.54, GREEN, alpha=out_local, width=1.8)

    # Performance bar
    bar_alpha = clamp(progress * 1.8 - 0.4) * 0.9
    rounded_box(ax, 0.52, 0.22, 0.38, 0.07, "#132035", alpha=bar_alpha, radius=0.02, lw=0.8)
    fill = 0.38 * (0.82 + 0.18 * math.sin(progress * math.pi))
    rounded_box(ax, 0.52, 0.22, fill, 0.07, GREEN, alpha=bar_alpha * 0.85, radius=0.02, lw=0.0)
    ax.text(0.71, 0.255, "+24.1% vs greedy", color=TEXT, fontsize=9, fontweight="bold", ha="center", va="center", alpha=bar_alpha)


def build_frame(frame: int):
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.05, 0.95, "Select to Think", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.05, 0.90, "SLM reranking via local sufficiency: LLM's choice lives in top-K, SLM just needs to pick it", color=MUTED, fontsize=10.2)

    p1 = stage(frame, -2, 16)
    p2 = stage(frame, 12, 36)

    draw_topk_panel(ax, p1)
    draw_selection_panel(ax, p2)

    footer_alpha = clamp(stage(frame, 32, 47) * 1.2)
    rounded_box(ax, 0.16, 0.04, 0.68, 0.065, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.072,
        "S2T-LOCAL: autonomous re-ranking, no inference-time LLM call needed",
        color=TEXT,
        fontsize=9.0,
        ha="center",
        va="center",
        alpha=footer_alpha,
    )

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("23-select-to-think.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=False,
    )
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()