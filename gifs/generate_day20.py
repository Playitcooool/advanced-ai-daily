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
FRAMES = 44
BG = "#07111b"
PANEL = "#101c2b"
TEXT = "#e7eef8"
MUTED = "#8ca4c5"
BLUE = "#60b0ff"
GREEN = "#37d39d"
ORANGE = "#ffb454"
RED = "#ff6b6b"


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


def draw_task_panel(ax, progress: float):
    alpha = clamp(progress * 1.4)
    rounded_box(ax, 0.05, 0.18, 0.23, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.73, "Problem Mix", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.69, "difficulty is uneven", color=MUTED, fontsize=9.4, alpha=alpha)

    tasks = [
        ("Easy", GREEN, 0.58),
        ("Medium", ORANGE, 0.46),
        ("Hard", RED, 0.34),
    ]
    for idx, (label, color, y) in enumerate(tasks):
        local = clamp(progress * 2.2 - idx * 0.18)
        if local <= 0:
            continue
        rounded_box(ax, 0.08, y, 0.14, 0.075, "#16263b", alpha=0.96 * local, radius=0.025, lw=0.8)
        rounded_box(ax, 0.08, y, 0.045, 0.075, color, alpha=0.82 * local, radius=0.025, lw=0)
        ax.text(0.165, y + 0.038, label, color=TEXT, fontsize=10, fontweight="bold", ha="center", va="center", alpha=local)

    ax.text(0.165, 0.25, "fixed tokens waste compute", color=MUTED, fontsize=8.5, ha="center", alpha=alpha)


def draw_policy_panel(ax, progress: float):
    alpha = clamp(progress * 1.3)
    rounded_box(ax, 0.36, 0.18, 0.28, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.38, 0.73, "Budget Policy", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.38, 0.69, "estimate -> allocate -> stop", color=MUTED, fontsize=9.2, alpha=alpha)

    local = clamp(progress * 1.8)
    rounded_box(ax, 0.42, 0.53, 0.15, 0.09, "#17324a", alpha=0.94 * local, radius=0.03, lw=0.9)
    ax.text(0.495, 0.575, "Difficulty\nEstimator", color=TEXT, fontsize=10.2, fontweight="bold", ha="center", va="center", alpha=local)

    rounded_box(ax, 0.42, 0.34, 0.15, 0.09, "#17324a", alpha=0.94 * local, radius=0.03, lw=0.9)
    ax.text(0.495, 0.385, "Reasoning\nBudget", color=TEXT, fontsize=10.2, fontweight="bold", ha="center", va="center", alpha=local)

    if local > 0:
        arrow(ax, 0.495, 0.52, 0.495, 0.43, BLUE, alpha=local, width=1.8)

    markers = [
        (0.57, 0.58, GREEN, "2k"),
        (0.60, 0.48, ORANGE, "6k"),
        (0.57, 0.30, RED, "12k"),
    ]
    for idx, (x, y, color, label) in enumerate(markers):
        m = clamp(progress * 2.1 - idx * 0.16)
        if m <= 0:
            continue
        ax.add_patch(patches.Circle((x, y), 0.034, color=color, alpha=0.8 * m))
        ax.text(x, y, label, color=TEXT, fontsize=8.4, fontweight="bold", ha="center", va="center", alpha=m)


def draw_budget_panel(ax, progress: float):
    alpha = clamp(progress * 1.25)
    rounded_box(ax, 0.71, 0.18, 0.23, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.73, 0.73, "Token Budget", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.73, 0.69, "hard cases think longer", color=MUTED, fontsize=9.2, alpha=alpha)

    bars = [
        ("Easy", 0.26, GREEN, 0.58),
        ("Medium", 0.55, ORANGE, 0.46),
        ("Hard", 0.88, RED, 0.34),
    ]
    pulse = 0.92 + 0.08 * math.sin(progress * math.pi)
    for idx, (label, width, color, y) in enumerate(bars):
        local = clamp(progress * 2.0 - idx * 0.18)
        if local <= 0:
            continue
        rounded_box(ax, 0.75, y, 0.13, 0.065, "#14233a", alpha=alpha, radius=0.02, lw=0.8)
        rounded_box(ax, 0.75, y, 0.13 * width * pulse, 0.065, color, alpha=0.84 * local, radius=0.02, lw=0)
        ax.text(0.90, y + 0.033, label, color=MUTED, fontsize=8.2, va="center", alpha=alpha)

    local = clamp(progress * 1.5)
    rounded_box(ax, 0.76, 0.24, 0.13, 0.075, "#17324a", alpha=0.94 * local, radius=0.03, lw=0.8)
    ax.text(0.825, 0.277, "Stop when\nbudget ends", color=TEXT, fontsize=9.0, fontweight="bold", ha="center", va="center", alpha=local)


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

    ax.text(0.05, 0.91, "Adaptive Reasoning Budgets", color=TEXT, fontsize=21, fontweight="bold")
    ax.text(0.05, 0.86, "avoid overthinking by matching compute to difficulty", color=MUTED, fontsize=11)

    p1 = stage(frame, 0, 9)
    p2 = stage(frame, 7, 22)
    p3 = stage(frame, 16, 32)
    p4 = stage(frame, 31, 43)

    draw_task_panel(ax, p1)
    draw_policy_panel(ax, p2)
    draw_budget_panel(ax, p3)

    if p2 > 0:
        arrow(ax, 0.28, 0.49, 0.36, 0.49, BLUE, alpha=p2, width=2.0)
    if p3 > 0:
        arrow(ax, 0.64, 0.49, 0.71, 0.49, GREEN, alpha=p3, width=2.0)

    alpha = clamp(p4 * 1.2)
    rounded_box(ax, 0.34, 0.82, 0.56, 0.07, "#132138", alpha=0.92 * alpha, radius=0.03, lw=0.8)
    ax.text(
        0.62,
        0.855,
        "same model, variable thinking depth, better latency-quality tradeoff",
        color=TEXT,
        fontsize=8.8,
        ha="center",
        va="center",
        alpha=alpha,
    )

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("20-adaptive-reasoning-budgets.gif")
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
