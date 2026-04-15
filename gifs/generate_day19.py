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
FRAMES = 42
BG = "#08111f"
PANEL = "#101a2b"
TEXT = "#e6edf7"
MUTED = "#8ba2c7"
ACCENT = "#4fb3ff"
GREEN = "#36c98d"
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


def draw_prompt(ax, progress: float):
    alpha = clamp(progress * 1.4)
    rounded_box(ax, 0.05, 0.20, 0.22, 0.58, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.73, "Input Prompt", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.69, "same weights, more passes", color=MUTED, fontsize=9.4, alpha=alpha)

    rows = [
        ("Hard math", ACCENT),
        ("Tool plan", GREEN),
        ("Multi-step", ORANGE),
        ("Reasoning", RED),
    ]
    for idx, (label, color) in enumerate(rows):
        local = clamp(progress * 2.2 - idx * 0.18)
        if local <= 0:
            continue
        y = 0.58 - idx * 0.11
        rounded_box(ax, 0.08, y, 0.14, 0.065, "#16243b", alpha=local, radius=0.025, lw=0.8)
        rounded_box(ax, 0.08, y, 0.04, 0.065, color, alpha=0.82 * local, radius=0.025, lw=0)
        ax.text(0.155, y + 0.033, label, color=TEXT, fontsize=8.8, ha="center", va="center", alpha=local)


def draw_loop(ax, progress: float):
    alpha = clamp(progress * 1.3)
    rounded_box(ax, 0.34, 0.20, 0.32, 0.58, PANEL, alpha=0.96 * alpha)
    ax.text(0.36, 0.73, "Looped Block", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.36, 0.69, "reuse one stack repeatedly", color=MUTED, fontsize=9.4, alpha=alpha)

    box_alpha = clamp(progress * 1.8)
    rounded_box(ax, 0.43, 0.38, 0.14, 0.12, "#17314a", alpha=0.94 * box_alpha, radius=0.03, lw=0.9)
    ax.text(0.50, 0.44, "Shared\nLayers", color=TEXT, fontsize=11, fontweight="bold", ha="center", va="center", alpha=box_alpha)

    orbit = [
        ((0.50, 0.58), "Pass 1", ACCENT),
        ((0.61, 0.44), "Pass 2", GREEN),
        ((0.50, 0.28), "Pass 3", ORANGE),
        ((0.39, 0.44), "Pass 4", RED),
    ]
    for idx, ((x, y), label, color) in enumerate(orbit):
        local = clamp(progress * 2.2 - idx * 0.16)
        if local <= 0:
            continue
        ax.add_patch(patches.Circle((x, y), 0.048, color=color, alpha=0.78 * local))
        ax.text(x, y, label, color=TEXT, fontsize=8.4, fontweight="bold", ha="center", va="center", alpha=local)

    for idx in range(len(orbit)):
        local = clamp(progress * 1.9 - idx * 0.16)
        if local <= 0:
            continue
        x1, y1 = orbit[idx][0]
        x2, y2 = orbit[(idx + 1) % len(orbit)][0]
        arrow(ax, x1, y1, x2, y2, MUTED, alpha=local, width=1.5)

    ax.text(0.50, 0.23, "hidden state keeps refining", color=MUTED, fontsize=8.5, ha="center", alpha=alpha)


def draw_exit(ax, progress: float):
    alpha = clamp(progress * 1.25)
    rounded_box(ax, 0.73, 0.20, 0.21, 0.58, PANEL, alpha=0.96 * alpha)
    ax.text(0.75, 0.73, "Adaptive Exit", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.75, 0.69, "easy tokens stop early", color=MUTED, fontsize=9.2, alpha=alpha)

    bars = [
        ("easy", 0.35, GREEN),
        ("medium", 0.60, ORANGE),
        ("hard", 0.92, RED),
    ]
    for idx, (label, width, color) in enumerate(bars):
        local = clamp(progress * 2.0 - idx * 0.2)
        if local <= 0:
            continue
        y = 0.57 - idx * 0.12
        rounded_box(ax, 0.76, y, 0.12, 0.05, "#132035", alpha=alpha, radius=0.02, lw=0.8)
        rounded_box(ax, 0.76, y, 0.12 * width, 0.05, color, alpha=0.84 * local, radius=0.02, lw=0)
        ax.text(0.895, y + 0.025, label, color=MUTED, fontsize=8.2, va="center", alpha=alpha)

    local = clamp(progress * 1.5)
    rounded_box(ax, 0.77, 0.27, 0.11, 0.075, "#17314a", alpha=0.94 * local, radius=0.03, lw=0.8)
    ax.text(0.825, 0.307, "Final\nToken", color=TEXT, fontsize=9.5, fontweight="bold", ha="center", va="center", alpha=local)


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

    ax.text(0.05, 0.91, "Looped Language Models", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.05, 0.86, "Reason longer by reusing depth in latent space", color=MUTED, fontsize=11)

    p1 = stage(frame, 0, 9)
    p2 = stage(frame, 6, 22)
    p3 = stage(frame, 16, 32)
    p4 = stage(frame, 30, 41)

    draw_prompt(ax, p1)
    draw_loop(ax, p2)
    draw_exit(ax, p3)

    if p2 > 0:
        arrow(ax, 0.27, 0.49, 0.34, 0.49, ACCENT, alpha=p2, width=2.0)
    if p3 > 0:
        arrow(ax, 0.66, 0.49, 0.73, 0.49, GREEN, alpha=p3, width=2.0)

    alpha = clamp(p4 * 1.3)
    rounded_box(ax, 0.33, 0.82, 0.56, 0.07, "#132035", alpha=0.95 * alpha, radius=0.03, lw=0.8)
    ax.text(
        0.61,
        0.855,
        "same parameters, deeper compute, stronger reasoning on hard tokens",
        color=TEXT,
        fontsize=9.0,
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
    output_path = Path(__file__).with_name("19-looped-language-models.gif")
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
