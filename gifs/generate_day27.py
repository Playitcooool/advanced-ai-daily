import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

WIDTH = 9
HEIGHT = 5
FRAMES = 48
BG = "#08111f"
PANEL = "#101a2b"
TEXT = "#e6edf7"
MUTED = "#8ba2c7"
BLUE = "#4a9eff"
GREEN = "#00d97e"
ORANGE = "#ffd93d"
PURPLE = "#9b59b6"


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
        edgecolor=(1, 1, 1, 0.1),
        linewidth=lw,
        alpha=alpha,
    )
    ax.add_patch(patch)
    return patch


def draw_token(ax, x, y, color, size=0.08):
    rect = patches.FancyBboxPatch(
        (x - size / 2, y - size / 2),
        size,
        size,
        boxstyle="round,pad=0.01",
        linewidth=0.5,
        edgecolor="white",
        facecolor=color,
    )
    ax.add_patch(rect)


def draw_vision_token_grid(ax, x_start, y_start, cols, rows, color_blue, color_redundancy):
    for i in range(rows):
        for j in range(cols):
            x = x_start + j * 0.12
            y = y_start - i * 0.08
            color = color_blue if (i + j) % 3 != 0 else color_redundancy
            draw_token(ax, x, y, color, size=0.07)


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

    ax.text(0.02, 0.96, "Day 27: LightKV", color=TEXT, fontsize=18, fontweight="bold")
    ax.text(0.02, 0.91, "Prompt-Aware Vision Token Compression for LVLMs", color=MUTED, fontsize=10)

    # ===== LEFT PANEL: Redundancy =====
    p1 = clamp(stage(frame, -2, 22) * 1.2)
    ax1 = ax
    rounded_box(ax1, 0.01, 0.04, 0.44, 0.82, PANEL, alpha=0.95 * p1, radius=0.04, lw=1.2)

    label_alpha = clamp(p1)
    ax1.text(0.23, 0.83, "Vision Token Redundancy", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=label_alpha)

    # Left: many tokens
    ax1.text(0.10, 0.74, "Vision Tokens", color=MUTED, fontsize=9, ha="left", alpha=label_alpha)
    draw_vision_token_grid(ax1, 0.03, 0.68, 4, 6, BLUE, "#2a6bc9")

    # Arrow with label
    arr_alpha = clamp(stage(frame, 10, 26) * 2)
    if arr_alpha > 0:
        ax1.annotate(
            "",
            xy=(0.30, 0.42),
            xytext=(0.50, 0.42),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=ORANGE, alpha=arr_alpha),
        )
        ax1.text(0.40, 0.48, "55% redundant", color=ORANGE, fontsize=8, ha="center", alpha=arr_alpha)

    # Right: few compressed tokens
    ax1.text(0.38, 0.74, "Compressed", color=MUTED, fontsize=9, ha="left", alpha=label_alpha)
    for i in range(2):
        for j in range(3):
            x = 0.28 + j * 0.12
            y = 0.60 - i * 0.12
            draw_token(ax1, x, y, GREEN, size=0.09)

    # ===== RIGHT PANEL: Cross-Modality Aggregation =====
    p2 = clamp(stage(frame, 14, 46) * 1.2)
    rounded_box(ax, 0.50, 0.04, 0.48, 0.82, PANEL, alpha=0.95 * p2, radius=0.04, lw=1.2)

    ax2 = ax
    ax2.text(0.74, 0.83, "LightKV: Cross-Modality Aggregation", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=p2)

    # Text prompt box
    text_alpha = clamp(stage(frame, 18, 34) * 1.5)
    rounded_box(ax2, 0.54, 0.65, 0.18, 0.13, "#1a1a2e", alpha=0.9 * text_alpha, radius=0.025, lw=1)
    ax2.text(0.63, 0.73, "Text Prompt", color=ORANGE, fontsize=7, fontweight="bold", ha="center", alpha=text_alpha)
    ax2.text(0.63, 0.68, '"Describe"', color="#888888", fontsize=6, ha="center", alpha=text_alpha)

    # Vision tokens (left cluster)
    ax2.text(0.55, 0.52, "Vision Tokens", color=MUTED, fontsize=8, ha="left", alpha=text_alpha)
    draw_vision_token_grid(ax2, 0.53, 0.46, 2, 4, BLUE, BLUE)

    # Aggregation hub
    hub_alpha = clamp(stage(frame, 22, 40) * 1.8)
    hub = plt.Circle((0.72, 0.48), 0.07, color=PURPLE, alpha=0.85 * hub_alpha)
    ax2.add_patch(hub)
    ax2.text(0.72, 0.48, "Agg", color="white", fontsize=7, fontweight="bold", ha="center", va="center", alpha=hub_alpha)

    # Output: lightweight tokens
    out_alpha = clamp(stage(frame, 30, 46) * 2)
    ax2.text(0.88, 0.52, "LightKV", color=GREEN, fontsize=8, ha="center", alpha=out_alpha)
    for i in range(2):
        for j in range(2):
            x = 0.83 + j * 0.10
            y = 0.40 - i * 0.10
            draw_token(ax2, x, y, GREEN, size=0.07)

    # Stats bar
    stat_alpha = clamp(stage(frame, 36, 46) * 1.5)
    stats = [("55%", "Tokens"), ("2x", "KV Cache ↓"), ("40%", "Compute ↓")]
    for i, (val, label) in enumerate(stats):
        x = 0.56 + i * 0.15
        rounded_box(ax, x - 0.055, 0.08, 0.12, 0.10, "#132138", alpha=0.85 * stat_alpha, radius=0.02, lw=0.8)
        ax.text(x, 0.155, val, color=ORANGE, fontsize=11, fontweight="bold", ha="center", alpha=stat_alpha)
        ax.text(x, 0.10, label, color=MUTED, fontsize=7, ha="center", alpha=stat_alpha)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("27-lightkv.gif")
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
