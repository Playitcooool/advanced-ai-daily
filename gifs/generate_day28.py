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


def draw_token_row(ax, x_start, y, count, color, size=0.065, spacing=0.10):
    for i in range(count):
        draw_token(ax, x_start + i * spacing, y, color, size=size)


def draw_arrow(ax, x0, y0, x1, y1, color, label="", label_offset=(0, 0.05)):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=color),
    )
    if label:
        ax.text(x0 + label_offset[0], y0 + label_offset[1], label, color=color, fontsize=7, ha="center")


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

    ax.text(0.02, 0.96, "Day 28: SpecKV", color=TEXT, fontsize=18, fontweight="bold")
    ax.text(0.02, 0.91, "Adaptive Speculative Decoding via Compression-Aware Gamma Selection", color=MUTED, fontsize=10)

    # ===== LEFT: Fixed γ=4 baseline =====
    p1 = clamp(stage(frame, -2, 20) * 1.2)
    rounded_box(ax, 0.01, 0.04, 0.44, 0.82, PANEL, alpha=0.95 * p1, radius=0.04, lw=1.2)

    ax.text(0.23, 0.83, "Fixed γ = 4", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=p1)

    # Draft model
    draft_alpha = clamp(stage(frame, 4, 18) * 1.5)
    rounded_box(ax, 0.04, 0.62, 0.18, 0.16, "#1a2744", alpha=0.9 * draft_alpha, radius=0.025, lw=1)
    ax.text(0.13, 0.72, "Draft", color=BLUE, fontsize=9, fontweight="bold", ha="center", alpha=draft_alpha)

    # Draft proposes 4 tokens
    draft_alpha2 = clamp(stage(frame, 8, 22) * 1.5)
    ax.text(0.04, 0.56, "Proposes:", color=MUTED, fontsize=8, alpha=draft_alpha2)
    draw_token_row(ax, 0.04, 0.50, 4, BLUE, size=0.055, spacing=0.08)
    ax.text(0.13, 0.42, "γ = 4", color=ORANGE, fontsize=8, ha="center", alpha=draft_alpha2)

    # Arrow down
    arr1_alpha = clamp(stage(frame, 14, 26) * 1.5)
    if arr1_alpha > 0:
        ax.annotate(
            "",
            xy=(0.13, 0.30),
            xytext=(0.13, 0.38),
            arrowprops=dict(arrowstyle="->", lw=1.8, color=MUTED, alpha=arr1_alpha),
        )

    # Target model
    tgt_alpha = clamp(stage(frame, 16, 28) * 1.5)
    rounded_box(ax, 0.04, 0.12, 0.18, 0.16, "#1a3d2e", alpha=0.9 * tgt_alpha, radius=0.025, lw=1)
    ax.text(0.13, 0.22, "Target", color=GREEN, fontsize=9, fontweight="bold", ha="center", alpha=tgt_alpha)

    # Stats fixed
    stat_alpha = clamp(stage(frame, 22, 36) * 1.5)
    ax.text(0.32, 0.42, "Accept:", color=MUTED, fontsize=8, alpha=stat_alpha)
    ax.text(0.32, 0.35, "~50%", color=RED, fontsize=10, fontweight="bold", alpha=stat_alpha)
    ax.text(0.32, 0.26, "Wasted:", color=MUTED, fontsize=8, alpha=stat_alpha)
    ax.text(0.32, 0.19, "50%", color=RED, fontsize=10, fontweight="bold", alpha=stat_alpha)

    # ===== CENTER: MLP Controller =====
    p_mlp = clamp(stage(frame, 10, 28) * 1.5)
    mlp_x = 0.50
    mlp_y = 0.52

    # MLP hexagon-ish shape
    mlp_color = PURPLE
    mlp_patch = plt.Polygon(
        [[mlp_x - 0.07, mlp_y - 0.05], [mlp_x + 0.07, mlp_y - 0.05],
         [mlp_x + 0.09, mlp_y], [mlp_x + 0.07, mlp_y + 0.05],
         [mlp_x - 0.07, mlp_y + 0.05], [mlp_x - 0.09, mlp_y]],
        closed=True, facecolor=mlp_color, edgecolor="white", linewidth=1.5, alpha=0.85 * p_mlp
    )
    ax.add_patch(mlp_patch)
    ax.text(mlp_x, mlp_y, "MLP", color="white", fontsize=9, fontweight="bold", ha="center", va="center", alpha=p_mlp)

    # Input signals
    sig_alpha = clamp(stage(frame, 6, 22) * 1.5)
    ax.text(0.50, 0.72, "Confidence + Entropy", color=MUTED, fontsize=8, ha="center", alpha=sig_alpha)

    # Confidence signal
    draw_arrow(ax, 0.50, 0.67, 0.50, 0.58, BLUE, label="", label_offset=(0.05, 0))

    # γ outputs
    out_alpha = clamp(stage(frame, 20, 34) * 1.5)
    ax.text(0.50, 0.35, "Adaptive γ", color=TEXT, fontsize=8, ha="center", alpha=out_alpha)
    draw_arrow(ax, 0.50, 0.47, 0.50, 0.38, GREEN, label="", label_offset=(0.05, 0))

    # γ value indicator
    gamma_vals = [1, 2, 3, 4, 5, 6]
    for i, g in enumerate(gamma_vals):
        g_x = 0.42 + i * 0.03
        g_alpha = clamp(stage(frame, 24, 40) * 2) * clamp(1 - abs(g - 3) * 0.15)
        ax.text(g_x, 0.30, str(g), color=ORANGE, fontsize=6, ha="center", alpha=g_alpha)

    # ===== RIGHT: Adaptive SpecKV =====
    p3 = clamp(stage(frame, 18, 40) * 1.2)
    rounded_box(ax, 0.55, 0.04, 0.43, 0.82, PANEL, alpha=0.95 * p3, radius=0.04, lw=1.2)

    ax.text(0.765, 0.83, "SpecKV: Adaptive γ", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=p3)

    # Draft model
    d_alpha = clamp(stage(frame, 22, 36) * 1.5)
    rounded_box(ax, 0.58, 0.62, 0.18, 0.16, "#1a2744", alpha=0.9 * d_alpha, radius=0.025, lw=1)
    ax.text(0.67, 0.72, "Draft", color=BLUE, fontsize=9, fontweight="bold", ha="center", alpha=d_alpha)

    # Proposes variable tokens
    prop_alpha = clamp(stage(frame, 26, 40) * 1.5)
    ax.text(0.58, 0.56, "Proposes:", color=MUTED, fontsize=8, alpha=prop_alpha)

    # Show 6 tokens in a row, some faded based on adaptive γ
    gamma_to_show = 3
    for i in range(6):
        c = BLUE if i < gamma_to_show else type('x', (object,), {'facecolor': lambda s: (0.2, 0.2, 0.3, 0.3)})()
        if i < gamma_to_show:
            draw_token(ax, 0.58 + i * 0.07, 0.50, BLUE, size=0.055)
        else:
            draw_token(ax, 0.58 + i * 0.07, 0.50, "#2a2a4a", size=0.055)

    ax.text(0.67, 0.42, "γ = 3", color=ORANGE, fontsize=8, ha="center", alpha=prop_alpha)

    # Arrow down
    arr2_alpha = clamp(stage(frame, 32, 44) * 1.5)
    if arr2_alpha > 0:
        ax.annotate(
            "",
            xy=(0.67, 0.30),
            xytext=(0.67, 0.38),
            arrowprops=dict(arrowstyle="->", lw=1.8, color=GREEN, alpha=arr2_alpha),
        )

    # Target model
    t_alpha = clamp(stage(frame, 34, 46) * 1.5)
    rounded_box(ax, 0.58, 0.12, 0.18, 0.16, "#1a3d2e", alpha=0.9 * t_alpha, radius=0.025, lw=1)
    ax.text(0.67, 0.22, "Target", color=GREEN, fontsize=9, fontweight="bold", ha="center", alpha=t_alpha)

    # Stats SpecKV
    s_alpha = clamp(stage(frame, 38, 48) * 2)
    ax.text(0.82, 0.42, "Accept:", color=MUTED, fontsize=8, alpha=s_alpha)
    ax.text(0.82, 0.35, "~78%", color=GREEN, fontsize=10, fontweight="bold", alpha=s_alpha)
    ax.text(0.82, 0.26, "Gain:", color=MUTED, fontsize=8, alpha=s_alpha)
    ax.text(0.82, 0.19, "+56%", color=GREEN, fontsize=10, fontweight="bold", alpha=s_alpha)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("28-speckv.gif")
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