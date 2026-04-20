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


def draw_serial_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.05, 0.18, 0.27, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.73, "Classic SD", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.69, "draft waits, then verify waits", color=MUTED, fontsize=9.2, alpha=alpha)

    steps = [
        ("Draft", BLUE, 0.56),
        ("Verify", ORANGE, 0.45),
        ("Draft", BLUE, 0.34),
    ]
    for idx, (label, color, y) in enumerate(steps):
        local = clamp(progress * 2.6 - idx * 0.18)
        if local <= 0:
            continue
        rounded_box(ax, 0.10, y, 0.14, 0.075, "#16243b", alpha=0.96 * local, radius=0.025, lw=0.8)
        rounded_box(ax, 0.10, y, 0.055, 0.075, color, alpha=0.84 * local, radius=0.025, lw=0.0)
        ax.text(0.185, y + 0.038, label, color=TEXT, fontsize=10, fontweight="bold", ha="center", va="center", alpha=local)
        if idx < len(steps) - 1:
            arrow(ax, 0.17, y, 0.17, steps[idx + 1][2] + 0.075, MUTED, alpha=local, width=1.4)

    rounded_box(ax, 0.09, 0.24, 0.15, 0.07, "#132035", alpha=alpha, radius=0.02, lw=0.8)
    width = 0.15 * clamp(progress)
    rounded_box(ax, 0.09, 0.24, width, 0.07, RED, alpha=0.84 * alpha, radius=0.02, lw=0.0)
    ax.text(0.165, 0.275, "critical path", color=TEXT, fontsize=8.6, fontweight="bold", ha="center", va="center", alpha=alpha)


def draw_dflash_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.37, 0.18, 0.27, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.39, 0.73, "DFlash", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.39, 0.69, "generate a draft block at once", color=MUTED, fontsize=9.2, alpha=alpha)

    local = clamp(progress * 1.8)
    rounded_box(ax, 0.42, 0.50, 0.17, 0.10, "#17314a", alpha=0.92 * local, radius=0.03, lw=0.8)
    ax.text(0.505, 0.55, "Block Draft", color=TEXT, fontsize=11, fontweight="bold", ha="center", va="center", alpha=local)

    pulse = 0.90 + 0.10 * math.sin(progress * math.pi)
    for idx, x in enumerate([0.415, 0.463, 0.511, 0.559]):
        token_alpha = clamp(progress * 2.2 - idx * 0.10)
        if token_alpha <= 0:
            continue
        rounded_box(ax, x, 0.36, 0.034, 0.055, BLUE, alpha=0.82 * token_alpha * pulse, radius=0.015, lw=0.0)

    if local > 0:
        arrow(ax, 0.505, 0.49, 0.505, 0.41, BLUE, alpha=local, width=1.8)

    rounded_box(ax, 0.44, 0.24, 0.13, 0.07, "#132035", alpha=alpha, radius=0.02, lw=0.8)
    width = 0.13 * clamp(progress * 0.55)
    rounded_box(ax, 0.44, 0.24, width, 0.07, GREEN, alpha=0.84 * alpha, radius=0.02, lw=0.0)
    ax.text(0.505, 0.275, "shorter draft stage", color=TEXT, fontsize=8.6, fontweight="bold", ha="center", va="center", alpha=alpha)


def draw_ssd_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.69, 0.18, 0.26, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.71, 0.73, "SSD / Saguaro", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.71, 0.69, "speculate while verify runs", color=MUTED, fontsize=9.2, alpha=alpha)

    verify_alpha = clamp(progress * 1.8)
    rounded_box(ax, 0.73, 0.50, 0.18, 0.085, "#17314a", alpha=0.92 * verify_alpha, radius=0.03, lw=0.8)
    ax.text(0.82, 0.542, "Verify t", color=TEXT, fontsize=11, fontweight="bold", ha="center", va="center", alpha=verify_alpha)

    cache_alpha = clamp(progress * 2.0 - 0.12)
    rounded_box(ax, 0.73, 0.34, 0.18, 0.085, "#1c2b44", alpha=0.92 * cache_alpha, radius=0.03, lw=0.8)
    ax.text(0.82, 0.382, "Spec Cache t+1", color=TEXT, fontsize=10.5, fontweight="bold", ha="center", va="center", alpha=cache_alpha)

    if verify_alpha > 0:
        arrow(ax, 0.82, 0.49, 0.82, 0.43, PURPLE, alpha=verify_alpha, width=1.6)

    pulse = 0.88 + 0.12 * math.sin(progress * math.pi * 1.5)
    for idx, (x, color) in enumerate([(0.75, GREEN), (0.81, GREEN), (0.87, RED)]):
        local = clamp(progress * 2.4 - idx * 0.18)
        if local <= 0:
            continue
        ax.add_patch(patches.Circle((x, 0.28), 0.025, color=color, alpha=0.78 * local * pulse))
    ax.text(0.82, 0.22, "green = hit, red = miss", color=MUTED, fontsize=8.2, ha="center", alpha=alpha)


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

    ax.text(0.05, 0.91, "Parallel Drafting", color=TEXT, fontsize=21, fontweight="bold")
    ax.text(0.05, 0.86, "speculative decoding gets faster when draft latency stops being fully serial", color=MUTED, fontsize=10.5)

    p1 = stage(frame, -3, 14)
    p2 = stage(frame, 10, 29)
    p3 = stage(frame, 24, 44)

    draw_serial_panel(ax, p1)
    draw_dflash_panel(ax, p2)
    draw_ssd_panel(ax, p3)

    footer_alpha = clamp(stage(frame, 35, 47) * 1.2)
    rounded_box(ax, 0.23, 0.05, 0.54, 0.07, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.085,
        "optimize the critical path: shrink draft time or hide it behind verification",
        color=TEXT,
        fontsize=8.8,
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
    output_path = Path(__file__).with_name("22-parallel-drafting.gif")
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
