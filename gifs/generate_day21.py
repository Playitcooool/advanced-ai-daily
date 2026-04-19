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
    rounded_box(ax, 0.05, 0.18, 0.39, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.73, "Serial Tool Calls", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.69, "agent waits on each tool in sequence", color=MUTED, fontsize=9.5, alpha=alpha)

    labels = [("Search", BLUE), ("Docs", ORANGE), ("Calc", GREEN)]
    x_positions = [0.09, 0.22, 0.35]
    active_idx = min(2, int(progress * 3.2))
    for idx, ((label, color), x) in enumerate(zip(labels, x_positions)):
        local = clamp(progress * 2.6 - idx * 0.22)
        if local <= 0:
            continue
        rounded_box(ax, x, 0.46, 0.09, 0.10, "#16243b", alpha=0.96 * local, radius=0.025, lw=0.8)
        fill = color if idx <= active_idx else "#1b2c45"
        rounded_box(ax, x + 0.008, 0.468, 0.074, 0.064, fill, alpha=0.84 * local, radius=0.02, lw=0.0)
        ax.text(x + 0.045, 0.512, label, color=TEXT, fontsize=8.8, fontweight="bold", ha="center", va="center", alpha=local)
        if idx < 2:
            arrow(ax, x + 0.09, 0.51, x_positions[idx + 1], 0.51, MUTED, alpha=local, width=1.4)

    rounded_box(ax, 0.09, 0.29, 0.30, 0.08, "#132035", alpha=alpha, radius=0.02, lw=0.8)
    width = 0.30 * clamp(progress)
    rounded_box(ax, 0.09, 0.29, width, 0.08, RED, alpha=0.84 * alpha, radius=0.02, lw=0.0)
    ax.text(0.24, 0.33, "total latency = 2s + 3s + 1s", color=TEXT, fontsize=9, fontweight="bold", ha="center", va="center", alpha=alpha)


def draw_parallel_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.56, 0.18, 0.39, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.58, 0.73, "Parallel Tool Calls", color=TEXT, fontsize=15, fontweight="bold", alpha=alpha)
    ax.text(0.58, 0.69, "launch ready tools together, then merge", color=MUTED, fontsize=9.5, alpha=alpha)

    labels = [("Search", BLUE), ("Docs", ORANGE), ("Calc", GREEN)]
    xs = [0.60, 0.73, 0.86]
    pulse = 0.9 + 0.1 * math.sin(progress * math.pi)
    for idx, ((label, color), x) in enumerate(zip(labels, xs)):
        local = clamp(progress * 2.4 - idx * 0.10)
        if local <= 0:
            continue
        rounded_box(ax, x, 0.46, 0.09, 0.10, "#16243b", alpha=0.96 * local, radius=0.025, lw=0.8)
        rounded_box(ax, x + 0.008, 0.468, 0.074, 0.064, color, alpha=0.86 * local * pulse, radius=0.02, lw=0.0)
        ax.text(x + 0.045, 0.512, label, color=TEXT, fontsize=8.8, fontweight="bold", ha="center", va="center", alpha=local)

    local = clamp(progress * 1.5)
    rounded_box(ax, 0.70, 0.34, 0.11, 0.07, "#17314a", alpha=0.92 * local, radius=0.03, lw=0.8)
    ax.text(0.755, 0.375, "Merge", color=TEXT, fontsize=9.5, fontweight="bold", ha="center", va="center", alpha=local)
    for x in xs:
        arrow(ax, x + 0.045, 0.46, 0.755, 0.41, PURPLE, alpha=local, width=1.5)

    rounded_box(ax, 0.60, 0.24, 0.30, 0.08, "#132035", alpha=alpha, radius=0.02, lw=0.8)
    width = 0.30 * clamp(progress * 0.55)
    rounded_box(ax, 0.60, 0.24, width, 0.08, GREEN, alpha=0.84 * alpha, radius=0.02, lw=0.0)
    ax.text(0.75, 0.33, "critical path = max(2s, 3s, 1s)", color=TEXT, fontsize=9, fontweight="bold", ha="center", va="center", alpha=alpha)


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

    ax.text(0.05, 0.91, "Parallel Tool Calling", color=TEXT, fontsize=21, fontweight="bold")
    ax.text(0.05, 0.86, "latency drops when independent tools stop waiting on each other", color=MUTED, fontsize=11)

    p1 = stage(frame, -4, 12)
    p2 = stage(frame, 4, 22)
    p3 = stage(frame, 20, 38)

    draw_serial_panel(ax, p1)
    draw_parallel_panel(ax, p2)

    footer_alpha = clamp(p3 * 1.2)
    rounded_box(ax, 0.24, 0.05, 0.52, 0.07, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.085,
        "same tool set, different schedule: sum latency vs critical-path latency",
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
    output_path = Path(__file__).with_name("21-parallel-tool-calling.gif")
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
