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
CYAN = "#00bcd4"


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


def draw_pipeline_box(ax, x, y, w, h, color, label, sublabel="", alpha=1.0):
    rounded_box(ax, x, y, w, h, color, alpha=alpha, radius=0.02)
    ax.text(x + w / 2, y + h * 0.65, label, color="white", fontsize=8, fontweight="bold", ha="center", alpha=alpha)
    if sublabel:
        ax.text(x + w / 2, y + h * 0.35, sublabel, color=TEXT, fontsize=6, ha="center", alpha=alpha)


def draw_arrow(ax, x0, y0, x1, y1, color, label="", label_offset=(0, 0.05)):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.8, color=color),
    )
    if label:
        ax.text(x0 + label_offset[0], y0 + label_offset[1], label, color=color, fontsize=7, ha="center")


def draw_benchmark_bar(ax, x, y, width, height, color, value, label, alpha=1.0):
    bar_bg = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01",
        facecolor="#1a2744",
        edgecolor=(1, 1, 1, 0.1),
        linewidth=0.8,
        alpha=alpha * 0.8,
    )
    ax.add_patch(bar_bg)
    fill_w = width * value
    if fill_w > 0.01:
        bar_fill = patches.FancyBboxPatch(
            (x, y), fill_w, height,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
        ax.add_patch(bar_fill)
    ax.text(x + width / 2, y + height + 0.02, label, color=TEXT, fontsize=7, ha="center", alpha=alpha)
    ax.text(x + width + 0.01, y + height / 2, f"{value * 100:.0f}%", color=color, fontsize=7, ha="left", va="center", alpha=alpha)


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

    ax.text(0.02, 0.96, "Day 29: OpenSeeker-v2", color=TEXT, fontsize=18, fontweight="bold")
    ax.text(0.02, 0.91, "Frontier Search Agents via Simple SFT + Informative Trajectories", color=MUTED, fontsize=10)

    # ===== LEFT: Industry Heavy Pipeline =====
    p1 = clamp(stage(frame, -2, 18) * 1.2)
    rounded_box(ax, 0.01, 0.04, 0.44, 0.82, PANEL, alpha=0.95 * p1, radius=0.04, lw=1.2)

    ax.text(0.23, 0.83, "Industry Pipeline", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=p1)
    ax.text(0.23, 0.77, "(Heavy CPT + SFT + RL)", color=RED, fontsize=8, ha="center", alpha=p1 * 0.8)

    # Pipeline stages
    pipe_y = 0.60
    box_w = 0.12
    box_h = 0.10
    gap = 0.02

    for i, (stage_name, sc) in enumerate([
        ("PT", "#2a3a5c"),
        ("SFT", "#2a3a5c"),
        ("RL", "#2a3a5c"),
    ]):
        sx = 0.05 + i * (box_w + gap)
        draw_pipeline_box(ax, sx, pipe_y, box_w, box_h, sc, stage_name, alpha=p1)

    # Arrows between stages
    for i in range(2):
        arr_x = 0.05 + box_w + i * (box_w + gap)
        if stage(frame, 8, 24) > 0:
            ax.annotate(
                "",
                xy=(arr_x, pipe_y + box_h / 2),
                xytext=(arr_x - gap, pipe_y + box_h / 2),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=MUTED, alpha=p1 * clamp(stage(frame, 8, 24) * 2)),
            )

    # Cost label
    cost_alpha = clamp(stage(frame, 12, 28) * 1.5)
    ax.text(0.23, 0.44, "Massive Compute", color=RED, fontsize=9, fontweight="bold", ha="center", alpha=cost_alpha)

    # CPU/GPU icon simulation
    for i in range(3):
        bx = 0.10 + i * 0.08
        by = 0.36
        rect = patches.FancyBboxPatch(
            (bx, by), 0.06, 0.05,
            boxstyle="round,pad=0.005",
            facecolor="#3a2a4a",
            edgecolor=PURPLE,
            linewidth=1,
            alpha=cost_alpha * 0.8,
        )
        ax.add_patch(rect)
        ax.text(bx + 0.03, by + 0.025, "GPU", color=PURPLE, fontsize=5, ha="center", va="center", alpha=cost_alpha)

    # Result
    res_alpha = clamp(stage(frame, 20, 36) * 1.5)
    ax.text(0.23, 0.26, "Tongyi DeepResearch", color=MUTED, fontsize=9, ha="center", alpha=res_alpha)
    ax.text(0.23, 0.20, "43.4% | 46.7% | 32.9% | 75.0%", color=MUTED, fontsize=7, ha="center", alpha=res_alpha)

    # ===== CENTER: Arrow + "VS" =====
    vs_alpha = clamp(stage(frame, 18, 30) * 2)
    ax.text(0.50, 0.50, "VS", color=ORANGE, fontsize=16, fontweight="bold", ha="center", va="center", alpha=vs_alpha)

    # ===== RIGHT: OpenSeeker-v2 =====
    p3 = clamp(stage(frame, 16, 38) * 1.2)
    rounded_box(ax, 0.55, 0.04, 0.43, 0.82, PANEL, alpha=0.95 * p3, radius=0.04, lw=1.2)

    ax.text(0.765, 0.83, "OpenSeeker-v2", color=TEXT, fontsize=12, fontweight="bold", ha="center", alpha=p3)
    ax.text(0.765, 0.77, "Simple SFT + 10.6k data", color=GREEN, fontsize=8, ha="center", alpha=p3 * 0.8)

    # Three modifications
    mod_y = 0.62
    mod_h = 0.08
    mods = [
        ("KG Scaling", BLUE, "Knowledge Graph ↑"),
        ("Tool Expansion", PURPLE, "Tool Set Size ↑"),
        ("Low-Step Filter", ORANGE, "Strict Filtering"),
    ]

    for i, (name, color, full_name) in enumerate(mods):
        m_alpha = clamp(stage(frame, 20 + i * 4, 36 + i * 4) * 2)
        mx = 0.58 + i * 0.13
        rounded_box(ax, mx, mod_y, 0.12, mod_h, color, alpha=0.7 * m_alpha, radius=0.015)
        ax.text(mx + 0.06, mod_y + mod_h * 0.5, name, color="white", fontsize=7, fontweight="bold", ha="center", va="center", alpha=m_alpha)

    # Arrow down
    arr2_alpha = clamp(stage(frame, 28, 40) * 1.5)
    if arr2_alpha > 0:
        ax.annotate(
            "",
            xy=(0.765, 0.40),
            xytext=(0.765, 0.52),
            arrowprops=dict(arrowstyle="->", lw=2, color=GREEN, alpha=arr2_alpha),
        )

    # Benchmark bars
    bench_alpha = clamp(stage(frame, 32, 46) * 1.5)

    ax.text(0.58, 0.36, "BrowseComp", color=MUTED, fontsize=7, alpha=bench_alpha)
    draw_benchmark_bar(ax, 0.58, 0.30, 0.15, 0.04, GREEN, 0.46, "", alpha=bench_alpha)

    ax.text(0.58, 0.23, "BrowseComp-ZH", color=MUTED, fontsize=7, alpha=bench_alpha)
    draw_benchmark_bar(ax, 0.58, 0.17, 0.15, 0.04, GREEN, 0.581, "", alpha=bench_alpha)

    ax.text(0.75, 0.36, "HLE", color=MUTED, fontsize=7, alpha=bench_alpha)
    draw_benchmark_bar(ax, 0.75, 0.30, 0.15, 0.04, GREEN, 0.346, "", alpha=bench_alpha)

    ax.text(0.75, 0.23, "xbench", color=MUTED, fontsize=7, alpha=bench_alpha)
    draw_benchmark_bar(ax, 0.75, 0.17, 0.15, 0.04, GREEN, 0.78, "", alpha=bench_alpha)

    # Winner badge
    win_alpha = clamp(stage(frame, 40, 48) * 2)
    win_x, win_y = 0.85, 0.10
    rounded_box(ax, win_x - 0.05, win_y, 0.10, 0.06, GREEN, alpha=0.9 * win_alpha, radius=0.02)
    ax.text(win_x, win_y + 0.03, "SOTA", color="white", fontsize=8, fontweight="bold", ha="center", va="center", alpha=win_alpha)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("29-openseeker.gif")
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
