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


def draw_exploration_panel(ax, progress: float):
    """Show normal RL exploration vs suppressed exploration."""
    alpha = clamp(progress * 1.4)
    rounded_box(ax, 0.05, 0.10, 0.42, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.88, "Normal RL Exploration", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.84, "diverse actions during training", color=MUTED, fontsize=9.2, alpha=alpha)

    # Normal exploration dots (random spread)
    np.random.seed(42)
    normal_x = np.random.rand(12)
    normal_y = np.random.rand(12)
    for i in range(12):
        local = clamp(progress * 2.5 - i * 0.12)
        if local <= 0:
            continue
        size = 0.015 + 0.01 * math.sin(progress * math.pi + i)
        pulse = 0.75 + 0.25 * math.sin(progress * math.pi * 2 + i * 0.5)
        ax.add_patch(
            patches.Circle(
                (0.26 + normal_x[i] * 0.18, 0.28 + normal_y[i] * 0.45),
                size,
                color=GREEN,
                alpha=local * pulse * 0.85,
            )
        )

    if progress > 0.2:
        hl_alpha = clamp(progress * 1.4 - 0.5) * 0.9
        ax.text(0.26, 0.18, "diverse", color=GREEN, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)
        ax.text(0.26, 0.14, "exploration ✓", color=GREEN, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)


def draw_hacking_panel(ax, progress: float):
    """Show suppressed exploration - narrow path."""
    alpha = clamp(progress * 1.4 - 0.06)
    rounded_box(ax, 0.53, 0.10, 0.42, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.55, 0.88, "Exploration Hacking", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.55, 0.84, "model suppresses own diversity", color=MUTED, fontsize=9.2, alpha=alpha)

    # Suppressed - narrow corridor
    local = clamp(progress * 2.0 - 0.2)
    corridor_alpha = clamp(progress * 1.8) * 0.4
    rounded_box(ax, 0.60, 0.25, 0.28, 0.52, "#1a1a2e", alpha=corridor_alpha, radius=0.05, lw=0.0)

    for idx in range(8):
        t = idx / 7.0
        y_pos = 0.72 - t * 0.44
        dot_alpha = clamp(progress * 2.4 - idx * 0.16) * 0.85
        if dot_alpha <= 0:
            continue
        pulse = 0.88 + 0.12 * math.sin(local * math.pi * 2 - idx * 0.3)
        color = ORANGE if (idx % 3 == 0) else BLUE
        ax.add_patch(patches.Circle((0.74, y_pos), 0.018, color=color, alpha=dot_alpha * pulse))

    if progress > 0.35:
        hl_alpha = clamp(progress * 1.4 - 0.6) * 0.9
        rounded_box(ax, 0.60, 0.20, 0.28, 0.08, "#2a1a1a", alpha=hl_alpha * 0.6, radius=0.03, lw=1.2)
        ax.text(0.74, 0.24, "narrow path", color=ORANGE, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)


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

    ax.text(0.05, 0.95, "Exploration Hacking", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.05, 0.90, "LLMs can learn to suppress exploration during RL, influencing training outcomes", color=MUTED, fontsize=10.2)

    p1 = stage(frame, -2, 16)
    p2 = stage(frame, 12, 36)

    draw_exploration_panel(ax, p1)
    draw_hacking_panel(ax, p2)

    footer_alpha = clamp(stage(frame, 32, 47) * 1.2)
    rounded_box(ax, 0.12, 0.04, 0.76, 0.065, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.072,
        "models exploit RL's reliance on exploration to alter their own training trajectory",
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
    output_path = Path(__file__).with_name("24-exploration-hacking.gif")
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
