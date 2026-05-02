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
PURPLE = "#b388ff"
TEAL = "#26c6da"


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


def draw_synthetic_computer(ax, progress: float):
    """Show synthetic computer being built with folder hierarchy."""
    alpha = clamp(progress * 1.5)
    rounded_box(ax, 0.05, 0.10, 0.42, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.07, 0.88, "Synthetic Computer", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.84, "folder hierarchy + content-rich artifacts", color=MUTED, fontsize=9.2, alpha=alpha)

    # Folder tree structure
    folders = [
        ("Documents", 0.70, 0.75),
        ("Spreadsheets", 0.70, 0.62),
        ("Presentations", 0.70, 0.49),
        ("Projects", 0.50, 0.36),
    ]

    for i, (name, y, x) in enumerate(folders):
        local = clamp(progress * 2.5 - i * 0.25)
        if local <= 0:
            continue
        box_color = [BLUE, TEAL, PURPLE, ORANGE][i]
        rounded_box(ax, x, y - 0.06, 0.18, 0.07, box_color, alpha=local * 0.85, radius=0.02, lw=0.8)
        ax.text(x + 0.09, y - 0.025, name, color=TEXT, fontsize=8.5, ha="center", va="center", alpha=local)

        if i > 0:
            parent_y = folders[i - 1][2] - 0.06
            line_alpha = clamp(progress * 3 - i * 0.3) * 0.5
            if line_alpha > 0:
                ax.plot([x + 0.09, x + 0.09], [y + 0.01, parent_y - 0.01], color=MUTED, alpha=line_alpha, lw=0.8)

    if progress > 0.3:
        hl_alpha = clamp(progress * 1.5 - 0.5) * 0.9
        ax.text(0.26, 0.16, "realistic user context", color=GREEN, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)


def draw_agent_simulation(ax, progress: float):
    """Show agent running long-horizon simulation."""
    alpha = clamp(progress * 1.5 - 0.06)
    rounded_box(ax, 0.53, 0.10, 0.42, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.55, 0.88, "Agent Simulation", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.55, 0.84, "long-horizon productivity tasks", color=MUTED, fontsize=9.2, alpha=alpha)

    # Agent icon (simplified)
    local = clamp(progress * 2.0)
    agent_x, agent_y = 0.74, 0.70
    ax.add_patch(patches.Circle((agent_x, agent_y), 0.04, color=BLUE, alpha=local * 0.9))
    ax.text(agent_x, agent_y, "A", color=TEXT, fontsize=12, ha="center", va="center", fontweight="bold", alpha=local)

    # Steps/turns counter
    steps = int(local * 50)
    ax.text(0.74, 0.55, f"{steps} turns", color=TEAL, fontsize=10, ha="center", alpha=clamp(local * 1.5))

    # Progress bar for 8+ hours simulation
    bar_width = 0.30
    bar_x = 0.59
    bar_y = 0.40
    rounded_box(ax, bar_x, bar_y, bar_width, 0.06, "#1a2a3a", alpha=0.8, radius=0.02, lw=0)
    filled = local * bar_width
    if filled > 0:
        rounded_box(ax, bar_x, bar_y, filled, 0.06, GREEN, alpha=0.85, radius=0.02, lw=0)

    if progress > 0.4:
        hl_alpha = clamp(progress * 1.5 - 0.7) * 0.9
        ax.text(0.74, 0.30, "2000+ turns", color=ORANGE, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)
        ax.text(0.74, 0.24, "8+ hour runtime", color=MUTED, fontsize=9, ha="center", alpha=hl_alpha)


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

    ax.text(0.05, 0.95, "Synthetic Computers at Scale", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.05, 0.90, "Scalable synthetic user environments for long-horizon agent productivity training", color=MUTED, fontsize=10.2)

    p1 = stage(frame, -2, 18)
    p2 = stage(frame, 14, 38)

    draw_synthetic_computer(ax, p1)
    draw_agent_simulation(ax, p2)

    footer_alpha = clamp(stage(frame, 32, 47) * 1.2)
    rounded_box(ax, 0.12, 0.04, 0.76, 0.065, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.072,
        "synthetic data at scale enables agent self-improvement in long-horizon productivity scenarios",
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
    output_path = Path(__file__).with_name("25-synthetic-computers.gif")
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
