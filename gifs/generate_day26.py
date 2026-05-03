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


def draw_pipeline_stages(ax, progress: float):
    """Show the 3-stage PRISM pipeline: SFT -> Alignment -> RLVR."""
    stages = [
        ("SFT", "Supervised\nFine-Tuning", BLUE, 0.10),
        ("PRISM", "Distribution\nAlignment", PURPLE, 0.40),
        ("RLVR", "RL with\nVerifiable Rewards", GREEN, 0.70),
    ]

    for i, (short, full, color, x_base) in enumerate(stages):
        local = clamp(progress * 1.8 - i * 0.2)
        box_w = 0.22
        rounded_box(ax, x_base, 0.38, box_w, 0.30, PANEL, alpha=0.95 * local, radius=0.04, lw=1.5)
        ax.text(x_base + box_w / 2, 0.60, short, color=color, fontsize=13, fontweight="bold", ha="center", alpha=local)
        ax.text(x_base + box_w / 2, 0.49, full, color=TEXT, fontsize=8.5, ha="center", va="top", alpha=local)

        if i < 2:
            arrow_alpha = clamp(progress * 2 - (i + 1) * 0.4) * 0.7
            if arrow_alpha > 0:
                ax.annotate(
                    "",
                    xy=(x_base + box_w + 0.025, 0.53),
                    xytext=(x_base + box_w - 0.005, 0.53),
                    arrowprops=dict(arrowstyle="->", lw=2.0, color=TEAL, alpha=arrow_alpha, shrinkA=4, shrinkB=4),
                )

    if progress > 0.45:
        hl_alpha = clamp(progress * 1.5 - 0.7) * 0.9
        ax.text(0.50, 0.22, "mitigates distributional drift between SFT and RLVR", color=ORANGE, fontsize=9.5, ha="center", alpha=hl_alpha)


def draw_moe_discriminator(ax, progress: float):
    """Show MoE discriminator with perception and reasoning experts."""
    alpha = clamp(progress * 1.5 - 0.08)
    rounded_box(ax, 0.53, 0.10, 0.42, 0.82, PANEL, alpha=0.96 * alpha)
    ax.text(0.55, 0.88, "MoE Discriminator", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.55, 0.84, "perception + reasoning experts", color=MUTED, fontsize=9.2, alpha=alpha)

    # Router
    local = clamp(progress * 2.2)
    rounded_box(ax, 0.60, 0.55, 0.28, 0.10, "#1a2a4a", alpha=0.85 * local, radius=0.025, lw=1.0)
    ax.text(0.74, 0.60, "Router", color=TEAL, fontsize=10, fontweight="bold", ha="center", va="center", alpha=local)

    # Two expert boxes
    exp_alpha = clamp(progress * 2.5 - 0.3)
    for j, (name, color, x) in enumerate([("Perception\nExpert", BLUE, 0.60), ("Reasoning\nExpert", PURPLE, 0.77)]):
        rounded_box(ax, x, 0.28, 0.15, 0.22, color, alpha=exp_alpha * 0.6, radius=0.025, lw=0.8)
        ax.text(x + 0.075, 0.39, name, color=TEXT, fontsize=8, ha="center", va="center", alpha=exp_alpha)

    # Connection lines from router to experts
    line_alpha = clamp(progress * 3 - 0.5) * 0.5
    if line_alpha > 0:
        for x in [0.675, 0.845]:
            ax.plot([0.74, x], [0.55, 0.50], color=MUTED, alpha=line_alpha, lw=1.0)

    if progress > 0.55:
        hl_alpha = clamp(progress * 1.5 - 0.8) * 0.85
        ax.text(0.74, 0.16, "disentangled corrective signals", color=GREEN, fontsize=10, fontweight="bold", ha="center", alpha=hl_alpha)


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

    ax.text(0.05, 0.95, "PRISM", color=TEXT, fontsize=20, fontweight="bold")
    ax.text(0.05, 0.90, "Pre-alignment via Black-box On-policy Distillation for Multimodal RL", color=MUTED, fontsize=10.2)

    p1 = stage(frame, -2, 20)
    p2 = stage(frame, 16, 40)

    draw_pipeline_stages(ax, p1)
    draw_moe_discriminator(ax, p2)

    footer_alpha = clamp(stage(frame, 34, 47) * 1.2)
    rounded_box(ax, 0.12, 0.04, 0.76, 0.065, "#132138", alpha=0.92 * footer_alpha, radius=0.03, lw=0.8)
    ax.text(
        0.50,
        0.072,
        "on-policy distillation bridges SFT and RLVR — +4.4 to +6.0 accuracy on Qwen3-VL",
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
    output_path = Path(__file__).with_name("26-prism.gif")
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