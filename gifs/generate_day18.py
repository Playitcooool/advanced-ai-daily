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
BG = "#071018"
PANEL = "#101b2a"
TEXT = "#e7eef8"
MUTED = "#8ea4c6"
ACCENT = "#62b0ff"
GREEN = "#3dd39f"
ORANGE = "#ffb35c"
RED = "#ff6f6f"


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


def draw_document(ax, progress: float):
    alpha = clamp(progress * 1.3)
    rounded_box(ax, 0.05, 0.18, 0.24, 0.60, PANEL, alpha=0.95 * alpha)
    ax.text(0.07, 0.73, "Document Page", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.69, "mixed layout and text", color=MUTED, fontsize=9.5, alpha=alpha)

    blocks = [
        (0.08, 0.60, 0.18, 0.08, ACCENT, "Title"),
        (0.08, 0.49, 0.07, 0.08, GREEN, "Fig"),
        (0.17, 0.49, 0.09, 0.08, ORANGE, "Text"),
        (0.08, 0.37, 0.18, 0.08, RED, "Table"),
        (0.08, 0.26, 0.08, 0.07, GREEN, "Seal"),
        (0.18, 0.26, 0.08, 0.07, ACCENT, "Note"),
    ]
    for idx, (x, y, w, h, color, label) in enumerate(blocks):
        local = clamp(progress * 2.3 - idx * 0.18)
        if local <= 0:
            continue
        rounded_box(ax, x, y, w, h, color, alpha=0.78 * local, radius=0.02, lw=0.8)
        ax.text(x + w / 2, y + h / 2, label, color=TEXT, fontsize=8.5, ha="center", va="center", alpha=local)


def draw_latent_layout(ax, progress: float):
    alpha = clamp(progress * 1.25)
    rounded_box(ax, 0.37, 0.18, 0.24, 0.60, PANEL, alpha=0.95 * alpha)
    ax.text(0.39, 0.73, "Layout-as-Thought", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.39, 0.69, "explicit latent plan", color=MUTED, fontsize=9.5, alpha=alpha)

    centers = [
        (0.44, 0.60, "Title", ACCENT),
        (0.53, 0.52, "Caption", ORANGE),
        (0.45, 0.41, "Table", RED),
        (0.55, 0.33, "Stamp", GREEN),
    ]
    for idx, (x, y, label, color) in enumerate(centers):
        local = clamp(progress * 2.1 - idx * 0.2)
        if local <= 0:
            continue
        ax.add_patch(patches.Circle((x, y), 0.04, color=color, alpha=0.72 * local))
        ax.text(x, y, label, color=TEXT, fontsize=8.2, fontweight="bold", ha="center", va="center", alpha=local)

    links = [(0, 1), (1, 2), (2, 3)]
    for idx, (src, dst) in enumerate(links):
        local = clamp(progress * 1.8 - idx * 0.2)
        if local <= 0:
            continue
        x1, y1 = centers[src][0], centers[src][1]
        x2, y2 = centers[dst][0], centers[dst][1]
        arrow(ax, x1 + 0.03, y1 - 0.01, x2 - 0.03, y2 + 0.01, MUTED, alpha=local, width=1.5)

    ax.text(0.49, 0.24, "reading order + regions", color=MUTED, fontsize=8.5, ha="center", alpha=alpha)


def draw_decoder(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.69, 0.18, 0.25, 0.60, PANEL, alpha=0.95 * alpha)
    ax.text(0.71, 0.73, "Unified Decoder", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.71, 0.69, "OCR, table, parse", color=MUTED, fontsize=9.5, alpha=alpha)

    tasks = [
        ("Text OCR", ACCENT, 0.58),
        ("Table", RED, 0.47),
        ("Chart", ORANGE, 0.36),
        ("Formula", GREEN, 0.25),
    ]
    for idx, (label, color, y) in enumerate(tasks):
        local = clamp(progress * 2.2 - idx * 0.18)
        if local <= 0:
            continue
        rounded_box(ax, 0.73, y, 0.16, 0.065, "#17263b", alpha=alpha, radius=0.025, lw=0.8)
        rounded_box(ax, 0.73, y, 0.16 * (0.55 + 0.1 * math.sin(idx + progress)), 0.065, color, alpha=0.78 * local, radius=0.025, lw=0)
        ax.text(0.81, y + 0.033, label, color=TEXT, fontsize=8.5, fontweight="bold", ha="center", va="center", alpha=local)

    ax.text(0.815, 0.205, "prompt selects task head", color=MUTED, fontsize=8.3, ha="center", alpha=alpha)


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

    ax.text(0.05, 0.91, "Qianfan-OCR", color=TEXT, fontsize=22, fontweight="bold")
    ax.text(0.05, 0.86, "Think in layout, then decode the page", color=MUTED, fontsize=11)

    p1 = stage(frame, 0, 9)
    p2 = stage(frame, 6, 20)
    p3 = stage(frame, 15, 30)
    p4 = stage(frame, 26, 40)

    draw_document(ax, p1)
    draw_latent_layout(ax, p2)
    draw_decoder(ax, p3)

    if p2 > 0:
        arrow(ax, 0.29, 0.48, 0.37, 0.48, ACCENT, alpha=p2, width=2.0)
    if p3 > 0:
        arrow(ax, 0.61, 0.48, 0.69, 0.48, GREEN, alpha=p3, width=2.0)

    alpha = clamp(p4 * 1.2)
    rounded_box(ax, 0.52, 0.82, 0.40, 0.07, "#132035", alpha=0.92 * alpha, radius=0.03, lw=0.8)
    ax.text(
        0.72,
        0.855,
        "1 model, many document tasks, lower layout confusion",
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
    output_path = Path(__file__).with_name("18-qianfan-ocr.gif")
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
