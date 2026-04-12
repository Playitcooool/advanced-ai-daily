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
FRAMES = 40
BG = "#08111f"
PANEL = "#101a2b"
TEXT = "#e6edf7"
MUTED = "#8ba2c7"
ACCENT = "#4fb3ff"
GREEN = "#36c98d"
ORANGE = "#ffb454"
RED = "#ff6b6b"
PURPLE = "#b388ff"


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def stage_progress(frame: int, start: int, end: int) -> float:
    if frame <= start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / float(end - start)


def rounded_box(ax, x, y, w, h, color, alpha=1.0, radius=0.03, lw=1.4):
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


def pill(ax, x, y, w, h, label, fill, alpha=1.0):
    rounded_box(ax, x, y, w, h, fill, alpha=alpha, radius=0.08, lw=1.0)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        color=TEXT,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        alpha=alpha,
    )


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
            shrinkA=5,
            shrinkB=6,
        ),
    )


def draw_task_panel(ax, progress: float):
    alpha = clamp(progress * 1.25)
    rounded_box(ax, 0.05, 0.18, 0.23, 0.60, PANEL, alpha=0.95 * alpha)
    ax.text(0.07, 0.74, "Everyday Tasks", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.07, 0.70, "153 tasks on 144 live sites", color=MUTED, fontsize=9.5, alpha=alpha)

    items = [
        ("Travel", ACCENT),
        ("Shopping", GREEN),
        ("Jobs", ORANGE),
        ("Finance", PURPLE),
        ("Daily", RED),
    ]
    for idx, (label, color) in enumerate(items):
        local = clamp(progress * 2.4 - idx * 0.22)
        if local <= 0:
            continue
        y = 0.60 - idx * 0.09
        pill(ax, 0.08, y, 0.15, 0.05, label, color, alpha=local)
        ax.text(
            0.25,
            y + 0.025,
            "multi-step form flow",
            color=MUTED,
            fontsize=8.5,
            va="center",
            alpha=local,
        )


def draw_live_web_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.33, 0.18, 0.22, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.35, 0.74, "Live Web", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.35, 0.70, "real browser, not sandbox", color=MUTED, fontsize=9.5, alpha=alpha)

    site_rows = [
        ("book flight", ACCENT),
        ("apply job", ORANGE),
        ("buy groceries", GREEN),
        ("schedule visit", PURPLE),
    ]
    for idx, (label, color) in enumerate(site_rows):
        local = clamp(progress * 2.0 - idx * 0.20)
        if local <= 0:
            continue
        y = 0.60 - idx * 0.11
        rounded_box(ax, 0.36, y, 0.15, 0.075, "#16243b", alpha=local, radius=0.02, lw=1.0)
        ax.add_patch(patches.Circle((0.385, y + 0.037), 0.012, color=color, alpha=local))
        ax.text(0.405, y + 0.037, label, color=TEXT, fontsize=9, va="center", alpha=local)

    progress_bar = clamp(progress * 1.05)
    rounded_box(ax, 0.36, 0.24, 0.15, 0.05, "#132035", alpha=alpha, radius=0.02, lw=1.0)
    rounded_box(ax, 0.36, 0.24, 0.15 * progress_bar, 0.05, ACCENT, alpha=0.75 * alpha, radius=0.02, lw=0)
    ax.text(0.435, 0.265, "browser actions", color=TEXT, fontsize=8.5, ha="center", va="center", alpha=alpha)


def draw_intercept_panel(ax, progress: float):
    alpha = clamp(progress * 1.2)
    rounded_box(ax, 0.59, 0.18, 0.15, 0.60, PANEL, alpha=0.96 * alpha)
    ax.text(0.605, 0.74, "Safe Eval", color=TEXT, fontsize=14, fontweight="bold", alpha=alpha)
    ax.text(0.605, 0.70, "block final submit", color=MUTED, fontsize=9.5, alpha=alpha)

    shield = patches.RegularPolygon(
        (0.665, 0.56),
        numVertices=6,
        radius=0.06,
        orientation=math.pi / 6,
        facecolor="#17314a",
        edgecolor=(1, 1, 1, 0.08),
        alpha=alpha,
    )
    ax.add_patch(shield)
    ax.text(0.665, 0.56, "HTTP", color=TEXT, fontsize=12, fontweight="bold", ha="center", va="center", alpha=alpha)
    ax.text(0.665, 0.44, "capture request", color=MUTED, fontsize=8.5, ha="center", alpha=alpha)
    ax.text(0.665, 0.38, "stop real action", color=MUTED, fontsize=8.5, ha="center", alpha=alpha)


def draw_trace_panel(ax, progress: float):
    alpha = clamp(progress * 1.15)
    rounded_box(ax, 0.76, 0.18, 0.18, 0.33, PANEL, alpha=0.96 * alpha)
    ax.text(0.78, 0.47, "Trace Layers", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)

    traces = [
        ("replay", ACCENT),
        ("shots", ORANGE),
        ("traffic", GREEN),
        ("msgs", PURPLE),
        ("actions", RED),
    ]
    for idx, (label, color) in enumerate(traces):
        local = clamp(progress * 2.5 - idx * 0.24)
        if local <= 0:
            continue
        y = 0.41 - idx * 0.055
        rounded_box(ax, 0.79, y, 0.11, 0.035, color, alpha=0.82 * local, radius=0.03, lw=0)
        ax.text(0.845, y + 0.0175, label, color=TEXT, fontsize=8.5, fontweight="bold", ha="center", va="center", alpha=local)


def draw_eval_panel(ax, progress: float):
    alpha = clamp(progress * 1.25)
    rounded_box(ax, 0.76, 0.54, 0.18, 0.24, PANEL, alpha=0.96 * alpha)
    ax.text(0.78, 0.73, "Agentic Eval", color=TEXT, fontsize=13, fontweight="bold", alpha=alpha)
    ax.text(0.78, 0.69, "compare to human path", color=MUTED, fontsize=8.7, alpha=alpha)

    bars = [
        ("Claude 4.6", 0.333, GREEN),
        ("GLM-5", 0.242, ACCENT),
        ("GPT-5.4", 0.065, ORANGE),
    ]
    for idx, (label, value, color) in enumerate(bars):
        local = clamp(progress * 2.0 - idx * 0.22)
        if local <= 0:
            continue
        y = 0.63 - idx * 0.06
        rounded_box(ax, 0.79, y, 0.11, 0.028, "#132035", alpha=alpha, radius=0.02, lw=0.8)
        rounded_box(ax, 0.79, y, 0.11 * value * local, 0.028, color, alpha=0.88 * alpha, radius=0.02, lw=0)
        ax.text(0.79, y + 0.04, label, color=MUTED, fontsize=7.8, alpha=alpha)
        ax.text(0.905, y + 0.014, f"{value*100:.1f}%", color=TEXT, fontsize=8.2, ha="left", va="center", alpha=alpha)


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

    title_alpha = clamp(stage_progress(frame, 0, 6) * 1.5)
    ax.text(0.05, 0.92, "ClawBench", color=TEXT, fontsize=22, fontweight="bold", alpha=title_alpha)
    ax.text(
        0.05,
        0.87,
        "Real-world benchmark for everyday online tasks",
        color=MUTED,
        fontsize=11,
        alpha=title_alpha,
    )
    ax.text(0.95, 0.92, "Day 17", color=ACCENT, fontsize=12, fontweight="bold", ha="right", alpha=title_alpha)

    p_task = stage_progress(frame, 0, 12)
    p_live = stage_progress(frame, 8, 22)
    p_safe = stage_progress(frame, 16, 30)
    p_trace = stage_progress(frame, 24, 38)
    p_eval = stage_progress(frame, 32, 47)

    draw_task_panel(ax, p_task)
    draw_live_web_panel(ax, p_live)
    draw_intercept_panel(ax, p_safe)
    draw_trace_panel(ax, p_trace)
    draw_eval_panel(ax, p_eval)

    arrow(ax, 0.28, 0.48, 0.33, 0.48, ACCENT, alpha=clamp(p_live * 1.2))
    arrow(ax, 0.55, 0.48, 0.59, 0.48, GREEN, alpha=clamp(p_safe * 1.2))
    arrow(ax, 0.74, 0.48, 0.76, 0.35, ORANGE, alpha=clamp(p_trace * 1.2))
    arrow(ax, 0.74, 0.58, 0.76, 0.65, PURPLE, alpha=clamp(p_eval * 1.2))

    footer_alpha = clamp(stage_progress(frame, 18, 47))
    ax.text(
        0.05,
        0.06,
        "153 tasks | 144 live platforms | 5 evidence layers | binary verdict",
        color=MUTED,
        fontsize=10,
        alpha=footer_alpha,
    )

    fig.canvas.draw()
    raw = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return raw


def main():
    out_path = Path(__file__).resolve().parent / "17-clawbench.gif"
    frames = []
    for frame in range(FRAMES):
        img = build_frame(frame)
        if frame == 0:
            frames.extend([img] * 4)
        frames.append(img)
    frames.extend([frames[-1]] * 5)

    pil_frames = [
        Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=48)
        for frame in frames
    ]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=90,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(out_path)


if __name__ == "__main__":
    main()
