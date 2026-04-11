#!/usr/bin/env python3
"""
Day 16 GIF: Routing Distraction in Multimodal MoE
Animates: (1) text-only routing vs visual input routing, (2) layer-wise divergence
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import imageio.v3 as iio

# ── helpers ──────────────────────────────────────────────────────────────────
def dark_bg(ax):
    ax.set_facecolor("#0d1117")
    fig = ax.figure
    fig.patch.set_facecolor("#0d1117")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

def rounded_rect(ax, xy, w, h, color, alpha=1.0, lw=1.5, ec="#58a6ff"):
    r = patches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05",
                               facecolor=color, edgecolor=ec, alpha=alpha, lw=lw)
    ax.add_patch(r)

def arrow(ax, x0, y0, x1, y1, color="#8b949e", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

def textlabel(ax, x, y, s, fs=9, color="#c9d1d9", bold=False):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs,
            color=color, fontweight="bold" if bold else "normal")

# ── layout constants ──────────────────────────────────────────────────────────
EXPERTS = ["Expert-1", "Expert-2", "Expert-3", "Expert-4", "Expert-5", "Expert-6"]
N_EXP = len(EXPERTS)
EXP_COLORS = ["#1f6feb", "#238636", "#a371f7", "#f0883e", "#3fb950", "#58a6ff"]
EXP_Y     = 0.32
EXP_H     = 0.13
EXP_W     = 0.70

LAYER_X  = [0.08, 0.28, 0.48, 0.68, 0.88]
LAYER_LABELS = ["L1", "L3", "L5", "L7", "L9"]
N_LAYERS = len(LAYER_X)

IMG_Y    = 0.82
IMG_W    = 0.10

TEXT_W   = 0.10

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
dark_bg(ax)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.97, "Routing Distraction in Multimodal MoE",
        ha="center", va="top", fontsize=15, fontweight="bold",
        color="#f0f6fc")
ax.text(0.5, 0.93, '"Seeing but Not Thinking" — Layer-wise Expert Activation',
        ha="center", va="top", fontsize=10, color="#8b949e")

# ── Left: image input ─────────────────────────────────────────────────────────
# Image box
rounded_rect(ax, (0.01, IMG_Y - 0.10), IMG_W, 0.14, "#21262d", alpha=0.9, ec="#58a6ff")
ax.text(0.06, IMG_Y - 0.03, "🖼️ Image", ha="center", va="center", fontsize=10, color="#c9d1d9")

# Text box
rounded_rect(ax, (0.01, IMG_Y - 0.26), IMG_W, 0.14, "#21262d", alpha=0.9, ec="#8b949e")
ax.text(0.06, IMG_Y - 0.19, "📝 Text", ha="center", va="center", fontsize=10, color="#c9d1d9")

# ── Layer columns ─────────────────────────────────────────────────────────────
def draw_layer_col(ax, lx, ly_center, activated, alpha, show_label=True):
    """Draw one layer column with routing bars"""
    bar_w = EXP_W / N_EXP
    for i, (exp, col, ec) in enumerate(zip(EXPERTS, EXP_COLORS, EXP_COLORS)):
        bx = lx - EXP_W / 2 + i * bar_w + 0.01
        act = activated[i]
        fill_alpha = alpha * (0.3 + 0.7 * act)
        ec2 = "#f0f6fc" if act > 0.5 else col
        rounded_rect(ax, (bx, ly_center - EXP_H / 2), bar_w - 0.02, EXP_H,
                     col, alpha=fill_alpha, lw=1, ec=ec2)
        ax.text(bx + (bar_w - 0.02) / 2, ly_center,
                f"{int(act*100)}%", ha="center", va="center",
                fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

LAYER_CENTER_Y = 0.50

# Expert labels
for i, (exp, col) in enumerate(zip(EXPERTS, EXP_COLORS)):
    bx = 0.25 - EXP_W / 2 + i * (EXP_W / N_EXP) + (EXP_W / N_EXP) / 2
    ax.text(bx, 0.71, exp, ha="center", va="bottom", fontsize=7, color=col)

# ── Text-only routing (top half) ───────────────────────────────────────────────
ax.text(0.22, 0.70, "Text-Only Input", ha="center", va="bottom",
        fontsize=9, color="#58a6ff", style="italic")

# Arrow from text box to first layer
arrow(ax, 0.11, IMG_Y - 0.26, LAYER_X[0], 0.66, color="#58a6ff", lw=1.5)

text_activations = [
    [0.10, 0.85, 0.10, 0.10, 0.10, 0.10],  # L1 — domain expert dominates
    [0.10, 0.80, 0.10, 0.10, 0.10, 0.10],   # L3
    [0.10, 0.75, 0.10, 0.10, 0.10, 0.10],   # L5
    [0.10, 0.70, 0.10, 0.10, 0.10, 0.10],   # L7
    [0.10, 0.65, 0.10, 0.10, 0.10, 0.10],   # L9
]

# ── Image routing (bottom half) ────────────────────────────────────────────────
ax.text(0.22, 0.52, "Visual Input", ha="center", va="bottom",
        fontsize=9, color="#f0883e", style="italic")

# Arrow from image box to first layer (bottom path)
arrow(ax, 0.11, IMG_Y - 0.03, LAYER_X[0], 0.55, color="#f0883e", lw=1.5)

img_activations = [
    [0.55, 0.15, 0.15, 0.15, 0.15, 0.15],   # L1 — visual experts activate
    [0.60, 0.10, 0.10, 0.10, 0.10, 0.10],   # L3 — routing distraction begins
    [0.65, 0.08, 0.08, 0.08, 0.08, 0.08],   # L5 — peak distraction
    [0.60, 0.10, 0.10, 0.10, 0.10, 0.10],    # L7
    [0.45, 0.15, 0.15, 0.15, 0.15, 0.15],    # L9 — domain expert weakly activated
]

# ── Baseline vs routing-guided ───────────────────────────────────────────────
# Separator
ax.axhline(0.75, xmin=0.01, xmax=0.99, color="#30363d", lw=0.8, linestyle="--")

# Bottom half label
ax.text(0.50, 0.72, "Baseline Routing", ha="center", va="bottom",
        fontsize=9, color="#f0883e")
ax.text(0.50, 0.01, "After Routing-Guided Intervention: Domain Expert Activated",
        ha="center", va="bottom", fontsize=9, color="#3fb950")

# ── Build frames ──────────────────────────────────────────────────────────────
N_FRAMES = 48

# Phase 1 (0-15): Show text-only routing — Expert-2 dominates all layers
# Phase 2 (16-31): Show visual routing — visual experts distract, domain expert weak
# Phase 3 (32-47): Routing-guided intervention — domain expert activates for visual inputs

frames = []
for frame in range(N_FRAMES):
    fig_temp, ax_temp = plt.subplots(figsize=(12, 7))
    ax_temp.set_xlim(0, 1); ax_temp.set_ylim(0, 1)
    dark_bg(ax_temp)

    # Title
    ax_temp.text(0.5, 0.97, "Routing Distraction in Multimodal MoE",
                 ha="center", va="top", fontsize=15, fontweight="bold", color="#f0f6fc")
    ax_temp.text(0.5, 0.93, '"Seeing but Not Thinking" — Layer-wise Expert Activation',
                 ha="center", va="top", fontsize=10, color="#8b949e")

    # Separator
    ax_temp.axhline(0.75, xmin=0.01, xmax=0.99, color="#30363d", lw=0.8, linestyle="--")

    # Expert labels (top)
    for i, (exp, col) in enumerate(zip(EXPERTS, EXP_COLORS)):
        bx = 0.25 - EXP_W / 2 + i * (EXP_W / N_EXP) + (EXP_W / N_EXP) / 2
        ax_temp.text(bx, 0.71, exp, ha="center", va="bottom", fontsize=7, color=col)

    # Image & text boxes
    rounded_rect(ax_temp, (0.01, IMG_Y - 0.10), IMG_W, 0.14, "#21262d", alpha=0.9, ec="#58a6ff")
    ax_temp.text(0.06, IMG_Y - 0.03, "Visual", ha="center", va="center", fontsize=10, color="#c9d1d9")
    rounded_rect(ax_temp, (0.01, IMG_Y - 0.26), IMG_W, 0.14, "#21262d", alpha=0.9, ec="#8b949e")
    ax_temp.text(0.06, IMG_Y - 0.19, "Text", ha="center", va="center", fontsize=10, color="#c9d1d9")

    # Phase logic
    if frame < 16:
        phase = "text_only"
        t = frame / 16.0
        # Fade in text-only routing
        alpha_mult = min(1.0, t * 3)
        ax_temp.text(0.50, 0.70, "Text-Only: Domain Expert Stays Active",
                     ha="center", va="bottom", fontsize=10, color="#58a6ff")
        ax_temp.text(0.50, 0.52, "Visual: Routing Distraction Starts Here",
                     ha="center", va="bottom", fontsize=9, color="#21262d")

        # Text routing — show gradually
        for li, (lx, acts) in enumerate(zip(LAYER_X, text_activations)):
            alpha_l = alpha_mult * max(0, min(1, (t - li * 0.15) * 6))
            if alpha_l <= 0: continue
            for i, (exp, col, ec) in enumerate(zip(EXPERTS, EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                fill_alpha = alpha_l * (0.3 + 0.7 * acts[i])
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 + 0.23),
                             bar_w - 0.02, EXP_H, col, alpha=fill_alpha,
                             lw=1, ec=("#f0f6fc" if acts[i] > 0.5 else col))
                if alpha_l > 0.5:
                    ax_temp.text(bx + (bar_w - 0.02) / 2, LAYER_CENTER_Y + 0.23,
                                 f"{int(acts[i]*100)}%", ha="center", va="center",
                                 fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

        # Visual routing (dim)
        for li, (lx, acts) in enumerate(zip(LAYER_X, img_activations)):
            for i, (col, ec) in enumerate(zip(EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 - 0.01),
                             bar_w - 0.02, EXP_H, col, alpha=0.08,
                             lw=1, ec=col)

        # Arrows
        arrow(ax_temp, 0.11, IMG_Y - 0.19, LAYER_X[0], 0.66, color="#58a6ff", lw=1.5)
        arrow(ax_temp, 0.11, IMG_Y - 0.03, LAYER_X[0], 0.55, color="#f0883e", lw=1.5)

    elif frame < 32:
        phase = "visual_distraction"
        t = (frame - 16) / 16.0
        alpha_mult = min(1.0, t * 3)

        ax_temp.text(0.50, 0.70, "Text-Only: Domain Expert Dominates",
                     ha="center", va="bottom", fontsize=10, color="#58a6ff")

        # Text routing (solid)
        for li, (lx, acts) in enumerate(zip(LAYER_X, text_activations)):
            for i, (col, ec) in enumerate(zip(EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                fill_alpha = (0.3 + 0.7 * acts[i])
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 + 0.23),
                             bar_w - 0.02, EXP_H, col, alpha=fill_alpha,
                             lw=1, ec=("#f0f6fc" if acts[i] > 0.5 else col))
                ax_temp.text(bx + (bar_w - 0.02) / 2, LAYER_CENTER_Y + 0.23,
                             f"{int(acts[i]*100)}%", ha="center", va="center",
                             fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

        # Visual routing (distraction animation)
        ax_temp.text(0.50, 0.52, "Visual: Visual Experts Distract — Domain Expert Starved!",
                     ha="center", va="bottom", fontsize=10, color="#f0883e")
        for li, (lx, acts) in enumerate(zip(LAYER_X, img_activations)):
            alpha_l = alpha_mult * max(0, min(1, (t - li * 0.12) * 5))
            if alpha_l <= 0: continue
            for i, (col, ec) in enumerate(zip(EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                fill_alpha = alpha_l * (0.3 + 0.7 * acts[i])
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 - 0.01),
                             bar_w - 0.02, EXP_H, col, alpha=fill_alpha,
                             lw=1, ec=("#f0f6fc" if acts[i] > 0.5 else col))
                if alpha_l > 0.5:
                    ax_temp.text(bx + (bar_w - 0.02) / 2, LAYER_CENTER_Y - 0.01,
                                 f"{int(acts[i]*100)}%", ha="center", va="center",
                                 fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

        arrow(ax_temp, 0.11, IMG_Y - 0.19, LAYER_X[0], 0.66, color="#58a6ff", lw=1.5)
        arrow(ax_temp, 0.11, IMG_Y - 0.03, LAYER_X[0], 0.55, color="#f0883e", lw=1.5)

        # Highlight distraction at L5
        if t > 0.5:
            lx5 = LAYER_X[2]
            ax_temp.add_patch(patches.FancyBboxPatch((lx5 - 0.38, 0.04), 0.76, 0.46,
                               boxstyle="round,pad=0.02", facecolor="none",
                               edgecolor="#f0883e", lw=2, linestyle="--", alpha=0.8))
            ax_temp.text(lx5, 0.03, "Peak\nDistraction!", ha="center", va="bottom",
                         fontsize=7, color="#f0883e")

    else:
        phase = "intervention"
        t = (frame - 32) / 16.0
        alpha_mult = min(1.0, t * 3)

        ax_temp.text(0.50, 0.70, "Text-Only: Domain Expert Dominates",
                     ha="center", va="bottom", fontsize=10, color="#58a6ff")

        # Text routing (solid)
        for li, (lx, acts) in enumerate(zip(LAYER_X, text_activations)):
            for i, (col, ec) in enumerate(zip(EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                fill_alpha = (0.3 + 0.7 * acts[i])
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 + 0.23),
                             bar_w - 0.02, EXP_H, col, alpha=fill_alpha,
                             lw=1, ec=("#f0f6fc" if acts[i] > 0.5 else col))
                ax_temp.text(bx + (bar_w - 0.02) / 2, LAYER_CENTER_Y + 0.23,
                             f"{int(acts[i]*100)}%", ha="center", va="center",
                             fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

        # Intervention: domain expert +1 now activates for visual
        intervened_acts = [
            [0.25, 0.65, 0.15, 0.15, 0.15, 0.15],   # L1
            [0.20, 0.70, 0.15, 0.15, 0.15, 0.15],   # L3
            [0.15, 0.75, 0.15, 0.15, 0.15, 0.15],   # L5 — FIXED!
            [0.20, 0.70, 0.15, 0.15, 0.15, 0.15],   # L7
            [0.25, 0.65, 0.15, 0.15, 0.15, 0.15],   # L9
        ]

        ax_temp.text(0.50, 0.52, "Visual + Routing-Guided Intervention: Domain Expert Activated!",
                     ha="center", va="bottom", fontsize=10, color="#3fb950")
        for li, (lx, acts) in enumerate(zip(LAYER_X, intervened_acts)):
            alpha_l = alpha_mult * max(0, min(1, (t - li * 0.12) * 5))
            if alpha_l <= 0: continue
            for i, (col, ec) in enumerate(zip(EXP_COLORS, EXP_COLORS)):
                bar_w = EXP_W / N_EXP
                bx = lx - EXP_W / 2 + i * bar_w + 0.01
                fill_alpha = alpha_l * (0.3 + 0.7 * acts[i])
                rounded_rect(ax_temp, (bx, LAYER_CENTER_Y - EXP_H / 2 - 0.01),
                             bar_w - 0.02, EXP_H, col, alpha=fill_alpha,
                             lw=1, ec=("#f0f6fc" if acts[i] > 0.5 else col))
                if alpha_l > 0.5:
                    ax_temp.text(bx + (bar_w - 0.02) / 2, LAYER_CENTER_Y - 0.01,
                                 f"{int(acts[i]*100)}%", ha="center", va="center",
                                 fontsize=6.5, color="#f0f6fc" if fill_alpha > 0.4 else "#8b949e")

        arrow(ax_temp, 0.11, IMG_Y - 0.19, LAYER_X[0], 0.66, color="#58a6ff", lw=1.5)
        arrow(ax_temp, 0.11, IMG_Y - 0.03, LAYER_X[0], 0.55, color="#3fb950", lw=1.5)

        if t > 0.5:
            lx5 = LAYER_X[2]
            ax_temp.add_patch(patches.FancyBboxPatch((lx5 - 0.38, 0.04), 0.76, 0.46,
                               boxstyle="round,pad=0.02", facecolor="none",
                               edgecolor="#3fb950", lw=2, linestyle="--", alpha=0.8))
            ax_temp.text(lx5, 0.03, "Expert-2\nRestored!", ha="center", va="bottom",
                         fontsize=7, color="#3fb950")

    # Legend
    ax_temp.text(0.94, 0.95, "Expert-2\n(Domain)", ha="center", va="top",
                 fontsize=7, color=EXP_COLORS[1], bbox=dict(boxstyle="round,pad=0.2",
                 facecolor="#0d1117", edgecolor=EXP_COLORS[1], lw=1))
    ax_temp.text(0.94, 0.85, "Visual Experts\n(Others)", ha="center", va="top",
                 fontsize=7, color="#f0883e", bbox=dict(boxstyle="round,pad=0.2",
                 facecolor="#0d1117", edgecolor="#f0883e", lw=1))

    fig_temp.canvas.draw()
    raw = np.array(fig_temp.canvas.renderer.buffer_rgba())
    plt.close(fig_temp)
    frames.append(raw)

# Hold last frame
frames.extend([frames[-1]] * 4)

# Save
out_path = "/Volumes/Samsung/Projects/advanced-ai-daily/gifs/16-routing-distraction.gif"
iio.imwrite(out_path, frames, duration=80, loop=0)
print(f"Saved: {out_path} ({len(frames)} frames)")
