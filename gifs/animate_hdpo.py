#!/usr/bin/env python3
"""
HDPO: Meta-Cognitive Tool Use Animation
3Blue1B style - matplotlib + ffmpeg
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import subprocess
import os
import tempfile

# 3B1B Color Palette
BG = "#0a0a14"
TX = "#ece6e0"
BLUE = "#5ca0ff"
RED = "#ff4b4b"
GREEN = "#00d2d3"
YELLOW = "#ffd43b"
PURPLE = "#b340eb"
GRAY = "#868e96"

fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Grid
for i in range(17):
    ax.axvline(i, color=TX, alpha=0.03, lw=1)
for i in range(10):
    ax.axhline(i, color=TX, alpha=0.03, lw=1)

def clamp(x):
    return max(0.0, min(1.0, x))

def ease(t):
    if t < 0:
        return 0
    elif t > 1:
        return 1
    return t * t * (3 - 2 * t)

def ease_out(t):
    if t < 0:
        return 0
    elif t > 1:
        return 1
    return 1 - (1 - t) ** 3

TOTAL = 300
tmp = tempfile.mkdtemp()

for f in range(TOTAL):
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Redraw grid
    for i in range(17):
        ax.axvline(i, color=TX, alpha=0.03, lw=1)
    for i in range(10):
        ax.axhline(i, color=TX, alpha=0.03, lw=1)

    # ============ PHASE 1: Title (0-40) ============
    if f < 40:
        t = ease(f / 40)

        ax.text(8, 5.5, "HDPO", fontsize=72, fontweight='bold',
                color=BLUE, ha='center', va='center', alpha=t)
        ax.text(8, 4.2, "Hybrid Decoupled Policy Optimization",
                fontsize=28, color=TX, ha='center', va='center', alpha=t)
        ax.text(8, 3.2, "Meta-Cognitive Tool Use in Agentic Models",
                fontsize=20, color=GRAY, ha='center', va='center', alpha=t * 0.8)

        ax.text(8, 1.5, "arXiv:2604.08545 · Act Wisely",
                fontsize=14, color=GRAY, ha='center', va='center', alpha=t * 0.6)

    # ============ PHASE 2: The Problem (40-110) ============
    elif f < 110:
        t = ease((f - 40) / 70)

        # Agent icon (left)
        ax.add_patch(patches.Circle((2.5, 5.5), 0.6, facecolor=BLUE, alpha=t))
        ax.text(2.5, 5.5, "A", fontsize=24, fontweight='bold',
                color=BG, ha='center', va='center')

        # Question box
        ax.add_patch(patches.FancyBboxPatch((3.5, 4.5), 4.5, 2,
                boxstyle="round,pad=0.1", facecolor="#1a1a2e", edgecolor=YELLOW, alpha=t))
        ax.text(5.75, 5.5, "? Visual Query", fontsize=16, color=YELLOW,
                ha='center', va='center')

        # Tool box
        ax.add_patch(patches.FancyBboxPatch((8.5, 4.5), 3.5, 2,
                boxstyle="round,pad=0.1", facecolor="#1a1a2e", edgecolor=RED, alpha=t))
        ax.text(10.25, 5.5, "[Tool]", fontsize=16, color=RED,
                ha='center', va='center')

        # Arrows showing blind invocation
        if t > 0.3:
            arrow_t = ease(clamp((t - 0.3) / 0.4))
            ax.annotate("", xy=(8.4, 5.5), xytext=(3.2, 5.5),
                       arrowprops=dict(arrowstyle="->", color=RED, lw=2), alpha=arrow_t)
            ax.text(5.5, 6.2, "Blind Tool\nInvocation", fontsize=12,
                    color=RED, ha='center', va='center', alpha=arrow_t)

        if t > 0.6:
            problem_t = ease(clamp((t - 0.6) / 0.3))
            ax.text(8, 2.5, "Problem: Agent doesn't know WHEN to use tools",
                    fontsize=18, color=RED, ha='center', va='center', alpha=problem_t)

    # ============ PHASE 3: Scalarized Reward Dilemma (110-180) ============
    elif f < 180:
        t = ease((f - 110) / 70)

        ax.text(8, 7.8, "Traditional RL: Scalarized Reward",
                fontsize=22, color=TX, ha='center', va='center', alpha=t)

        # Reward equation
        ax.text(4, 6.5, r"$R_{total} = R_{acc} - \lambda \cdot T_{penalty}$",
                fontsize=20, color=TX, ha='center', va='center', alpha=t)

        if t > 0.3:
            dilemma_t = ease(clamp((t - 0.3) / 0.5))
            # Left side - aggressive
            ax.add_patch(patches.FancyBboxPatch((1, 3), 5.5, 2.5,
                    boxstyle="round,pad=0.1", facecolor="#2d1a1a", edgecolor=RED, alpha=dilemma_t))
            ax.text(3.75, 4.7, r"$\lambda$ HIGH", fontsize=16, color=RED, ha='center', alpha=dilemma_t)
            ax.text(3.75, 4.0, "Tool use suppressed", fontsize=14, color=RED, ha='center', alpha=dilemma_t*0.8)
            ax.text(3.75, 3.5, "Essential calls blocked", fontsize=12, color=GRAY, ha='center', alpha=dilemma_t*0.6)

            # Right side - mild
            ax.add_patch(patches.FancyBboxPatch((9.5, 3), 5.5, 2.5,
                    boxstyle="round,pad=0.1", facecolor="#1a2d1a", edgecolor=YELLOW, alpha=dilemma_t))
            ax.text(12.25, 4.7, r"$\lambda$ MILD", fontsize=16, color=YELLOW, ha='center', alpha=dilemma_t)
            ax.text(12.25, 4.0, "Penalty overwhelmed", fontsize=14, color=YELLOW, ha='center', alpha=dilemma_t*0.8)
            ax.text(12.25, 3.5, "by accuracy variance", fontsize=12, color=GRAY, ha='center', alpha=dilemma_t*0.6)

        if t > 0.7:
            conclusion_t = ease(clamp((t - 0.7) / 0.3))
            ax.text(8, 1.8, "Irreconcilable optimization dilemma!",
                    fontsize=20, color=RED, ha='center', va='center', alpha=conclusion_t)

    # ============ PHASE 4: HDPO Solution (180-260) ============
    elif f < 260:
        t = ease((f - 180) / 80)

        ax.text(8, 8.3, "HDPO: Decoupled Reward Channels",
                fontsize=24, color=GREEN, ha='center', va='center', alpha=t)

        # Channel 1 - Accuracy
        if t > 0.15:
            acc_t = ease(clamp((t - 0.15) / 0.3))
            ax.add_patch(patches.FancyBboxPatch((1.5, 5), 6, 2.5,
                    boxstyle="round,pad=0.1", facecolor="#1a2d2d", edgecolor=BLUE, alpha=acc_t))
            ax.text(4.5, 6.7, "Channel 1: Accuracy", fontsize=18, color=BLUE,
                    ha='center', va='center', alpha=acc_t)
            ax.text(4.5, 6.0, r"$R_{acc} \rightarrow$ Task Correctness",
                    fontsize=16, color=TX, ha='center', va='center', alpha=acc_t*0.9)
            ax.text(4.5, 5.3, "Maximize accuracy", fontsize=14, color=GRAY,
                    ha='center', va='center', alpha=acc_t*0.7)

        # Channel 2 - Efficiency
        if t > 0.4:
            eff_t = ease(clamp((t - 0.4) / 0.3))
            ax.add_patch(patches.FancyBboxPatch((8.5, 5), 6, 2.5,
                    boxstyle="round,pad=0.1", facecolor="#2d1a2d", edgecolor=GREEN, alpha=eff_t))
            ax.text(11.5, 6.7, "Channel 2: Efficiency", fontsize=18, color=GREEN,
                    ha='center', va='center', alpha=eff_t)
            ax.text(11.5, 6.0, r"$R_{eff} \rightarrow$ Only if Accurate",
                    fontsize=16, color=TX, ha='center', va='center', alpha=eff_t*0.9)
            ax.text(11.5, 5.3, "Conditional advantage", fontsize=14, color=GRAY,
                    ha='center', va='center', alpha=eff_t*0.7)

        # Key insight
        if t > 0.65:
            insight_t = ease(clamp((t - 0.65) / 0.3))
            ax.text(8, 3.5, "Conditional Advantage Estimation",
                    fontsize=20, color=YELLOW, ha='center', va='center', alpha=insight_t)
            ax.text(8, 2.8, "Efficiency optimized ONLY within correct trajectories",
                    fontsize=16, color=TX, ha='center', va='center', alpha=insight_t*0.8)

        if t > 0.85:
            result_t = ease(clamp((t - 0.85) / 0.15))
            ax.add_patch(patches.FancyBboxPatch((4, 1.5), 8, 1.2,
                    boxstyle="round,pad=0.1", facecolor="#1a3d1a", edgecolor=GREEN, alpha=result_t))
            ax.text(8, 2.1, "Cognitive Curriculum: Master task → Then refine self-reliance",
                    fontsize=16, color=GREEN, ha='center', va='center', alpha=result_t)

    # ============ PHASE 5: Result (260-300) ============
    else:
        t = ease((f - 260) / 40)

        ax.text(8, 6.5, "Result: Metis Model",
                fontsize=36, fontweight='bold', color=GREEN, ha='center', va='center', alpha=t)

        if t > 0.3:
            stat1_t = ease(clamp((t - 0.3) / 0.3))
            ax.text(5, 4.8, "Tool Invocations", fontsize=20, color=TX, ha='center', alpha=stat1_t)
            ax.text(5, 4.0, "↓ Orders of Magnitude", fontsize=28, color=RED, ha='center', fontweight='bold', alpha=stat1_t)

        if t > 0.5:
            stat2_t = ease(clamp((t - 0.5) / 0.3))
            ax.text(11, 4.8, "Reasoning Accuracy", fontsize=20, color=TX, ha='center', alpha=stat2_t)
            ax.text(11, 4.0, "↑ Improved", fontsize=28, color=GREEN, ha='center', fontweight='bold', alpha=stat2_t)

        if t > 0.8:
            bottom_t = ease(clamp((t - 0.8) / 0.2))
            ax.text(8, 2.0, "First solve the problem, then solve it efficiently",
                    fontsize=18, color=YELLOW, ha='center', va='center', alpha=bottom_t)
            ax.text(8, 1.2, "Cognitive curriculum induced by decoupled optimization",
                    fontsize=14, color=GRAY, ha='center', va='center', alpha=bottom_t*0.7)

    fig.savefig(f"{tmp}/frame_{f:04d}.png", dpi=100, bbox_inches='tight',
                facecolor=BG, pad_inches=0.3)
    if f % 50 == 0:
        print(f"  Frame {f}/{TOTAL}")

# Compile to GIF
OUTPUT_GIF = "/Volumes/Samsung/Projects/advanced-ai-daily/gifs/15-hdpo.gif"
subprocess.run([
    "ffmpeg", "-y", "-framerate", "24", "-i", f"{tmp}/frame_%04d.png",
    "-vf", "fps=24,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
    "-loop", "0", OUTPUT_GIF
], check=True, capture_output=True)

# Check file size
size = os.path.getsize(OUTPUT_GIF) / 1024
print(f"Saved: {OUTPUT_GIF} ({size:.1f} KB)")
if size < 100:
    print("WARNING: GIF file too small, might be corrupted")