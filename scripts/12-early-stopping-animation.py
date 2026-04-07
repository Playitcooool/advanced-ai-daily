#!/usr/bin/env python3
"""
3Blue1Brown-style animation: Early Stopping for Large Reasoning Models
via Confidence Dynamics.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os, glob

BG = '#0f0f2f'
COLOR_A  = '#5dade2'   # blue continuing
COLOR_B  = '#2ecc71'   # green early stop
COLOR_C  = '#e74c3c'   # red overthinking
COLOR_W  = '#f5f5f5'
COLOR_Y  = '#f1c40f'
COLOR_G  = '#7f8c8d'
FONT = 'DejaVu Sans'

os.makedirs('gifs', exist_ok=True)
os.makedirs('frames_tmp', exist_ok=True)

np.random.seed(42)
rng = np.random.RandomState(42)
n_steps = 50
confidence = np.zeros(n_steps)
for t in range(11):
    confidence[t] = 0.15 + 0.05*t + rng.normal(0, 0.04)
for t in range(11, 26):
    p = (t-11)/14
    confidence[t] = 0.5 + 0.35*p + rng.normal(0, 0.025)
plateau = 0.87
for t in range(26, 36):
    confidence[t] = plateau + rng.normal(0, 0.012)
for t in range(36, n_steps):
    drift = -0.005*(t-35)
    confidence[t] = plateau + drift + rng.normal(0, 0.018)
confidence = np.clip(confidence, 0.05, 0.99)

confidence_delta = np.zeros_like(confidence)
for t in range(1, len(confidence)):
    confidence_delta[t] = confidence[t] - confidence[t-1]

window = 5
confidence_stability = np.zeros_like(confidence)
for t in range(window, len(confidence)):
    confidence_stability[t] = np.std(confidence[t-window:t])

quality = np.clip(confidence*1.05, 0, 1.0)
quality[35:] = quality[35:] - 0.008*np.arange(len(quality[35:]))
quality = np.clip(quality, 0, 1.0)

threshold = 0.025
early_stop_idx = 28

print(f"Early stop at step {early_stop_idx}, savings {(1-early_stop_idx/45)*100:.0f}%")

def style(ax, title, xlim, ylim):
    ax.set_facecolor(BG)
    ax.set_title(title, color=COLOR_W, fontsize=13, fontweight='bold', fontname=FONT)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(colors=COLOR_G)
    for sp in ax.spines.values():
        sp.set_color(COLOR_G)

total_frames = 55
for frame in range(total_frames):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(BG)
    t = min(max(frame-3, 1), 50)

    # Panel 1: Confidence
    ax1 = axes[0,0]
    style(ax1, 'Confidence Over Time', [0, 50], [0, 1.05])
    ax1.set_xticks(range(0, 51, 10))
    ax1.set_xlabel('Reasoning Step', color=COLOR_G, fontsize=9, fontname=FONT)
    ax1.set_ylabel('Confidence', color=COLOR_G, fontsize=9, fontname=FONT)
    ax1.plot(np.arange(t), confidence[:t], color=COLOR_A, linewidth=2.5)
    ax1.axhline(y=0.8, color=COLOR_Y, linestyle=':', linewidth=1, alpha=0.3)
    if frame > early_stop_idx+1:
        ax1.axvline(x=early_stop_idx, color=COLOR_B, linestyle='--', lw=2)
        ax1.axvspan(early_stop_idx, 45, alpha=0.08, color=COLOR_C)
    if frame > 44:
        ax1.axvline(x=45, color=COLOR_C, linestyle=':', lw=1.5, alpha=0.5)

    # Panel 2: Delta
    ax2 = axes[0,1]
    style(ax2, 'Confidence Change Rate', [0, 50], [-0.06, 0.08])
    ax2.set_xticks(range(0, 51, 10))
    ax2.axhline(y=0, color=COLOR_G, linewidth=0.5)
    ax2.set_xlabel('Step', color=COLOR_G, fontsize=9, fontname=FONT)
    ax2.set_ylabel('d(conf)/dt', color=COLOR_G, fontsize=9, fontname=FONT)
    if t > 1:
        cols2 = [COLOR_B if i <= early_stop_idx else COLOR_C for i in range(t)]
        ax2.bar(np.arange(t), confidence_delta[:t], color=cols2, alpha=0.75, width=0.8)
    if frame > early_stop_idx+1:
        ax2.axvline(x=early_stop_idx, color=COLOR_B, linestyle='--', lw=1.5)

    # Panel 3: Stability
    ax3 = axes[1,0]
    style(ax3, 'Stability (Rolling Std w=5)', [0, 50], [0, 0.065])
    ax3.set_xticks(range(0, 51, 10))
    ax3.axhline(y=threshold, color=COLOR_Y, linestyle='--', lw=1.5, alpha=0.6)
    ax3.set_xlabel('Step', color=COLOR_G, fontsize=9, fontname=FONT)
    ax3.set_ylabel('Std', color=COLOR_G, fontsize=9, fontname=FONT)
    if t > window:
        ax3.plot(np.arange(window, t), confidence_stability[window:t], color=COLOR_A, lw=2.5)
        if t > early_stop_idx:
            xs = np.arange(early_stop_idx, t)
            ys = confidence_stability[early_stop_idx:t]
            ax3.bar(xs, ys, color=COLOR_B, alpha=0.4, width=0.8)
            ax3.axvline(x=early_stop_idx, color=COLOR_B, linestyle='--', lw=1.5)

    # Panel 4: Quality vs Compute
    ax4 = axes[1,1]
    style(ax4, 'Quality vs. Compute', [0, 50], [0, 1.05])
    ax4.set_xticks(range(0, 51, 10))
    ax4.set_xlabel('Compute (steps)', color=COLOR_G, fontsize=9, fontname=FONT)
    ax4.set_ylabel('Quality', color=COLOR_G, fontsize=9, fontname=FONT)
    ax4.plot(np.arange(t), quality[:t], color=COLOR_A, lw=2, alpha=0.7)
    if frame > early_stop_idx:
        ax4.plot([early_stop_idx], [quality[early_stop_idx]], 'go', markersize=12, alpha=0.9)
        if frame > early_stop_idx+3:
            end = min(frame-3, 45)
            if end > early_stop_idx:
                ax4.plot(np.arange(early_stop_idx+1, end+1), quality[early_stop_idx+1:end+1],
                        color=COLOR_C, lw=2, alpha=0.5, linestyle='--')

    if frame < 5:
        ttl = 'When Should Reasoning Models Stop Thinking?'
    elif frame < 45:
        sav = (1-early_stop_idx/45)*100
        ttl = f'Early stop @ step {early_stop_idx} saves {sav:.0f}% compute'
    else:
        ttl = 'Day 12: Early Stopping via Confidence Dynamics'
    fig.suptitle(ttl, color=COLOR_W if frame < 45 else COLOR_Y,
                 fontsize=15, fontweight='bold', fontname=FONT, y=0.99)
    plt.tight_layout()
    p = f'frames_tmp/frame_{frame:03d}.png'
    fig.savefig(p, dpi=100, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    if frame % 10 == 0:
        print(f"  frame {frame}/{total_frames}")

print("\nCombining into GIF...")
frames = []
for f in sorted(glob.glob('frames_tmp/frame_*.png')):
    img = Image.open(f)
    w, h = img.size
    frames.append(img.resize((w//2, h//2), Image.LANCZOS))

if frames:
    out = 'gifs/12-early-stopping.gif'
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    kb = os.path.getsize(out) / 1024
    print(f"GIF saved: {out} ({kb:.1f} KB)")
    for f in glob.glob('frames_tmp/frame_*.png'):
        os.remove(f)
    os.rmdir('frames_tmp')
else:
    print("ERROR: no frames!")
    exit(1)
