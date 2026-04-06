#!/usr/bin/env python3
"""Day 11: Gradient Boosting Attention - 3Blue1Brown style animation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gifs", "11-gradient-boosted-attention.gif")

np.random.seed(42)
n_train = 50
n_query = 100
x_train = np.sort(np.random.uniform(0, 1, n_train))
target = np.sin(2 * np.pi * x_train) + 0.3 * np.cos(4 * np.pi * x_train)
x_query = np.linspace(0, 1, n_query)
# Ground truth on query grid
target_fn = lambda x: np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x)
target_query = target_fn(x_query)

def rbf_weights(query_pts, keys):
    q = query_pts.reshape(-1, 1)
    k = keys.reshape(1, -1)
    sim = -10.0 * (q - k) ** 2
    w = np.exp(sim) / (np.exp(sim).sum(axis=1, keepdims=True) + 1e-10)
    return w

# Standard attention
w_std = rbf_weights(x_query, x_train)
standard_pred = w_std @ target

# Gradient boosted attention
gamma = 0.7
n_iter = 5
current_pred = np.zeros(n_query)
residual = target.copy()
boosted_preds_list = []

for t in range(n_iter):
    w_q = rbf_weights(x_query, x_train)
    correction = w_q @ residual
    current_pred += gamma * correction
    boosted_preds_list.append(current_pred.copy())
    w_train = rbf_weights(x_train, x_train)
    corr_at_train = w_train @ residual
    residual = target - gamma * corr_at_train

# === Frame generation ===
n_frames = n_iter + 1
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

C = dict(target="#1a1aff", standard="#ff4444", boosted="#00cc44", data="#ff8800")

def make_frame(idx):
    for ax in axes:
        ax.clear()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-1.8, 1.8)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
        ax.set_facecolor("#fafafa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("x", fontsize=11, fontweight="bold")
        ax.set_ylabel("y", fontsize=11, fontweight="bold")
    
    ax1, ax2 = axes
    it = min(idx, n_iter - 1)
    
    # Left panel
    ax1.scatter(x_train, target, s=18, c=C["data"], alpha=0.6, zorder=5)
    ax1.plot(x_query, target_query, color=C["target"], linewidth=1.5, linestyle="--", alpha=0.4, zorder=2, label="True function")
    ax1.plot(x_query, standard_pred, color=C["standard"], linewidth=2.5, zorder=3, label="Standard attention")
    err_std = np.mean((standard_pred - target_query) ** 2)
    ax1.fill_between(x_query, standard_pred, target_query, alpha=0.08, color=C["standard"])
    ax1.set_title("Standard Attention\n(One-Pass Softmax)", fontsize=13, fontweight="bold", pad=10)
    ax1.text(0.03, 0.05, f"MSE: {err_std:.4f}", transform=ax1.transAxes, fontsize=10,
             fontweight="bold", color=C["standard"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["standard"], lw=2))
    ax1.legend(loc="upper right", fontsize=9)

    # Right panel
    bp = boosted_preds_list[it]
    ax2.scatter(x_train, target, s=18, c=C["data"], alpha=0.6, zorder=5)
    ax2.plot(x_query, target_query, color=C["target"], linewidth=1.5, linestyle="--", alpha=0.4, zorder=2, label="True function")
    ax2.plot(x_query, bp, color=C["boosted"], linewidth=2.5, zorder=3, label=f"Boosted (iter {it+1})")
    err_b = np.mean((bp - target_query) ** 2)
    ax2.fill_between(x_query, bp, target_query, alpha=0.08, color=C["boosted"])
    imp = (err_std - err_b) / err_std * 100 if err_std > 0 else 0
    ax2.set_title(f"Gradient Boosted Attention\n(Iter {it + 1}/{n_iter})", fontsize=13, fontweight="bold", pad=10)
    ax2.text(0.03, 0.05, f"MSE: {err_b:.4f}", transform=ax2.transAxes, fontsize=10,
             fontweight="bold", color=C["boosted"],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C["boosted"], lw=2))
    ax2.text(0.58, 0.05, f"(-{imp:.0f}%)", transform=ax2.transAxes, fontsize=11, fontweight="bold", color="#555")
    ax2.legend(loc="upper right", fontsize=9)

# Add progress bar via figure-level
anim = FuncAnimation(fig, make_frame, frames=n_frames, interval=1500, repeat=True, blit=False)
fig.suptitle("Gradient Boosting within a Single Attention Layer", fontsize=14, fontweight="bold", y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
print(f"Saving GIF: {OUTPUT}")
anim.save(OUTPUT, writer="pillow", fps=1.0, dpi=120)

size = os.path.getsize(OUTPUT) if os.path.exists(OUTPUT) else 0
print(f"Size: {size / 1024:.1f} KB")
