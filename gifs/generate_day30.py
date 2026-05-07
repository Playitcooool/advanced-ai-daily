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
YELLOW = "#f5a623"


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


def draw_node(ax, x, y, r, color, label, sublabel="", alpha=1.0):
    circle = patches.CirclePolygon((x, y), r, facecolor=color, edgecolor=(1, 1, 1, 0.15), linewidth=1.2, alpha=alpha)
    ax.add_patch(circle)
    ax.text(x, y + 0.012, label, color="white", fontsize=7.5, fontweight="bold", ha="center", alpha=alpha)
    if sublabel:
        ax.text(x, y - 0.018, sublabel, color=TEXT, fontsize=5.5, ha="center", alpha=alpha * 0.8)


def draw_edge(ax, x0, y0, x1, y1, color, alpha=1.0, lw=2.0):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def draw_arrow(ax, x0, y0, x1, y1, color, label="", label_offset=(0, 0.022), alpha=1.0, lw=1.5):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=lw, color=color, alpha=alpha),
    )
    if label:
        ax.text(x0 + label_offset[0], y0 + label_offset[1], label, color=color, fontsize=6.5, ha="center", alpha=alpha)


def draw_var_bar(ax, x, y, w, h, fast_val, slow_val, alpha=1.0):
    """Draw fast+slow variable indicator on an edge."""
    bh = 0.018
    by = y - bh / 2
    # bg
    bg = patches.FancyBboxPatch((x, by), w, bh, boxstyle="round,pad=0.003", facecolor="#1a2744", edgecolor="none", alpha=alpha * 0.6)
    ax.add_patch(bg)
    # fast (orange)
    fw = w * fast_val
    if fw > 0.005:
        fast_patch = patches.FancyBboxPatch((x, by), fw, bh, boxstyle="round,pad=0.003", facecolor=ORANGE, edgecolor="none", alpha=alpha)
        ax.add_patch(fast_patch)
    # slow (cyan) overlay
    sw = w * slow_val
    if sw > 0.005:
        slow_patch = patches.FancyBboxPatch((x, by + bh * 0.5), sw, bh * 0.5, boxstyle="round,pad=0.003", facecolor=CYAN, edgecolor="none", alpha=alpha * 0.9)
        ax.add_patch(slow_patch)


class GraphState:
    def __init__(self):
        # Node positions (normalized 0-1)
        self.nodes = {
            "S": (0.13, 0.50),   # Source / new input
            "A": (0.35, 0.72),   # Node A
            "B": (0.35, 0.28),   # Node B
            "C": (0.57, 0.50),   # Node C
            "D": (0.80, 0.50),   # Output / consolidated
        }
        # Edge fast/slow values (0-1)
        self.edges = {
            ("S", "A"): [0.0, 0.0],
            ("S", "B"): [0.0, 0.0],
            ("A", "C"): [0.0, 0.0],
            ("B", "C"): [0.0, 0.0],
            ("C", "D"): [0.0, 0.0],
            ("A", "B"): [0.0, 0.0],  # cross link
        }

    def step(self):
        """Advance memory dynamics one step."""
        for edge in self.edges:
            f, s = self.edges[edge]
            # Fast decays quickly, slow builds gradually
            self.edges[edge][0] = max(0, f * 0.82)          # fast decay
            self.edges[edge][1] = min(1.0, s * 0.97 + f * 0.04)  # slow consolidation

    def activate(self, edge_key, strength=0.9):
        f, s = self.edges[edge_key]
        self.edges[edge_key][0] = min(1.0, f + strength)
        self.edges[edge_key][1] = min(1.0, s + strength * 0.05)


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

    ax.text(0.02, 0.96, "Day 30: Memini", color=TEXT, fontsize=18, fontweight="bold")
    ax.text(0.02, 0.91, "Multi-Timescale Memory Dynamics for LLM External Knowledge", color=MUTED, fontsize=10)

    # ---- Panel: Memory Graph ----
    panel_alpha = clamp(stage(frame, -2, 20) * 1.3)
    rounded_box(ax, 0.01, 0.04, 0.60, 0.82, PANEL, alpha=0.95 * panel_alpha, radius=0.04, lw=1.2)
    ax.text(0.31, 0.83, "Associative Memory Graph", color=TEXT, fontsize=11, fontweight="bold", ha="center", alpha=panel_alpha)

    # Node positions
    N = {
        "S": (0.13, 0.50),
        "A": (0.35, 0.72),
        "B": (0.35, 0.28),
        "C": (0.57, 0.50),
        "D": (0.80, 0.50),
    }

    node_labels = {
        "S": ("INPUT", "new knowledge"),
        "A": ("Concept A", ""),
        "B": ("Concept B", ""),
        "C": ("Bridge", ""),
        "D": ("STORED", "consolidated"),
    }

    # ---- Simulate memory dynamics over phases ----
    phases = [
        # (phase_frame, events)
        # Phase 1: New input arrives
        (8,  [("S", "A", 0.9), ("S", "B", 0.85)]),
        # Phase 2: Cross association
        (18, [("A", "B", 0.7), ("A", "C", 0.6), ("B", "C", 0.6)]),
        # Phase 3: Consolidation path
        (28, [("C", "D", 0.8)]),
        # Phase 4: Repeated → stronger slow
        (38, [("S", "A", 0.5), ("S", "B", 0.5), ("A", "C", 0.3), ("B", "C", 0.3), ("C", "D", 0.3)]),
    ]

    # Build a local graph state that advances with frame
    gs = GraphState()
    for pf, events in phases:
        if frame >= pf:
            for e in events:
                gs.activate((e[0], e[1]), strength=e[2])
            for _ in range(min(frame - pf, 8)):
                gs.step()

    # Draw edges
    edge_colors = {
        ("S", "A"): BLUE,
        ("S", "B"): PURPLE,
        ("A", "C"): GREEN,
        ("B", "C"): ORANGE,
        ("C", "D"): CYAN,
        ("A", "B"): YELLOW,
    }

    for (src, dst), (fx, sx) in gs.edges.items():
        x0, y0 = N[src]
        x1, y1 = N[dst]
        total = fx + sx * 0.5
        color = edge_colors.get((src, dst), MUTED)
        alpha_edge = clamp(total * 1.5) * panel_alpha
        lw = 1.0 + total * 3.0
        if total > 0.02:
            draw_edge(ax, x0, y0, x1, y1, color, alpha=alpha_edge * 0.6, lw=lw)
            # Variable bar at midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            bw = 0.07
            if total > 0.05:
                draw_var_bar(ax, mx - bw / 2, my, bw, 0.025, fx, sx, alpha=alpha_edge)

    # Draw nodes
    for nid, (nx, ny) in N.items():
        label, sublabel = node_labels[nid]
        node_alpha = panel_alpha
        if nid == "S":
            node_color = BLUE
        elif nid == "D":
            node_color = GREEN
        else:
            node_color = PANEL
        draw_node(ax, nx, ny, 0.042, node_color, label, sublabel, alpha=node_alpha)

    # New input pulse animation
    pulse_t = (frame % 20) / 20.0
    pulse_alpha = clamp(math.sin(pulse_t * math.pi) * 1.5) * clamp(stage(frame, 4, 44))
    pulse_r = 0.042 + pulse_t * 0.025
    pulse_circle = patches.CirclePolygon((N["S"][0], N["S"][1]), pulse_r, facecolor="none", edgecolor=BLUE, linewidth=1.5, alpha=pulse_alpha * 0.7)
    ax.add_patch(pulse_circle)

    # Labels for variable types
    var_legend_alpha = clamp(stage(frame, 14, 30)) * panel_alpha
    ax.text(0.31, 0.10, "F", color=ORANGE, fontsize=8, fontweight="bold", ha="center", alpha=var_legend_alpha)
    ax.text(0.31, 0.07, "Fast (immediate)", color=ORANGE, fontsize=6, ha="center", alpha=var_legend_alpha)
    ax.text(0.42, 0.10, "S", color=CYAN, fontsize=8, fontweight="bold", ha="center", alpha=var_legend_alpha)
    ax.text(0.42, 0.07, "Slow (consolidation)", color=CYAN, fontsize=6, ha="center", alpha=var_legend_alpha)

    # ---- RIGHT: Key Properties ----
    prop_alpha = clamp(stage(frame, 14, 34) * 1.3)
    rounded_box(ax, 0.64, 0.04, 0.34, 0.82, PANEL, alpha=0.95 * prop_alpha, radius=0.04, lw=1.2)
    ax.text(0.81, 0.83, "Emergent Properties", color=TEXT, fontsize=11, fontweight="bold", ha="center", alpha=prop_alpha)

    props = [
        (GREEN,  "Episodic Sensitivity",  "New associations\nimmediately usable",  stage(frame, 16, 30)),
        (CYAN,   "Gradual Consolidation", "Repeated confirms\nstrengthen slow var",  stage(frame, 22, 36)),
        (ORANGE, "Selective Forgetting",  "Unused edges fade\nfast >> slow decay",   stage(frame, 28, 44)),
    ]

    prop_y_start = 0.72
    prop_h = 0.18
    for i, (color, title, desc, t) in enumerate(props):
        py = prop_y_start - i * (prop_h + 0.04)
        pa = clamp(t * 2) * prop_alpha
        if pa < 0.01:
            continue
        rounded_box(ax, 0.66, py - prop_h, 0.30, prop_h, color, alpha=0.25 * pa, radius=0.02)
        ax.text(0.81, py - 0.05, title, color=color, fontsize=8.5, fontweight="bold", ha="center", alpha=pa)
        ax.text(0.81, py - 0.11, desc, color=TEXT, fontsize=6.5, ha="center", alpha=pa * 0.85, linespacing=1.3)

    # Benna-Fusi label
    bffa = clamp(stage(frame, 20, 40) * prop_alpha)
    ax.text(0.81, 0.14, "Benna-Fusi Model", color=MUTED, fontsize=7.5, ha="center", alpha=bffa)
    ax.text(0.81, 0.09, "Coupled fast + slow\nsynaptic variables", color=MUTED, fontsize=6, ha="center", alpha=bffa * 0.7, linespacing=1.3)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(image[:, :, :3])


def main():
    frames = [build_frame(frame) for frame in range(FRAMES)]
    output_path = Path(__file__).with_name("30-memini.gif")
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
