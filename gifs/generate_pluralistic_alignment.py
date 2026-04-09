#!/usr/bin/env python3
"""
3Blue1Brown-style GIF: Pluralistic Alignment
- Left: Multiple colored arrows (user preferences)
- Center: LLM response distribution (collapse vs spread)
- Right: Pareto frontier
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc
from matplotlib.lines import Line2D
import matplotlib.animation as animation

# 3Blue1Brown style colors
BG_COLOR = '#1a1a2e'
AXES_COLOR = '#e6e6e6'
ACCENT_BLUE = '#4a90d9'
ACCENT_ORANGE = '#f4a261'
ACCENT_GREEN = '#2a9d8f'
ACCENT_PINK = '#e76f51'
ACCENT_PURPLE = '#9b5de5'
ACCENT_YELLOW = '#ffd166'

plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(BG_COLOR)

for ax in axes:
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal')
    ax.axis('off')

# --- LEFT PANEL: User Preference Arrows ---
ax1 = axes[0]
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylim(-0.5, 2.5)

preference_angles = [20, 45, 70, 110, 140, 170]
colors = [ACCENT_BLUE, ACCENT_ORANGE, ACCENT_GREEN, ACCENT_PINK, ACCENT_PURPLE, ACCENT_YELLOW]

# --- CENTER PANEL: LLM Response Distribution ---
ax2 = axes[1]
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-0.3, 2.5)

x_spread = np.linspace(-2, 2, 100)

# --- RIGHT PANEL: Pareto Frontier ---
ax3 = axes[2]
ax3.set_xlim(0, 2.5)
ax3.set_ylim(0, 2.5)

# Pareto frontier curve
pareto_x = np.linspace(0, 2.0, 100)
pareto_y = np.sqrt(np.maximum(0, 4 - pareto_x**2)) * 0.9
pareto_y = np.clip(pareto_y, 0, 2.5)

# Draw static Pareto frontier elements
ax3.fill_between(pareto_x, pareto_y, alpha=0.2, color=ACCENT_BLUE)
ax3.plot(pareto_x, pareto_y, color=ACCENT_BLUE, linewidth=2.5)

sample_x = [0.3, 0.8, 1.3, 1.8, 2.0]
sample_y = [np.sqrt(np.maximum(0, 4 - x**2)) * 0.9 for x in sample_x]
for px, py in zip(sample_x, sample_y):
    circle = Circle((px, py), 0.06, color=AXES_COLOR, alpha=0.6, zorder=5)
    ax3.add_patch(circle)

ax3.set_xlabel('User 1 Utility', color=AXES_COLOR, fontsize=9)
ax3.set_ylabel('User 2 Utility', color=AXES_COLOR, fontsize=9)

# Panel labels
ax1.text(1.5, 2.3, 'User Preferences', ha='center', color=AXES_COLOR, fontsize=12, fontweight='bold')
ax2.text(0, 2.3, 'LLM Response', ha='center', color=AXES_COLOR, fontsize=12, fontweight='bold')
ax3.text(1.25, 2.3, 'Pareto Frontier', ha='center', color=AXES_COLOR, fontsize=12, fontweight='bold')

ax1.text(1.5, 1.9, 'Diverse Goals', ha='center', color=AXES_COLOR, fontsize=9, alpha=0.7)
ax2.text(0, 1.9, 'Collapse vs Spread', ha='center', color=AXES_COLOR, fontsize=9, alpha=0.7)
ax3.text(1.25, 1.9, 'Trade-off Space', ha='center', color=AXES_COLOR, fontsize=9, alpha=0.7)

# Arrow lines for left panel (will be updated each frame)
arrow_lines = []
for i, (angle, color) in enumerate(zip(preference_angles, colors)):
    line = Line2D([0.5, 0.5], [0.3, 0.3], color=color, linewidth=2.5, alpha=0)
    ax1.add_line(line)
    arrow_lines.append(line)

# Distribution elements (center panel) - use polygon for fill
dist_polygon = Polygon(np.column_stack([x_spread, np.zeros_like(x_spread)]), 
                       closed=True, alpha=0.4, color=ACCENT_BLUE)
ax2.add_patch(dist_polygon)
dist_line, = ax2.plot([], [], color=ACCENT_BLUE, linewidth=2)

collapse_text = ax2.text(0, 2.5, '', ha='center', color=ACCENT_ORANGE, fontsize=11, fontweight='bold')

def init():
    return arrow_lines + [dist_line, dist_polygon, collapse_text]

def animate(frame):
    t = frame / 50.0
    
    # Animate preference arrows
    for i, line in enumerate(arrow_lines):
        arrow_time = i * 0.15
        if t > arrow_time:
            alpha = min(1.0, (t - arrow_time) * 4)
            line.set_alpha(alpha)
            angle = preference_angles[i] + 5 * np.sin(t * 2 + i)
            arrow_length = 0.8 + 0.4 * np.sin(t * 1.5 + i * 0.5)
            angle_rad = np.radians(angle)
            dx = arrow_length * np.cos(angle_rad)
            dy = arrow_length * np.sin(angle_rad)
            line.set_data([0.5, 0.5 + dx], [0.3, 0.3 + dy])
    
    # Animate distribution (collapse vs spread)
    spread_factor = 0.3 + 1.5 * (np.sin(t * 0.8) * 0.5 + 0.5)
    
    if spread_factor < 0.8:
        color = ACCENT_ORANGE
        label = 'COLLAPSE'
    else:
        color = ACCENT_GREEN
        label = 'SPREAD'
    
    collapse_text.set_text(label)
    collapse_text.set_color(color)
    
    sigma = spread_factor * 0.5
    y_vals = np.exp(-x_spread**2 / (2 * sigma**2)) * 1.8
    
    dist_line.set_data(x_spread, y_vals)
    
    # Update polygon vertices
    vertices = np.column_stack([x_spread, y_vals])
    dist_polygon.set_xy(vertices)
    dist_polygon.set_color(color)
    
    return arrow_lines + [dist_line, dist_polygon, collapse_text]

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=80, 
                               interval=80, blit=False)

# Save with reduced quality for small file size
Writer = animation.writers['pillow']
writer = Writer(fps=12, bitrate=2000)
anim.save('/Volumes/Samsung/Projects/advanced-ai-daily/gifs/13-pluralistic-alignment.gif', 
          writer=writer, dpi=80)

plt.close(fig)
print("GIF saved successfully!")