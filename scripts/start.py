#!/usr/bin/env python3
"""
fractal_zoom.py
Visualize an endless Mandelbrot‑set zoom animation.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- parameters you might like to tweak ----------
width, height = 1000, 1000    # pixel resolution
max_iter       = 200          # initial maximum iterations
frames         = 600          # how many animation frames to generate
interval       = 40           # ms between frames (≈ 25 fps)
zoom_factor    = 0.97         # <1 → zoom in each frame
zoom_x, zoom_y = -0.743643887037151, 0.131825904205330  # “Seahorse Valley”
# --------------------------------------------------------

def mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Vectorized Mandelbrot escape‑time computation.
    Returns a (height × width) array of iteration counts.
    """
    # Create a complex grid
    real = np.linspace(xmin, xmax, width, dtype=np.float64)
    imag = np.linspace(ymin, ymax, height, dtype=np.float64)
    C = real + imag[:, None]*1j
    Z = np.zeros_like(C)
    output = np.zeros(C.shape, dtype=int)

    mask = np.full(C.shape, True, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + C[mask]
        escaped = np.abs(Z) > 2
        newly_escaped = escaped & mask
        output[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    output[mask] = max_iter
    return output

# Initial view window
span_x, span_y = 3.5, 3.5 * height / width
xmin = zoom_x - span_x/2
xmax = zoom_x + span_x/2
ymin = zoom_y - span_y/2
ymax = zoom_y + span_y/2

fig, ax = plt.subplots(figsize=(6, 6))
plt.axis("off")  # hide axes

# First frame
data = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
img = ax.imshow(data, cmap="hot", extent=[xmin, xmax, ymin, ymax], origin="lower")

def update(frame):
    global xmin, xmax, ymin, ymax, max_iter
    # Tighten window around the focal point
    span_x = (xmax - xmin) * zoom_factor
    span_y = (ymax - ymin) * zoom_factor
    xmin = zoom_x - span_x/2
    xmax = zoom_x + span_x/2
    ymin = zoom_y - span_y/2
    ymax = zoom_y + span_y/2
    max_iter = int(max_iter * 1.01)  # slowly increase detail

    data = mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    img.set_data(data)
    img.set_extent([xmin, xmax, ymin, ymax])
    return [img]

ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)

# ---------- DISPLAY OR SAVE ----------
plt.show()          # comment this line if you only want to save

# To save, uncomment the next two lines (requires ffmpeg installed):
# ani.save("mandelbrot_zoom.mp4", dpi=200, fps=1000/interval, codec="libx264")
# print("Saved mandelbrot_zoom.mp4")
