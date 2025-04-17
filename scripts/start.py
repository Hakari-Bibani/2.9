#!/usr/bin/env python3
"""
scripts/start.py
Generate a Mandelbrot‑set “endless” zoom frame‑by‑frame, then
save the *last* frame to mandelbrot.png.

• Works both locally (`python scripts/start.py`) and in CI.
• No GUI needed – we skip plt.show() when the CI environment
  variable is present (GitHub Actions sets CI=true).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- tweakables ----------
WIDTH, HEIGHT = 1000, 1000
MAX_ITER      = 200
FRAMES        = 120            # reduce if you want a faster CI run
INTERVAL_MS   = 40
ZOOM_FACTOR   = 0.97
FOCUS_X, FOCUS_Y = -0.743643887037151, 0.131825904205330
# --------------------------------

def mandelbrot(xmin, xmax, ymin, ymax, w, h, max_iter):
    # Vectorised escape‑time
    real = np.linspace(xmin, xmax, w, dtype=np.float64)
    imag = np.linspace(ymin, ymax, h, dtype=np.float64)
    C = real + imag[:, None]*1j
    Z = np.zeros_like(C)
    M = np.full(C.shape, True, dtype=bool)
    output = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        Z[M] = Z[M]**2 + C[M]
        escaped = np.abs(Z) > 2
        output[escaped & M] = i
        M &= ~escaped
        if not M.any():
            break
    output[M] = max_iter
    return output

# initial bounds
span_x = span_y = 3.5
xmin, xmax = FOCUS_X - span_x/2, FOCUS_X + span_x/2
ymin, ymax = FOCUS_Y - span_y/2, FOCUS_Y + span_y/2

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")
img = ax.imshow(
    mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER),
    cmap="hot", extent=[xmin, xmax, ymin, ymax], origin="lower"
)

def update(_):
    global xmin, xmax, ymin, ymax, MAX_ITER
    span_x = (xmax - xmin) * ZOOM_FACTOR
    span_y = (ymax - ymin) * ZOOM_FACTOR
    xmin, xmax = FOCUS_X - span_x/2, FOCUS_X + span_x/2
    ymin, ymax = FOCUS_Y - span_y/2, FOCUS_Y + span_y/2
    MAX_ITER = int(MAX_ITER * 1.02)
    data = mandelbrot(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, MAX_ITER)
    img.set_data(data)
    img.set_extent([xmin, xmax, ymin, ymax])
    return [img]

ani = FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL_MS, blit=True)

# In CI we just save an image; locally we preview
if os.getenv("CI"):
    ani.event_source.stop()           # finish immediately
    fig.savefig("mandelbrot.png", dpi=300, bbox_inches="tight")
    print("✅  Saved mandelbrot.png")
else:
    plt.show()
