"""Helpers for drawing the gridworld of gridworld.py."""
import numpy as np
from gridworld import ROWS, COLS, WALLS, GOAL, PIT, START, ACTIONS


def draw_grid(ax, values=None, cmap="RdBu", vmin=-1, vmax=1, fmt="{:+.2f}",
              arrows=None, env=None, show_start=True, fontsize=9):
    """Draw the grid, optionally shaded by `values` (a ROWS x COLS array)."""
    if values is not None:
        im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im = ax.imshow(np.zeros((ROWS, COLS)), cmap="Greys", vmin=0, vmax=1)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in WALLS:
                ax.add_patch(__import__("matplotlib").patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, color="#444444"))
                continue
            label = ""
            if (r, c) == GOAL:
                label = "G"
            elif (r, c) == PIT:
                label = "P"
            if label:
                ax.text(c, r, label, ha="center", va="center", fontsize=14,
                        fontweight="bold", color="k")
            elif values is not None and fmt:
                ax.text(c, r - (0.2 if arrows is not None else 0), fmt.format(values[r, c]), ha="center", va="center",
                        fontsize=fontsize, color="k")
            if show_start and (r, c) == START:
                ax.text(c - 0.42, r + 0.40, "S", ha="left", va="bottom", fontsize=8,
                        color="k", fontweight="bold")
    if arrows is not None:
        for s, (r, c) in enumerate(env.cells):
            if (r, c) in (GOAL, PIT):
                continue
            dr, dc = ACTIONS[arrows[s]]
            ax.annotate("", xy=(c + 0.22 * dc, r + 0.24 + 0.2 * dr),
                        xytext=(c - 0.22 * dc, r + 0.24 - 0.2 * dr),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.3))
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="k", lw=0.8)
    ax.grid(which="major", visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
    return im
