"""Figures for Section 2.6: computational graphs and backpropagation."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN, save

# ---------------------------------------------------------------- Figure 2.22: forward/backward on a tiny expression
# Compute L = (tanh(w*x + b) - y)^2 with concrete numbers; draw the graph.
# x=0.5, w=1.2, b=-0.3, y=0.8
x, w, b, y = 0.5, 1.2, -0.3, 0.8
u = w * x          # 0.6
z = u + b          # 0.3
h = np.tanh(z)     # ~0.2913
e = h - y          # ~-0.5087
L = e ** 2         # ~0.2588
print("forward", dict(u=u, z=z, h=h, e=e, L=L))

# reverse
dL_dL = 1.0
dL_de = 2 * e
dL_dh = dL_de * 1.0
dL_dz = dL_dh * (1 - h ** 2)
dL_du = dL_dz * 1.0
dL_db = dL_dz * 1.0
dL_dw = dL_du * x
dL_dx = dL_du * w
print("backward", dict(de=dL_de, dh=dL_dh, dz=dL_dz, dw=dL_dw, db=dL_db))

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title(r"Computational graph for $L=(\tanh(wx+b)-y)^2$  "
             r"with $x=0.5,\,w=1.2,\,b=-0.3,\,y=0.8$")

# nodes: (name, x, y, fwd_value, bwd_value or None)
nodes = [
    ("$x$", 1.0, 3.5, f"{x}", None),
    ("$w$", 1.0, 1.5, f"{w}", f"{dL_dw:.3f}"),
    ("$\\times$", 3.0, 2.5, f"u={u:.2f}", f"{dL_du:.3f}"),
    ("$b$", 3.0, 0.7, f"{b}", f"{dL_db:.3f}"),
    ("$+$", 5.0, 2.0, f"z={z:.2f}", f"{dL_dz:.3f}"),
    ("$\\tanh$", 7.0, 2.0, f"h={h:.3f}", f"{dL_dh:.3f}"),
    ("$y$", 7.0, 0.7, f"{y}", None),
    ("$-$", 9.0, 1.5, f"e={e:.3f}", f"{dL_de:.3f}"),
    ("$(\\cdot)^2$", 11.0, 1.5, f"L={L:.3f}", "1"),
]
edges = [(0, 2), (1, 2), (2, 4), (3, 4), (4, 5), (5, 7), (6, 7), (7, 8)]

pos = {i: (nx, ny) for i, (_, nx, ny, _, _) in enumerate(nodes)}


def to_border(p, q, hw=0.58, hh=0.38):
    """Point where the segment p->q leaves the box of half-size (hw, hh) centered at p."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return (p[0] + t * dx, p[1] + t * dy)


for i, j in edges:
    ax.annotate("", xy=to_border(pos[j], pos[i]), xytext=to_border(pos[i], pos[j]),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.5, shrinkA=0, shrinkB=0))

for i, (name, nx, ny, fwd, bwd) in enumerate(nodes):
    ax.add_patch(FancyBboxPatch((nx - 0.55, ny - 0.35), 1.1, 0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor="white", edgecolor=BLUE, lw=1.8, zorder=2))
    ax.text(nx, ny + 0.08, name, ha="center", va="center", fontsize=11, zorder=3)
    ax.text(nx, ny - 0.22, fwd, ha="center", va="center", fontsize=8, color=GREEN, zorder=3)
    if bwd is not None:
        var = {1: "w", 2: "u", 3: "b", 4: "z", 5: "h", 7: "e", 8: "L"}[i]
        ax.text(nx, ny - 0.55, rf"$\partial L/\partial {var} = {bwd}$", ha="center", va="top",
                fontsize=8, color=RED, zorder=3)

ax.text(6, 4.5, "green = forward value · red = gradient of $L$ with respect to that node",
        ha="center", fontsize=10, color=GRAY)
save(fig, "fig2-22-comp-graph.png")

# ---------------------------------------------------------------- Figure 2.23: forward vs reverse mode cartoon
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax in axes:
    ax.set_xlim(0, 6)
    ax.set_ylim(0.7, 3.9)
    ax.axis("off")
# left: forward mode — seed one input
ax = axes[0]
ax.set_title("Forward mode AD\n(one seed per input)")
xs = [1, 2.5, 4.0, 5.2]
for i, x_ in enumerate(xs):
    ax.add_patch(Circle((x_, 2), 0.28, facecolor="white", edgecolor=BLUE, lw=1.5))
    if i < len(xs) - 1:
        ax.annotate("", xy=(xs[i + 1] - 0.28, 2), xytext=(x_ + 0.28, 2),
                    arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
ax.text(3, 3.2, "push directional\nderivative forward", ha="center", color=GREEN)
ax.text(1, 1.2, "seed\n$\\dot x=1$", ha="center", fontsize=9)
ax.text(5.2, 1.2, "get\n$\\dot L$", ha="center", fontsize=9)
# right: reverse mode
ax = axes[1]
ax.set_title("Reverse mode AD = backprop\n(one seed on the scalar loss)")
for i, x_ in enumerate(xs):
    ax.add_patch(Circle((x_, 2), 0.28, facecolor="white", edgecolor=BLUE, lw=1.5))
    if i < len(xs) - 1:
        ax.annotate("", xy=(x_ + 0.28, 2), xytext=(xs[i + 1] - 0.28, 2),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.text(3, 3.2, "pull gradient\nbackward", ha="center", color=RED)
ax.text(5.2, 1.2, "seed\n$\\overline{L}=1$", ha="center", fontsize=9)
ax.text(1, 1.2, "get all\n$\\partial L/\\partial w$", ha="center", fontsize=9)
save(fig, "fig2-23-fwd-vs-rev.png")

