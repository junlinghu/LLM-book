"""Network diagrams of the 2-2-1 worked example (Sections 2.3 and 2.6)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from style import BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, save
from worked_example import x, W1, b1, W2, b2, y, forward_backward

r = forward_backward()
pos = {"x1": (1, 3), "x2": (1, 1), "h1": (4.5, 3), "h2": (4.5, 1), "p": (8, 2)}


def base(ax, title):
    ax.set_xlim(-0.3, 10.2)
    ax.set_ylim(-0.4, 4.3)
    ax.axis("off")
    ax.set_title(title)


def node(ax, key, line1, line2, color):
    ax.add_patch(Circle(pos[key], 0.5, facecolor="white", edgecolor=color, lw=2.2, zorder=3))
    ax.text(pos[key][0], pos[key][1] + 0.13, line1, ha="center", va="center", fontsize=11, zorder=4)
    ax.text(pos[key][0], pos[key][1] - 0.2, line2, ha="center", va="center", fontsize=8.5, zorder=4)


def edge(ax, a, b, label, frac=0.3, color=GRAY, dy=0.12):
    (x0, y0), (x1, y1) = pos[a], pos[b]
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.4, zorder=1)
    lx, ly = x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)
    ax.text(lx, ly + dy, label, fontsize=9, ha="center", va="center", zorder=2,
            bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.5))


# ------------------------------------------------------------ forward
fig, ax = plt.subplots(figsize=(10.5, 4.6))
base(ax, "Forward pass through the 2-2-1 example (tanh hidden units, sigmoid output)")
edge(ax, "x1", "h1", f"{W1[0,0]}", 0.35)
edge(ax, "x1", "h2", f"{W1[0,1]}", 0.3)
edge(ax, "x2", "h1", f"{W1[1,0]}", 0.3)
edge(ax, "x2", "h2", f"{W1[1,1]}", 0.35)
edge(ax, "h1", "p", f"{W2[0]}", 0.4)
edge(ax, "h2", "p", f"{W2[1]}", 0.4)
node(ax, "x1", "$x_1$", f"{x[0]}", BLUE)
node(ax, "x2", "$x_2$", f"{x[1]}", BLUE)
node(ax, "h1", "$h_1$", f"{r['h'][0]:.4f}", ORANGE)
node(ax, "h2", "$h_2$", f"{r['h'][1]:.4f}", ORANGE)
node(ax, "p", r"$\hat p$", f"{r['p']:.4f}", GREEN)
ax.text(4.5, 3.75, f"$b^{{(1)}}_1={b1[0]}$,  $z^{{(1)}}_1={r['z1'][0]:.1f}$", ha="center", fontsize=9.5, color=PURPLE)
ax.text(4.5, 0.2, f"$b^{{(1)}}_2={b1[1]}$,  $z^{{(1)}}_2={r['z1'][1]:.1f}$", ha="center", fontsize=9.5, color=PURPLE)
ax.text(8, 2.8, f"$b^{{(2)}}={b2}$,  $z^{{(2)}}={r['z2']:.4f}$", ha="center", fontsize=9.5, color=PURPLE)
ax.text(9.3, 1.3, f"target $y={y:.0f}$\nloss $L={r['L']:.4f}$", ha="center", fontsize=10,
        bbox=dict(fc="#fff4e5", ec=ORANGE))
save(fig, "fig2-10-worked-forward.png")

# ------------------------------------------------------------ backward
fig, ax = plt.subplots(figsize=(10.5, 4.6))
base(ax, r"Backward pass: gradients $\partial L/\partial(\cdot)$ for every weight, bias, and unit")
edge(ax, "x1", "h1", f"{r['dW1'][0,0]:.4f}", 0.35, RED)
edge(ax, "x1", "h2", f"{r['dW1'][0,1]:.4f}", 0.3, RED)
edge(ax, "x2", "h1", f"{r['dW1'][1,0]:.4f}", 0.3, RED)
edge(ax, "x2", "h2", f"{r['dW1'][1,1]:.4f}", 0.35, RED)
edge(ax, "h1", "p", f"{r['dW2'][0]:.4f}", 0.4, RED)
edge(ax, "h2", "p", f"{r['dW2'][1]:.4f}", 0.4, RED)
node(ax, "x1", "$x_1$", "", BLUE)
node(ax, "x2", "$x_2$", "", BLUE)
node(ax, "h1", "$h_1$", f"{r['dh'][0]:.4f}", ORANGE)
node(ax, "h2", "$h_2$", f"{r['dh'][1]:.4f}", ORANGE)
node(ax, "p", "$z^{(2)}$", f"{r['dz2']:.4f}", GREEN)
ax.text(4.5, 3.75, rf"$\partial L/\partial z^{{(1)}}_1 = \partial L/\partial b^{{(1)}}_1 = {r['dz1'][0]:.4f}$",
        ha="center", fontsize=9.5, color=PURPLE)
ax.text(4.5, 0.2, rf"$\partial L/\partial z^{{(1)}}_2 = \partial L/\partial b^{{(1)}}_2 = {r['dz1'][1]:.4f}$",
        ha="center", fontsize=9.5, color=PURPLE)
ax.text(8, 2.8, rf"$\partial L/\partial z^{{(2)}} = \hat p - y = {r['dz2']:.4f}$", ha="center", fontsize=9.5, color=PURPLE)
ax.text(9.2, 1.1, "edges: gradient of that weight\nnodes: gradient of that value", ha="center", fontsize=9, color=RED)
save(fig, "fig2-25-worked-backward.png")
