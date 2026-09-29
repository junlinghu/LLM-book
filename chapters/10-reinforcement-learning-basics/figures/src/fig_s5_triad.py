"""Figure for Section 10.5: divergence of off-policy semi-gradient TD.

Two states share one weight w: V(s1) = w * 1 and V(s2) = w * 2. The only
transition from s1 goes to s2 with reward 0; s2 moves to a terminal state
with reward 0, so every true value is 0 and w = 0 is the right answer.
"""
import numpy as np
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GREEN, RED, GRAY, save

X1, X2 = 1.0, 2.0          # feature of s1 and s2
ALPHA = 0.1


def run(gamma, on_policy, steps=60, w0=1.0):
    """Semi-gradient TD(0) on the single weight w.

    Off-policy: only the s1 -> s2 transition is ever updated.
    On-policy: updates alternate between s1 -> s2 and s2 -> terminal,
    as they would when following the chain.
    """
    w, ws = w0, [w0]
    for t in range(steps):
        if on_policy and t % 2 == 1:     # s2 -> terminal, reward 0
            delta = 0.0 - w * X2
            w += ALPHA * delta * X2
        else:                            # s1 -> s2, reward 0
            delta = 0.0 + gamma * w * X2 - w * X1
            w += ALPHA * delta * X1
        ws.append(w)
    return np.array(ws)


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for gamma, col in [(0.4, GREEN), (0.9, RED)]:
        off = run(gamma, on_policy=False)
        on = run(gamma, on_policy=True)
        ax.plot(off, color=col, label=f"off-policy updates, $\\gamma$ = {gamma}")
        ax.plot(on, color=col, ls="--", label=f"on-policy updates, $\\gamma$ = {gamma}")
        print(f"gamma={gamma}: w after 60 updates  off-policy {off[-1]:.3g}  on-policy {on[-1]:.3g}")
    ax.set_yscale("symlog", linthresh=1e-2)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_xlabel("update")
    ax.set_ylabel("weight $w$ (true value: 0)")
    ax.set_title("Semi-gradient TD with $V(s_1) = w$, $V(s_2) = 2w$")
    ax.legend(fontsize=9)
    save(fig, "fig9-13-deadly-triad.png")
