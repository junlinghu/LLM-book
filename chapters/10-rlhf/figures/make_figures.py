"""Generate the figures for Chapter 10. Run from this directory: python make_figures.py"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# Figure 10.5.1: the Bradley-Terry model and its loss
d = np.linspace(-6, 6, 400)
sig = 1 / (1 + np.exp(-d))
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
ax[0].plot(d, sig)
ax[0].set(title="Preference probability", xlabel=r"$r(x,y_c) - r(x,y_r)$", ylabel=r"$P(y_c \succ y_r)$")
ax[1].plot(d, -np.log(sig))
ax[1].set(title="Loss", xlabel=r"$r(x,y_c) - r(x,y_r)$", ylabel=r"$-\log \sigma(\Delta)$")
ax[2].plot(d, -(1 - sig))
ax[2].set(title="Gradient of the loss", xlabel=r"$r(x,y_c) - r(x,y_r)$", ylabel=r"$\partial \mathcal{L} / \partial \Delta = -\sigma(-\Delta)$")
for a in ax:
    a.axvline(0, color="gray", lw=0.5)
fig.tight_layout()
fig.savefig("bradley-terry.png", dpi=150)

# Figure 10.6.1: the KL-regularized optimum on a toy problem with six responses
pi_ref = np.array([0.35, 0.25, 0.18, 0.12, 0.07, 0.03])   # toy reference probabilities
r = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])              # toy rewards
def tilt(beta):
    w = pi_ref * np.exp(r / beta)
    return w / w.sum()
def kl(p):
    return float(np.sum(p * np.log(p / pi_ref)))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
width = 0.2
labels = [r"$\pi_{\mathrm{ref}}$ ($\beta \to \infty$)", r"$\beta = 2$", r"$\beta = 0.5$", r"$\beta = 0.2$"]
for k, (lab, p) in enumerate(zip(labels, [pi_ref, tilt(2.0), tilt(0.5), tilt(0.2)])):
    ax[0].bar(np.arange(6) + (k - 1.5) * width, p, width, label=lab)
ax[0].set_xticks(np.arange(6), [f"$y_{i+1}$\nr={v:g}" for i, v in enumerate(r)])
ax[0].set(ylabel="probability", title=r"$\pi^*(y) \propto \pi_{\mathrm{ref}}(y)\,\exp(r(y)/\beta)$")
ax[0].legend(frameon=False, fontsize=8)

betas = np.logspace(-1.3, 1.5, 200)
ers = [float(tilt(b) @ r) for b in betas]
kls = [kl(tilt(b)) for b in betas]
ax[1].plot(kls, ers)
for b in [2.0, 0.5, 0.2]:
    p = tilt(b)
    ax[1].plot(kl(p), p @ r, "o", color="C3")
    ax[1].annotate(rf"$\beta={b:g}$", (kl(p), p @ r), textcoords="offset points", xytext=(6, -10), fontsize=8)
ax[1].axhline(r.max(), color="gray", ls="--", lw=0.8)
ax[1].text(0.02, r.max() - 0.15, "max reward", fontsize=8, color="gray")
ax[1].set(xlabel=r"$\mathbb{D}_{\mathrm{KL}}(\pi^* \| \pi_{\mathrm{ref}})$ (nats)", ylabel="expected reward",
          title=r"Reward vs. KL as $\beta$ varies")
fig.tight_layout()
fig.savefig("kl-tradeoff.png", dpi=150)
for b in [2.0, 0.5, 0.2]:
    p = tilt(b)
    print(f"beta={b}: E[r]={p @ r:.2f}, KL={kl(p):.2f}, pi*={np.round(p, 3)}")
print(f"reference: E[r]={pi_ref @ r:.2f}")
