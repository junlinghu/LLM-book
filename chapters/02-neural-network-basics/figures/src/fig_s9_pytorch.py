"""From-scratch NumPy MLP vs. the same model in PyTorch (Section 2.9)."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from style import BLUE, ORANGE, GREEN, RED, GRAY, save, make_moons
import numpy_mlp as nm

torch.set_default_dtype(torch.float64)
X, y = make_moons(n=400, noise=0.2, seed=2)
EPOCHS, LR, BS = 40, 0.3, 32

# ---- from scratch
P = nm.init_params(2, 16, 2, seed=0)
P0 = {k: v.copy() for k, v in P.items()}
P, hist_np = nm.train_sgd(P, X, y, lr=LR, epochs=EPOCHS, batch_size=BS, seed=0)

# ---- PyTorch with the same initial weights and the same minibatch order
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
with torch.no_grad():
    model[0].weight.copy_(torch.from_numpy(P0["W1"].T))   # nn.Linear stores (out, in)
    model[0].bias.copy_(torch.from_numpy(P0["b1"]))
    model[2].weight.copy_(torch.from_numpy(P0["W2"].T))
    model[2].bias.copy_(torch.from_numpy(P0["b2"]))
opt = torch.optim.SGD(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
rng = np.random.default_rng(0)
hist_pt = []
for epoch in range(EPOCHS):
    order = rng.permutation(len(X))
    for start in range(0, len(X), BS):
        idx = torch.from_numpy(order[start:start + BS])
        loss = loss_fn(model(Xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        hist_pt.append(loss_fn(model(Xt), yt).item())

hist_np, hist_pt = np.array(hist_np), np.array(hist_pt)
print("max |loss difference| over training:", np.abs(hist_np - hist_pt).max())
print("final losses", hist_np[-1], hist_pt[-1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.plot(np.arange(1, EPOCHS + 1), hist_np, color=BLUE, lw=4, alpha=0.5, label="NumPy, manual backprop")
ax.plot(np.arange(1, EPOCHS + 1), hist_pt, color=RED, lw=1.5, ls="--", label="PyTorch autograd")
ax.set_xlabel("epoch")
ax.set_ylabel("training cross-entropy")
ax.set_title("Same weights, same minibatches: identical curves")
ax.legend()
ax = axes[1]
diff = np.abs(hist_np - hist_pt)
ep = np.arange(1, EPOCHS + 1)
nz = diff > 0
ax.semilogy(ep[nz], diff[nz], "o", color=GRAY, ms=5, label="nonzero difference")
ax.set_ylim(1e-18, 1e-14)
ax.text(0.03, 0.08, f"{(~nz).sum()} of {EPOCHS} epochs: difference exactly 0 (not shown)",
        transform=ax.transAxes, fontsize=9, color=GRAY)
print("epochs with exactly zero difference:", (~nz).sum())
ax.set_xlabel("epoch")
ax.set_ylabel("|NumPy loss − PyTorch loss|")
ax.set_title("Difference stays at floating-point round-off (float64)")
save(fig, "fig2-39-numpy-vs-torch.png")

# ---- gradient comparison at the initial weights on one batch
P = {k: v.copy() for k, v in P0.items()}
Z, cache = nm.forward(P, X[:64])
_, dZ = nm.softmax_cross_entropy(Z, y[:64])
g_np = nm.backward(P, cache, dZ)
model2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
with torch.no_grad():
    model2[0].weight.copy_(torch.from_numpy(P0["W1"].T)); model2[0].bias.copy_(torch.from_numpy(P0["b1"]))
    model2[2].weight.copy_(torch.from_numpy(P0["W2"].T)); model2[2].bias.copy_(torch.from_numpy(P0["b2"]))
loss_fn(model2(Xt[:64]), yt[:64]).backward()
pairs = {
    "W1": (g_np["W1"], model2[0].weight.grad.numpy().T),
    "b1": (g_np["b1"], model2[0].bias.grad.numpy()),
    "W2": (g_np["W2"], model2[2].weight.grad.numpy().T),
    "b2": (g_np["b2"], model2[2].bias.grad.numpy()),
}
for k, (a, b) in pairs.items():
    print(f"{k}: max abs diff {np.abs(a - b).max():.2e}, allclose={np.allclose(a, b, rtol=1e-7, atol=1e-10)}")
