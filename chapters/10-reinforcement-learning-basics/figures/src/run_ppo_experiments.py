"""Train the PPO configurations of Section 10.7 in parallel and save the results."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from multiprocessing import Pool
from ppo import train_ppo

BASE = dict(total_steps=100_000, lr=1e-3, epochs=10, clip_eps=0.2, lam=0.95)
CONFIGS = [("PPO (clip 0.2, lambda 0.95)", {}, range(5)),
           ("no clipping", dict(clip_eps=1e9), range(5)),
           ("lambda = 0", dict(lam=0.0), range(3)),
           ("lambda = 1", dict(lam=1.0), range(3))]


def job(args):
    torch.set_num_threads(1)
    name, overrides, seed = args
    log = train_ppo(**{**BASE, **overrides}, seed=seed)
    print(f"{name:28s} seed {seed}: final return {log[-1, 1]:6.1f}, "
          f"max KL {np.nanmax(log[:, 2]):.3f}", flush=True)
    return name, seed, log


if __name__ == "__main__":
    jobs = [(n, o, s) for n, o, seeds in CONFIGS for s in seeds]
    with Pool(3) as p:
        out = p.map(job, jobs, chunksize=1)
    np.save("ppo_results.npy", np.array(out, dtype=object), allow_pickle=True)
