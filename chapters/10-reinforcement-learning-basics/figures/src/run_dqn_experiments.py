"""Train the DQN variants of Section 10.5 in parallel and save the results."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
from multiprocessing import Pool
from dqn import run_dqn

VARIANTS = {"full DQN": (True, True), "no target network": (True, False),
            "no replay": (False, True)}
SEEDS = [0, 1, 2]


def job(args):
    torch.set_num_threads(1)
    name, seed = args
    use_replay, use_target = VARIANTS[name]
    ends, rets, qlog = run_dqn(use_replay, use_target, seed=seed)
    print(f"{name:18s} seed {seed}: {len(rets)} episodes, mean return of last 20: "
          f"{rets[-20:].mean():.1f}; final mean max-Q {qlog[-1, 1]:.1f}", flush=True)
    return name, seed, ends, rets, qlog


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in SEEDS]
    with Pool(3) as p:
        out = p.map(job, jobs)
    np.save("dqn_results.npy", np.array(out, dtype=object), allow_pickle=True)
