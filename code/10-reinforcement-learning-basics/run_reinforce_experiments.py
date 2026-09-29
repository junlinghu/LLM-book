"""Train the REINFORCE variants of Section 10.6 in parallel and save the results."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import torch
from multiprocessing import Pool
from reinforce import train_reinforce

# (name, use_baseline, policy learning rate, seeds, measure variance?)
CONFIGS = [("no baseline", False, 1e-3, range(5), True),
           ("baseline", True, 1e-3, range(5), True),
           ("baseline, lr=1e-2", True, 1e-2, range(3), False),
           ("baseline, lr=3e-2", True, 3e-2, range(3), False)]


def job(args):
    torch.set_num_threads(1)
    name, use_baseline, lr, seed, var = args
    rets, variances = train_reinforce(use_baseline, n_episodes=1000, lr=lr, seed=seed,
                                      variance_every=100 if var else None)
    print(f"{name:18s} seed {seed}: mean return eps 1-200 {rets[:200].mean():6.1f}, "
          f"last 100 {rets[-100:].mean():6.1f}", flush=True)
    return name, seed, rets, variances


if __name__ == "__main__":
    jobs = [(n, b, lr, s, v) for n, b, lr, seeds, v in CONFIGS for s in seeds]
    with Pool(4) as p:
        out = p.map(job, jobs, chunksize=1)
    np.save(os.path.join(HERE, "reinforce_results.npy"), np.array(out, dtype=object), allow_pickle=True)
