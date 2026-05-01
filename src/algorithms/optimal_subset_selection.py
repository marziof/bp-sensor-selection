import numpy as np
import networkx as nx
from tqdm import tqdm
from bpepi.Modules import fg_torch as fg #pytorch version
from metrics import *

###-------------------- B.O. GREEDY SENSOR SELECTION ALGORITHM: select optimal subset of nodes as sensor ----------------------###
#### CAREFUL: only for small N!
import itertools

def bayes_optimal_subset(N, T, contacts, delta, status_nodes, rho, max_iter=50, tol=1e-6, damp=0.5, gt=False):
    """
    Exact Bayes-optimal subset by brute-force search over all subsets of size k.
    gt=True: maximize OV (requires ground truth)
    gt=False: maximize MOV (no ground truth)
    Only tractable for very small N.
    """
    k = int(rho * N)
    nodes = list(range(N))
    best_subset = None
    best_reward = -np.inf

    for subset in itertools.combinations(nodes, k):
        obs_rows = []
        for node in subset:
            for t in range(status_nodes.shape[0]):
                obs_rows.append((node, int(status_nodes[t, node]), t))

        obs_array = np.array(obs_rows, dtype=int) if len(obs_rows) > 0 else np.empty((0, 3), dtype=int)

        bp_fg = fg.FactorGraph(N, T, contacts, obs_array, delta)
        bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

        marg = bp_fg.marginals()
        Mt = get_Mt(marg, t=0)

        if gt:
            x_est = np.argmax(Mt, axis=0)
            reward = OV(x_est, status_nodes[0])
        else:
            reward = MOV(Mt)

        if reward > best_reward:
            best_reward = reward
            best_subset = subset

    return best_subset, best_reward