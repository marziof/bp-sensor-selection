import numpy as np
import networkx as nx
from src.utils.metrics import *
from tqdm import tqdm
# methods that do not implicitely use info of status nodes for selection

#-----------------
# 1. Entropy based selection
#-----------------

def entropy_sensor_selection(bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta, logger=None, G=None, alpha=0.5, beta=0.3, gamma=0.5):
    """
    Fast sensor selection using ONE BP run + entropy scoring.
    """

    target = int(rho_max * bp_base.size)

    sensor_set = set()
    sensor_order = []

    current_obs = np.empty((0, 3), dtype=int)

    # --- run BP once ---
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    marg = bp_base.marginals()  # (N, T)

    # --- precompute entropy signals ---
    H = compute_tau_entropy(marg)
    H_neigh = neighbor_entropy(G, H)
    early = early_bias(marg, gamma=gamma)

    # --- baseline score (optional reference) ---
    print(f"[Entropy method] initial global entropy = {np.mean(H):.4f}")

    #k = 0

    for k in tqdm(range(target + 1)):
        remaining = list(set(range(bp_base.size)) - sensor_set)
        if len(remaining) == 0:
            break

        # --- candidate scoring (NO BP reruns!) ---
        scores = {}

        #for i in remaining:
            # scores[i] = (
            #     -H[i]                      # uncertainty in infection time
            #     + alpha * early[i]        # early infection bias
            #     + beta * H_neigh[i]       # neighborhood uncertainty
            # )
        marginals = bp_base.marginals()
        Mt = get_Mt(marginals, t=0)
        p_base = Mt[1]  # keep true probabilistic meaning
        time_scores = time_score_from_b(marginals)
        time_scores = (time_scores - time_scores.min()) / (time_scores.max() - time_scores.min() + 1e-12)
        p_inf = p_base + alpha * time_scores
        A = nx.to_numpy_array(G)
        A = A / (A.sum(axis=1, keepdims=True) + 1e-12)

        score = p_inf + beta * (A @ p_inf)
        best_candidate = remaining[np.argmax(p_inf[remaining])] # select node with highest infection probability (biased by early infection score)
        # scores = adaptive_score(marginals, Mt, G, alpha=alpha, decay=0.3)
        # best_candidate = remaining[np.argmax(scores[remaining])]
        #best_candidate = max(scores, key=scores.get)
        sensor_set.add(best_candidate)
        sensor_order.append(best_candidate)

        # --- OPTIONAL: local bookkeeping (no BP rerun) ---
        update_cand_obs(bp_base, best_candidate, status_nodes, current_obs)
        warm_iter=20
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
        marg = bp_base.marginals()

        if k < 5 or k % 50 == 0:
            overlap = OV(np.argmax(get_Mt(marg, t=0), axis=0), status_nodes[0])
            print(f"[Entropy] selected {best_candidate} | k={k+1}/{target}, entropy changed from {H[best_candidate]:.4f} to {compute_tau_entropy(marg)[best_candidate]:.4f}")
            print(f"Overlap after adding candidate: {overlap:.4f}, error: {errors[-1]:.4f}, BP iters: {n_iter}")
        #k += 1

    return sensor_order


def update_cand_obs(bp_base, candidate, status_nodes, current_obs):
    candidate_rows = build_obs({candidate}, status_nodes)
    candidate_obs = np.vstack([current_obs, candidate_rows]) if current_obs.size else candidate_rows
    bp_base.reset_obs(candidate_obs)
    return 


def build_obs(subset, status_nodes):
    obs_rows = []
    for node in subset:
        if node is None:
            continue
        for t in range(status_nodes.shape[0]):
            # ensure status_nodes[t, node] is int (0 or 1) for obs array
            val = status_nodes[t, node]
            if isinstance(val, np.ndarray):
                print(status_nodes.shape)
                print("ARRAY FOUND:", val, val.shape, type(val))
                print("found for node", node, "at time", t)
            obs_rows.append((node, int(status_nodes[t, node]), t))
    return np.array(obs_rows, dtype=int) if obs_rows else np.empty((0, 3), dtype=int)

def adaptive_score(marg, Mt, G, alpha=0.5, decay=0.3):
    N = marg.shape[0]
    
    p_source = Mt[1]                          # P(source) — direct signal
    time_scores = time_score_from_b(marg, decay)
    
    # entropy over infection time — high = BP is uncertain about this node
    H = -np.sum(marg * np.log(marg + 1e-12), axis=1)  # (N,)
    H /= H.max() + 1e-12
    
    # current global confidence — how peaked is the posterior overall?
    # when this is high, we trust p_source more; when low, trust entropy more
    global_confidence = 1 - H.mean()         # 0 = maximally uncertain, 1 = certain
    
    score = (
        global_confidence       * p_source      # late phase: confirm infected
        + (1 - global_confidence) * H           # early phase: reduce uncertainty
        + alpha                 * time_scores   # always: bias toward early infected
    )
    return score

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))


def compute_tau_entropy(marginals):
    """
    marginals: (N, T) or (N, T+2) infection-time probabilities
    returns: entropy per node
    """
    H = np.zeros(marginals.shape[0])
    for i in range(marginals.shape[0]):
        H[i] = entropy(marginals[i])
    return H

def early_bias(marginals, gamma=0.5):
    """
    favors nodes with early infection probability
    """
    T = marginals.shape[1]
    t = np.arange(T)
    weights = np.exp(-gamma * t)

    return marginals @ weights


def neighbor_entropy(graph, H_nodes):
    """
    aggregates uncertainty in neighborhood
    """
    neigh_H = np.zeros_like(H_nodes)
    for i in range(len(H_nodes)):
        neigh = list(graph.neighbors(i))
        if len(neigh) > 0:
            neigh_H[i] = np.mean(H_nodes[neigh])
        else:
            neigh_H[i] = 0.0
    return neigh_H