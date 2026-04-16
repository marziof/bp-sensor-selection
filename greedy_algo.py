import numpy as np
import networkx as nx
from tqdm import tqdm
from bpepi.Modules import fg_torch as fg #pytorch version
from metrics import *

###------------------- GREEDY SENSOR SELECTION ALGORITHM: 1 sensor at a time chosen greedily ----------------------###
def run_bp_greedy(initial_obs, rho_max, N, T, contacts, delta, status_nodes, max_iter=200, tol=1e-6, damp=0.5, gt=None, print_progress=False):
    """ Greedy sensor selection based on OV or MOV gain.
        If gt is provided, we compute OV and select the node that maximizes OV gain.
        If gt is None, we compute MOV and select the node that maximizes MOV gain.
    """
    # 1) Initialize with initial_obs (usually empty to fully select all sensors, one at a time)
    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    # 2) Compute initial BP from initial_obs (should be empty)
    bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_fg.update(maxit=max_iter, tol=tol, damp=damp)
    mo_before = MOV(get_Mt(bp_fg.marginals(), t=0))
    target_n = int(rho_max * N)
    selected_nodes_history = []
    ov_history = []

    # 3) Compute one at a time until target sensor density is reached
    while len(already_selected) < target_n:
        if len(already_selected) >= N:
            break

        best_mo = -np.inf
        best_ov = -np.inf
        best_node = None
        best_gain = -np.inf

        # try each node as candidate and compute gain in MO or OV compared to current_obs
        for candidate in range(N):
            if candidate in already_selected:
                continue
            candidate_rows = [(candidate, int(status_nodes[t, candidate]), t) for t in range(status_nodes.shape[0])]
            candidate_obs = np.vstack([current_obs, np.array(candidate_rows, dtype=int)])

            # bp with candidate_obs
            bp_cand = fg.FactorGraph(N, T, contacts, candidate_obs, delta)
            bp_cand.update(maxit=max_iter, tol=tol, damp=damp)
            marg = bp_cand.marginals()  # (N, T+2)
            Mt = get_Mt(marg, t=0)      # (2, N)

            # if gt is given, compute OV gain; else compute MOV gain
            if gt is None:
                #mo = MOV(Mt)  # MOV with x_est = all infected (worst case)
                mo_after = MOV(Mt)
                gain = mo_after - mo_before
                if gain > best_gain:
                    best_gain = gain
                    best_mo = mo_after
                    best_node = candidate
            else:
                x_est = np.argmax(Mt, axis=0)
                ov = OV(x_est, gt)
                if ov > best_ov:
                    best_ov = ov
                    best_node = candidate

        # add best node's full trajectory to observations
        new_rows = [(best_node, int(status_nodes[t, best_node]), t) for t in range(status_nodes.shape[0])]
        already_selected.add(best_node)
        current_obs = np.vstack([current_obs, np.array(new_rows, dtype=int)])
        selected_nodes_history.append(best_node)

        if gt is not None:
            ov_history.append(best_ov)
            if print_progress:
                print(f"selected {best_node}, OV={best_ov:.4f}, rho={len(already_selected)/N:.3f}")
        else:
            if print_progress:
                print(f"selected {best_node}, MO={best_mo:.4f}, rho={len(already_selected)/N:.3f}")

        # update bp_fg with new observations for next iteration
        bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
        bp_fg.update(maxit=max_iter, tol=tol, damp=damp)
        mo_before = MOV(get_Mt(bp_fg.marginals(), t=0))

    # return final bp_fg, final observations (with target rho), history of selected nodes and OV values (if gt provided)
    return bp_fg, current_obs, selected_nodes_history, ov_history 


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

        reward = evaluate(subset, N, T, contacts, delta, status_nodes, max_iter=max_iter, tol=tol, damp=damp, gt=gt)

        if reward > best_reward:
            best_reward = reward
            best_subset = subset

    return best_subset, best_reward


def build_obs(subset, status_nodes):
    obs_rows = []
    for node in subset:
        for t in range(status_nodes.shape[0]):
            obs_rows.append((node, int(status_nodes[t, node]), t))
    return np.array(obs_rows, dtype=int) if obs_rows else np.empty((0, 3), dtype=int)

def evaluate(subset, N, T, contacts, delta, status_nodes, max_iter=50, tol=1e-6, damp=0.5, gt=False):
        obs = build_obs(subset, status_nodes)
        bp_fg = fg.FactorGraph(N, T, contacts, obs, delta)
        bp_fg.update(maxit=max_iter, tol=tol, damp=damp)
        marg = bp_fg.marginals()
        Mt = get_Mt(marg, t=0)
        if gt:
            x_est = np.argmax(Mt, axis=0)
            return OV(x_est, status_nodes[0])
        else:
            return MOV(Mt)


###-------------------- B.O. SAMPLE/REPLACE SENSOR SELECTION ALGORITHM: select optimal subset of nodes as sensor - random + sample & replace ----------------------###
import matplotlib.pyplot as plt
import seaborn as sns

def sampleReplaceSubset(N, T, contacts, delta, status_nodes, rho, max_iter=50, tol=1e-6, damp=0.5, gt=False, m=None):
    k = int(rho * N)
    nodes = list(range(N))
    best_subset = set(np.random.choice(nodes, size=k, replace=False))
    best_reward = evaluate(best_subset, N, T, contacts, delta, status_nodes, max_iter=max_iter, tol=tol, damp=damp, gt=gt)
    all_overlaps = [best_reward]    
    # iterative local search: cycle through sensors until no improvement
    improved = True
    while improved:
        improved = False
        for sensor in list(best_subset): # tqdm(list(best_subset), desc="Iterating through sensors"):  # iterate over copy since we may modify best_subset
            candidate_subset = set(nodes) - best_subset
            best_candidate = None
            best_candidate_reward = best_reward
            if m is not None and len(candidate_subset) > m:
                candidate_subset = set(np.random.choice(list(candidate_subset), size=m, replace=False))
            for candidate in candidate_subset:
                temp_subset = (best_subset - {sensor}) | {candidate}
                reward = evaluate(temp_subset, N, T, contacts, delta, status_nodes, max_iter=max_iter, tol=tol, damp=damp, gt=gt)
                if reward > best_candidate_reward:
                    best_candidate_reward = reward
                    best_candidate = candidate

                # exit candidate loop early if perfect
                if best_candidate_reward >= (1.0 if gt else 0.95):
                    break
            
            # accept best swap for this sensor if it improves
            if best_candidate is not None:
                best_subset = (best_subset - {sensor}) | {best_candidate}
                best_reward = best_candidate_reward
                # if best reward improved by at least 5%, keep going; else stop
                if best_reward - all_overlaps[-1] >= 0.05 * all_overlaps[-1]:
                    improved = True  # found improvement, keep outer loop going
                else:
                    improved = False  # no significant improvement, stop
                all_overlaps.append(best_reward)
    
    return best_subset, best_reward, all_overlaps






####-------------------- HELPER FUNCTIONS FOR GREEDY SELECTION ----------------------###
def compute_mean_dist(G, node, selected_set):
    if len(selected_set) == 0:
        return 0.0
    
    dists = []
    for s in selected_set:
        try:
            d = nx.shortest_path_length(G, source=node, target=s)
            dists.append(d)
        except nx.NetworkXNoPath:
            dists.append(np.inf)
    
    return float(np.mean(dists))


def compute_subset_features(G, subset):
    """
    Structural features of a selected sensor subset for BP + epidemic inference.

    Keeps only features that are meaningful for:
    - redundancy
    - spread in graph
    - observability of epidemic process
    - ER vs RRG comparison
    """

    subset = list(subset)
    N = G.number_of_nodes()

    # ---------------- EMPTY CASE ----------------
    if len(subset) == 0:
        return {
            "subset_size": 0,
            "mean_pairwise_distance": np.nan,
            "boundary_size": 0,
            "density": 0.0,
            "mean_degree": np.nan,
            "degree_bias": np.nan
        }

    subset_set = set(subset)

    # ---------------- NODE DEGREE ----------------
    degrees = np.array([G.degree(n) for n in subset])
    mean_degree = float(np.mean(degrees))

    # ---------------- GLOBAL GRAPH DEGREE (for ER bias) ----------------
    global_mean_degree = np.mean([d for _, d in G.degree()])
    degree_bias = mean_degree / global_mean_degree if global_mean_degree > 0 else np.nan

    # ---------------- DENSITY (redundancy proxy) ----------------
    G_sub = G.subgraph(subset)
    density = nx.density(G_sub)

    # ---------------- BOUNDARY SIZE (epidemic observability) ----------------
    boundary = set()
    for n in subset:
        for nb in G.neighbors(n):
            if nb not in subset_set:
                boundary.add(nb)

    boundary_size = len(boundary)

    # ---------------- PAIRWISE DISTANCE (redundancy) ----------------
    if len(subset) > 1:
        dist_sum = 0.0
        count = 0

        for i in subset:
            lengths = nx.single_source_shortest_path_length(G, i)
            for j in subset:
                if i < j:
                    dist_sum += lengths.get(j, np.inf)
                    count += 1

        mean_pairwise_distance = dist_sum / count if count > 0 else np.nan
    else:
        mean_pairwise_distance = np.nan

    # ---------------- RETURN ----------------
    return {
        "subset_size": len(subset),
        "mean_pairwise_distance": float(mean_pairwise_distance),
        "boundary_size": int(boundary_size),
        "density": float(density),
        "mean_degree": float(mean_degree),
        "degree_bias": float(degree_bias)
    }


def update_sensor_df(sensors_df, G, sim_id, selected_nodes_history, OV_values=None):
    """
    OV_values: list of overlap values aligned with selected_nodes_history
               (optional, if you track OV after each selection)
    """
    selected_set = set()
    prev_ov = None

    # Precompute centralities once per graph (important for efficiency)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)

    for i, node in enumerate(selected_nodes_history):
        
        # Features of the newly selected node
        b = betweenness.get(node, 0.0)
        pr = pagerank.get(node, 0.0)
        mean_dist = compute_mean_dist(G, node, selected_set)

        # OV gain
        if OV_values is not None:
            ov = OV_values[i]
            ov_gain = ov - prev_ov if prev_ov is not None else ov
            prev_ov = ov
        else:
            ov_gain = np.nan

        # Append row
        sensors_df.loc[len(sensors_df)] = {
            "sim": sim_id,
            "node": node,
            "betweenness": b,
            "pagerank": pr,
            "mean_dist_from_prev": mean_dist,
            "ov_gain": ov_gain
        }

        # Update selected set
        selected_set.add(node)

    return



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



####-------------------- HELPER FUNCTIONS FOR GREEDY SELECTION ----------------------###
def compute_mean_dist(G, node, selected_set):
    if len(selected_set) == 0:
        return 0.0
    
    dists = []
    for s in selected_set:
        try:
            d = nx.shortest_path_length(G, source=node, target=s)
            dists.append(d)
        except nx.NetworkXNoPath:
            dists.append(np.inf)
    
    return float(np.mean(dists))


def update_sensor_df(sensors_df, G, sim_id, selected_nodes_history, OV_values=None):
    """
    OV_values: list of overlap values aligned with selected_nodes_history
               (optional, if you track OV after each selection)
    """
    selected_set = set()
    prev_ov = None

    # Precompute centralities once per graph (important for efficiency)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)

    for i, node in enumerate(selected_nodes_history):
        
        # Features of the newly selected node
        b = betweenness.get(node, 0.0)
        pr = pagerank.get(node, 0.0)
        mean_dist = compute_mean_dist(G, node, selected_set)

        # OV gain
        if OV_values is not None:
            ov = OV_values[i]
            ov_gain = ov - prev_ov if prev_ov is not None else ov
            prev_ov = ov
        else:
            ov_gain = np.nan

        # Append row
        sensors_df.loc[len(sensors_df)] = {
            "sim": sim_id,
            "node": node,
            "betweenness": b,
            "pagerank": pr,
            "mean_dist_from_prev": mean_dist,
            "ov_gain": ov_gain
        }

        # Update selected set
        selected_set.add(node)

    return