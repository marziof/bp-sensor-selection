from bpepi.Modules import fg_torch as fg #pytorch version
import numpy as np
import networkx as nx

def gen_selected_sensor_obs(G, rho, status_nodes, method="betweenness"):
    """
    Receives: rho: fraction of sensors, status_nodes: matrix (T_max+1, N) of node states over time
    Returns: obs, array of observations of chosen sensors (list (i, 0/1, t))
    """
    T_plus1, N = status_nodes.shape
    n_sensors = int(rho * N)
    # choose sensors based on highest betweenness centrality
    if method == "random":
        sensor_indices = np.random.choice(N, size=n_sensors, replace=False)
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif method == "degree":
        centrality = nx.degree_centrality(G)
    elif method == "eigenvector":
        centrality = nx.eigenvector_centrality(G)
    elif method == "katz":
        centrality = nx.katz_centrality(G)
    elif method == "page_rank":
        centrality = nx.pagerank(G)
    else:       
        raise ValueError(f"Unknown method: {method}")
    if method != "random":
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
        sensor_indices = sorted_nodes[:n_sensors]
    obs = []
    for i in sensor_indices:
        for t in range(T_plus1):
            state = int(status_nodes[t, i])
            obs.append((i, state, t))
    return obs


def gen_selected_sensor_obs_div(G, rho, status_nodes, method="betweenness", alpha=0.7):
    T_plus1, N = status_nodes.shape
    n_sensors = int(rho * N)
    nodes = list(G.nodes())

    # --- centrality ---
    if method == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif method == "degree":
        centrality = nx.degree_centrality(G)
    elif method == "eigenvector":
        centrality = nx.eigenvector_centrality(G)
    elif method == "katz":
        centrality = nx.katz_centrality(G)
    elif method == "page_rank":
        centrality = nx.pagerank(G)
    else:
        raise ValueError(f"Unknown method: {method}")

    # normalize centrality
    c_vals = np.array([centrality[i] for i in nodes])
    c_vals = (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min() + 1e-10)
    c_dict = {i: c_vals[k] for k, i in enumerate(nodes)}

    # shortest path distances
    dist = dict(nx.all_pairs_shortest_path_length(G))

    selected = []

    # first pick: highest centrality
    first = max(nodes, key=lambda i: c_dict[i])
    selected.append(first)

    # greedy selection
    while len(selected) < n_sensors:
        best_node = None
        best_score = -1

        for i in nodes:
            if i in selected:
                continue

            # distance to closest selected node
            d = min(dist[i][j] for j in selected)

            # combine centrality + diversity
            score = alpha * c_dict[i] + (1 - alpha) * (d / len(G))

            if score > best_score:
                best_score = score
                best_node = i

        selected.append(best_node)

    sensor_indices = selected

    obs = []
    for i in sensor_indices:
        for t in range(T_plus1):
            state = int(status_nodes[t, i])
            obs.append((i, state, t))

    return obs


## OLD:
def converge_bp(bp_fg, max_iter=10, damp=0.5):
    for i in range(max_iter):
        bp_fg.iterate(damp=damp)
    return i + 1

def run_bp_dynamic(bp_fg, initial_obs, rho_max, N, status_nodes, max_iter=10, damp=0.5):
    initial_obs = np.array(initial_obs)
    if rho_max == 0 or len(initial_obs) == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()
    
    bp_fg.reset_obs(current_obs)
    converge_bp(bp_fg, max_iter, damp)
    
    while len(already_selected) / N < rho_max:
        marg = bp_fg.marginals()
        Mt = get_Mt(marg, t=0)
        
        confidence = np.maximum(Mt[0], Mt[1])
        confidence[list(already_selected)] = np.inf
        # take k sensors with lowest confidence
        k=0.1*rho_max*N
        selected = np.argmin(confidence)
        already_selected.add(selected)
        
        new_obs = np.array([
            (selected, int(status_nodes[t, selected]), t)
            for t in range(status_nodes.shape[0])
        ])
        current_obs = np.vstack([current_obs, new_obs])
        
        bp_fg.reset_obs(current_obs)
        n_iter = converge_bp(bp_fg, max_iter, damp)
        #print(f"rho={len(already_selected)/N:.2f}, selected={selected}, converged in {n_iter} iters")

    ####
def converge_bp(bp_fg, max_iter=10, damp=0.5):
    for i in range(max_iter):
        bp_fg.iterate(damp=damp)
    return i + 1


def run_bp_dynamic(bp_fg, initial_obs, rho_max, N, status_nodes,
                   max_iter=10, damp=0.5, k_frac=0.1, obs_time=None):
    """
    Args:
        bp_fg: FactorGraph instance
        initial_obs: array of shape (M, 3) with columns [node, state, time], or empty
        rho_max: target fraction of nodes to select as sensors
        N: total number of nodes
        status_nodes: array of shape (T, N) with node states at each timestep
        max_iter: BP iterations per round
        damp: BP damping
        k_frac: fraction of rho_max*N to select per round (batch size)
        obs_time: which timestep to observe when adding a sensor (default: last)
    """
    if obs_time is None:
        obs_time = status_nodes.shape[0] - 1

    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    bp_fg.reset_obs(current_obs)
    converge_bp(bp_fg, max_iter, damp)

    target_n = int(rho_max * N)
    k = max(1, int(k_frac * target_n))  # batch size, at least 1

    while len(already_selected) < target_n:
        # guard: if all nodes are selected, stop
        if len(already_selected) >= N:
            break

        marg = bp_fg.marginals()          # shape (N, T+2)
        Mt = get_Mt(marg, t=0)            # (p_S, p_I), each shape (N,)

        # confidence = how certain we are; select nodes where we're LEAST certain
        confidence = np.maximum(Mt[0], Mt[1])  # shape (N,)
        confidence[list(already_selected)] = np.inf  # exclude already selected

        # pick k nodes with lowest confidence (most uncertain)
        n_to_select = min(k, target_n - len(already_selected), N - len(already_selected))
        selected_nodes = np.argpartition(confidence, n_to_select)[:n_to_select]

        new_rows = []
        for selected in selected_nodes:
            already_selected.add(int(selected))
            new_rows.append((int(selected), int(status_nodes[obs_time, selected]), obs_time))

        new_obs = np.array(new_rows, dtype=int)
        current_obs = np.vstack([current_obs, new_obs])

        bp_fg.reset_obs(current_obs)
        converge_bp(bp_fg, max_iter, damp)

    return current_obs

####

def run_bp_dynamic_resets2(initial_obs, rho_max, N, T, contacts, delta, status_nodes,
                   max_iter=200, tol=1e-6, damp=0.5, k_frac=0.1):

    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

    target_n = int(rho_max * N)
    k = max(1, int(k_frac * target_n))

    while len(already_selected) < target_n:
        if len(already_selected) >= N:
            break

        marg = bp_fg.marginals()  # shape (N, T+2)

        # entropy over infection time marginal for each node
        entropy = -np.sum(marg * np.log(marg + 1e-20), axis=1)  # shape (N,)
        entropy[list(already_selected)] = -np.inf

        n_to_select = min(k, target_n - len(already_selected), N - len(already_selected))
        selected_nodes = np.argpartition(entropy, -n_to_select)[-n_to_select:]

        new_rows = [
            (int(s), int(status_nodes[t, s]), t)
            for s in selected_nodes
            for t in range(status_nodes.shape[0])
        ]
        already_selected.update(int(s) for s in selected_nodes)
        current_obs = np.vstack([current_obs, np.array(new_rows, dtype=int)])

        bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
        bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

    return bp_fg, current_obs

def run_bp_greedy_cheat(initial_obs, rho_max, N, T, contacts, delta, status_nodes, s0,
                        max_iter=200, tol=1e-6, damp=0.5):

    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

    target_n = int(rho_max * N)

    while len(already_selected) < target_n:
        if len(already_selected) >= N:
            break

        best_ov = -np.inf
        best_node = None

        for candidate in range(N):
            if candidate in already_selected:
                continue

            candidate_rows = [
                (candidate, int(status_nodes[t, candidate]), t)
                for t in range(status_nodes.shape[0])
            ]
            candidate_obs = np.vstack([current_obs, np.array(candidate_rows, dtype=int)])

            bp_cand = fg.FactorGraph(N, T, contacts, candidate_obs, delta)
            bp_cand.update(maxit=max_iter, tol=tol, damp=damp)

            marg = bp_cand.marginals()
            Mt = get_Mt(marg, t=0)
            x_est = np.argmax(Mt, axis=0)
            ov = OV(x_est, s0)

            if ov > best_ov:
                best_ov = ov
                best_node = candidate

        new_rows = [
            (best_node, int(status_nodes[t, best_node]), t)
            for t in range(status_nodes.shape[0])
        ]
        already_selected.add(best_node)
        current_obs = np.vstack([current_obs, np.array(new_rows, dtype=int)])
        print(f"selected {best_node}, OV={best_ov:.4f}, rho={len(already_selected)/N:.3f}")

        bp_fg = fg.FactorGraph(N, T, contacts, current_obs, delta)
        bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

    return bp_fg, current_obs


    ###

def max_mov_selection_prev(marginals, status_nodes):
    """
       Receives: marginals, status_nodes (shape (T_max+1, N))
       Returns: sensor obs to maximize mov (assuming ground truth is unknown)
    """
    T_plus1, N = status_nodes.shape
    Mt = get_Mt(marginals, t=0) # shape (2, N)
    selected = np.argmax(Mt[1]) # select node with highest P(x_i=1)
    for t in range(T_plus1):
        state = int(status_nodes[t, selected])
        obs = (selected, state, t)  
    return obs

def max_mov_selection(marginals, status_nodes, already_selected=None):
    Mt = get_Mt(marginals, t=0)  # (2, N)
    # entropy-based selection (proxy for MO)
    eps = 1e-12
    print("N:", marginals.shape[1])
    print("already_selected:", already_selected)
    entropy = -np.sum(Mt * np.log(Mt + eps), axis=0)
    # select node with highest entropy that hasnt been observed yet
    entropy_masked = entropy.copy()
    entropy_masked[list(already_selected)] = -np.inf
    selected = np.argmax(entropy_masked)
    already_selected.add(selected)
    T_plus1, N = status_nodes.shape
    obs = []
    for t in range(T_plus1):
        state = int(status_nodes[t, selected])
        obs.append((selected, state, t))
    return np.array(obs)

def run_bp(bp_fg, initial_obs, rho_max, N, status_nodes):
    rho = 0 # initial sensor density
    already_selected = set([obs[0] for obs in initial_obs])
    while rho < rho_max:
        for _ in range(5): # number of BP iterations before selecting new sensors
            bp_fg.iterate(damp=0.5)
        marg = bp_fg.marginals()
        obs_selected = max_mov_selection(marg, status_nodes, already_selected)
        updated_obs = np.vstack([initial_obs, obs_selected])
        bp_fg.reset_obs(updated_obs)
        rho += len(obs_selected) / N
    return