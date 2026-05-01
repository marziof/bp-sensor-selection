import numpy as np
import networkx as nx
from tqdm import tqdm
from bpepi.Modules import fg_torch as fg #pytorch version
from metrics import *


###
# Greedy one-by-one
###
def run_bp_greedy(initial_obs, rho_max, N, T, contacts, delta, status_nodes, max_iter=200, tol=1e-6, damp=0.5, gt=None, print_progress=False, m=None):
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
    print("Entering greedy sensor selection loop...")
    while len(already_selected) < target_n:
        k = len(already_selected)

        if k < 5 or k % 10 == 0:
            print(f"[Step {k}/{target_n}] selecting next sensor")

        if k >= N:
            break

        best_mo = -np.inf
        best_ov = -np.inf
        best_node = None
        best_gain = -np.inf

        # try each node as candidate and compute gain in MO or OV compared to current_obs
        remaining = list(set(range(N)) - already_selected)
        if m is None or m >= len(remaining):
            candidates = remaining
        else:
            candidates = np.random.choice(remaining, size=m, replace=False)

        for candidate in tqdm(candidates, desc=f"Step {len(already_selected)+1}", leave=False):
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

import torch
def run_bp_greedy_warm_start(initial_obs, rho_max, N, T, contacts, delta, status_nodes, 
                              max_iter=200, warm_iter=20, tol=1e-6, damp=0.5, 
                              gt=None, print_progress=False, m=None):
    """
    Greedy sensor selection based on OV or MOV gain, with warm-starting.
    For each candidate, BP is warm-started from the base fixed point (current sensor set),
    rather than rebuilt from scratch. All candidates start from the same base fixed point.
    
    Args:
        initial_obs: initial observations array of shape (n_obs, 3)
        rho_max: target sensor density
        N: number of nodes
        T: time horizon
        contacts: list of contacts
        delta: prior infection probability
        status_nodes: ground truth states, shape (T+1, N)
        max_iter: max BP iterations for base convergence
        warm_iter: max BP iterations for warm-started candidate evaluation
        tol: convergence tolerance
        damp: damping factor
        gt: ground truth at t=0 for OV computation (status_nodes[0]); if None, uses MOV
        print_progress: whether to print progress
        m: if not None, subsample m candidates at each step
    """
    # 1) initialize
    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    target_n = int(rho_max * N)
    selected_nodes_history = []
    ov_history = []
    mov_history = []
    ov_mov_tracker = []  # track (step, best_node, best_ov, best_mov, convergence_error)

    # 2) converge base BP from current_obs
    bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    Mt_base = get_Mt(bp_base.marginals(), t=0)
    mo_before = MOV(Mt_base)
    mo_before_constrained = ov_mimic_metric(Mt_base, delta, N, 0, bp_base.marginals()) #constrained_MOV_weighted(Mt_base, delta, N)

    # baseline OV with no sensors
    x_est_base = np.argmax(Mt_base, axis=0)
    ov_baseline = OV(x_est_base, status_nodes[0])
    print(f"Baseline OV (no sensors): {ov_baseline:.4f}, MOV: {mo_before:.4f}")
    print("Entering greedy warm-start sensor selection loop...")

    # 3) greedy loop
    while len(already_selected) < target_n:
        k = len(already_selected)
        if k < 5 or k % 10 == 0:
            print(f"[Step {k}/{target_n}] selecting next sensor")
        if k >= N:
            break

        # snapshot base fixed point — all candidates start from here
        saved_messages = torch.clone(bp_base.messages.values)

        best_node = None
        best_gain = -np.inf
        best_ov = -np.inf
        best_metric = -np.inf
        best_ov_of_best = None  # OV of the best candidate (fixed bug)

        # candidate set
        remaining = list(set(range(N)) - already_selected)
        if m is not None and m < len(remaining):
            candidates = list(np.random.choice(remaining, size=m, replace=False))
        else:
            candidates = remaining

        for candidate in tqdm(candidates): #, desc=f"Step {k+1}", leave=False):
            # restore base fixed point for this candidate
            bp_base.messages.values = torch.clone(saved_messages)

            # build candidate obs and reset
            candidate_rows = build_obs({candidate}, status_nodes)
            candidate_obs = np.vstack([current_obs, candidate_rows]) if current_obs.size else candidate_rows
            bp_base.reset_obs(candidate_obs)

            # warm-start from base fixed point
            bp_base.update(maxit=warm_iter, tol=tol, damp=damp)

            marg = bp_base.marginals()
            marg = np.nan_to_num(marg, nan=1.0/(T+2))  # guard against nan marginals
            Mt = get_Mt(marg, t=0)
            x_est = np.argmax(Mt, axis=0)
            ov = OV(x_est, status_nodes[0])

            if gt is None:
                mo_after = MOV(Mt)
                mo_after_constrained = ov_mimic_metric(Mt, delta, N, k, marg) #constrained_MOV_weighted(Mt, delta, N)  # MOV with constrained number of predicted infections
                #gain = mo_after_constrained - mo_before_constrained  # compare constrained MOV gain for fair sensor contribution evaluation
                gain = mo_after - mo_before  # original MOV gain for maximum improvement
                if gain > best_gain:
                    best_gain = gain
                    best_metric = mo_after
                    best_node = candidate
                    best_ov_of_best = ov  # OV of best candidate, not last candidate
            else:
                if ov > best_ov:
                    best_ov = ov
                    best_metric = ov
                    best_node = candidate
                    best_ov_of_best = ov

        # check convergence of best candidate specifically
        bp_base.messages.values = torch.clone(saved_messages)
        best_candidate_rows = build_obs({best_node}, status_nodes)
        best_candidate_obs = np.vstack([current_obs, best_candidate_rows]) if current_obs.size else best_candidate_rows
        bp_base.reset_obs(best_candidate_obs)
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=tol, damp=damp)

        if gt is None and k%10 == 0:
            print(f"  → Best candidate {best_node}: MOV={best_metric:.4f}, OV={best_ov_of_best:.4f}, "
                  f"convergence: {n_iter} iters, error={errors[1]:.2e}, "
                  f"rho={len(already_selected)/N:.3f}")
        elif gt is not None and k%5 == 0:
            print(f"  → Best candidate {best_node}: OV={best_metric:.4f}, "
                  f"convergence: {n_iter} iters, error={errors[1]:.2e}, "
                  f"rho={len(already_selected)/N:.3f}")
        else: 
            None

        ov_mov_tracker.append((k, best_node, best_ov_of_best, best_metric, errors[1]))

        # restore base fixed point then add best node permanently
        bp_base.messages.values = torch.clone(saved_messages)
        new_rows = build_obs({best_node}, status_nodes)
        current_obs = np.vstack([current_obs, new_rows]) if current_obs.size else new_rows
        already_selected.add(best_node)
        selected_nodes_history.append(best_node)

        if gt is not None:
            ov_history.append(best_metric)
        else:
            ov_history.append(best_ov_of_best)
        mov_history.append(best_metric if gt is None else MOV(Mt_base))

        if print_progress:
            metric_name = "OV" if gt is not None else "MO"
            #print(f"  → selected node {best_node}, {metric_name}={best_metric:.4f}, "f"rho={len(already_selected)/N:.3f}")

        # reconverge base BP with best node added
        bp_base.reset_obs(current_obs)
        bp_base.update(maxit=max_iter, tol=tol, damp=damp)
        B_base = bp_base.marginals()
        B_base = np.nan_to_num(B_base, nan=1.0/(T+2))
        Mt_base = get_Mt(B_base, t=0)
        mo_before = MOV(Mt_base)
        mo_before_constrained = ov_mimic_metric(Mt_base, delta, N, k, B_base) #constrained_MOV_weighted(Mt_base, delta, N)

    # plot OV, MOV, and convergence error over greedy steps
    import matplotlib.pyplot as plt
    steps = [x[0] for x in ov_mov_tracker]
    ov_vals = [x[2] for x in ov_mov_tracker]
    mov_vals = [x[3] for x in ov_mov_tracker]
    err_vals = [x[4] for x in ov_mov_tracker]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(steps, ov_vals, label='OV')
    axes[0].axhline(y=ov_baseline, color='r', linestyle='--', label=f'baseline OV={ov_baseline:.3f}')
    axes[0].set_xlabel("Greedy step")
    axes[0].set_ylabel("OV")
    axes[0].set_title("OV over greedy steps")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(steps, mov_vals)
    axes[1].set_xlabel("Greedy step")
    axes[1].set_ylabel("MOV")
    axes[1].set_title("MOV over greedy steps")
    axes[1].grid(True)

    axes[2].semilogy(steps, err_vals)
    axes[2].set_xlabel("Greedy step")
    axes[2].set_ylabel("Convergence error")
    axes[2].set_title("BP convergence error at warm-start")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/warmstart_greedy_rhomax{rho_max}.png")
    plt.close()

    # scatter OV vs MOV across all steps
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.scatter(mov_vals, ov_vals, alpha=0.6, c=steps, cmap='viridis')
    ax2.set_xlabel("MOV")
    ax2.set_ylabel("OV")
    ax2.set_title("OV vs MOV across greedy steps")
    ax2.grid(True)
    plt.colorbar(ax2.collections[0], ax=ax2, label='greedy step')
    plt.savefig(f"./results/OV_vs_MOV_warmstart_rhomax{rho_max}.png")
    plt.close()

    return bp_base, current_obs, selected_nodes_history, ov_history, mov_history, ov_mov_tracker



def select_by_entropy_neighbor(H_per_node, remaining, G, already_selected, beta=1.0): # tried for 500
    """
    Select node maximizing entropy of node + mean entropy of its unselected neighbors.
    
    Args:
        H_per_node: array of shape (N,) with binary entropy at t=0 per node
        remaining: list of unselected node indices
        G: networkx graph
        already_selected: set of already selected nodes
        beta: weight for neighbor entropy term
    """
    scores = np.zeros(len(remaining))
    for idx, node in enumerate(remaining):
        unselected_neighbors = [nb for nb in G.neighbors(node) if nb not in already_selected]
        neighbor_entropy = H_per_node[unselected_neighbors].mean() if unselected_neighbors else 0.0
        scores[idx] = H_per_node[node] + beta * neighbor_entropy
    return remaining[np.argmax(scores)]

def select_boundary_node(B, remaining, G, T_mid=None): # tried for 600
    """
    Select node most likely to be at the infection boundary.
    These are nodes uncertain about their state at an intermediate time.
    """
    if T_mid is None:
        T_mid = B.shape[1] // 2  # middle time point
    Mt_mid = get_Mt(B, t=T_mid)  # (2, N)
    p_inf_mid = Mt_mid[1]  # P(infected by T_mid)
    # most uncertain at T_mid = closest to infection boundary
    boundary_score = 1 - np.abs(p_inf_mid - 0.5) * 2  # 1 at p=0.5, 0 at p=0 or 1
    remaining = np.array(remaining)
    return remaining[np.argmax(boundary_score[remaining])]

def select_by_pinf(Mt, remaining): # tried for 350
    """Select node most likely to be infected at t=0."""
    remaining = np.array(remaining)
    p_inf = Mt[1]  # P(I at t=0), shape (N,)
    return remaining[np.argmax(p_inf[remaining])]

def select_early_infected(B, remaining): # try for 450
    """Select node most likely infected earliest."""
    remaining = np.array(remaining)
    # probability of being infected in first few time steps
    p_early = B[:, :3].sum(axis=1)  # P(t* <= 1)
    return remaining[np.argmax(p_early[remaining])]

def select_by_pinf_early_mov(B, Mt, remaining, alpha=0.5):
    """
    Select node balancing early infection probability and MOV contribution.
    
    Args:
        B: (N, T+2) infection time marginals
        Mt: (2, N) state marginals at t=0
        remaining: list of unselected nodes
        alpha: weight for early infection term (1-alpha for MOV term)
    """
    remaining = np.array(remaining)
    
    # early infection score: P(infected in first few steps)
    p_early = B[:, :3].sum(axis=1)  # P(t* <= 1)
    
    # MOV contribution per node: max(P_S, P_I)
    mov_per_node = np.maximum(Mt[0], Mt[1])
    # invert so uncertain nodes score high
    uncertainty = 1 - mov_per_node  
    
    # normalize both to [0,1]
    p_early_norm = (p_early - p_early.min()) / (p_early.max() - p_early.min() + 1e-10)
    uncertainty_norm = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min() + 1e-10)
    
    score = alpha * p_early_norm + (1 - alpha) * uncertainty_norm
    return remaining[np.argmax(score[remaining])]

def run_bp_greedy_entropy(initial_obs, rho_max, G, N, T, contacts, delta, status_nodes,
                          max_iter=200, tol=1e-6, damp=0.5,
                          gt=None, print_progress=False):
    """
    Greedy sensor selection based on binary entropy at t=0.
    At each step, selects the most uncertain node (highest binary entropy at t=0)
    from the current base BP run — no per-candidate BP calls.
    One BP call per greedy step, after committing to the selected node.
    """
    # 1) initialize
    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    target_n = int(rho_max * N)
    selected_nodes_history = []
    ov_history = []
    mov_history = []
    entropy_history = []

    # 2) converge base BP from current_obs
    bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    B = bp_base.marginals()

    print("Entering greedy entropy (t=0) sensor selection loop...")

    # 3) greedy loop
    while len(already_selected) < target_n:
        k = len(already_selected)
        if k < 5 or k % 10 == 0:
            print(f"[Step {k}/{target_n}] selecting next sensor")
        if k >= N:
            break

        # compute binary entropy at t=0 from current base BP
        Mt = get_Mt(B, t=0)  # (2, N)
        p = np.clip(Mt, 1e-10, 1 - 1e-10)
        H_per_node = -np.sum(p * np.log(p), axis=0)  # (N,)

        # track metrics
        mov = MOV(Mt)
        mov_history.append(mov)
        entropy_history.append(np.mean(H_per_node))
        if gt is not None:
            x_est = np.argmax(Mt, axis=0)
            ov = OV(x_est, gt)
            ov_history.append(ov)

        # select most uncertain unselected node
        remaining = list(set(range(N)) - already_selected)
        H_remaining = H_per_node[remaining]
        #best_node = remaining[np.argmax(H_remaining)]
        remaining = np.array(remaining)  # needs to be array for indexing
        #best_node = select_boundary_node(B, remaining, G)  #select_by_entropy_neighbor(H_per_node, remaining, G, already_selected, beta=1.0)
        #best_node = select_by_pinf(Mt, remaining)
        best_node = select_by_pinf_early_mov(B, Mt, remaining, alpha=0.5)

        if print_progress:
            ov_str = f", OV={ov_history[-1]:.4f}" if gt is not None else ""
            print(f"  → selected node {best_node} (H={H_per_node[best_node]:.4f}), "
                  f"MOV={mov:.4f}{ov_str}, "
                  f"rho={len(already_selected)/N:.3f}")

        # commit: add best node
        new_rows = build_obs({best_node}, status_nodes)
        current_obs = np.vstack([current_obs, new_rows]) if current_obs.size else new_rows
        already_selected.add(best_node)
        selected_nodes_history.append(best_node)

        # one BP call after committing
        bp_base.reset_obs(current_obs)
        bp_base.update(maxit=max_iter, tol=tol, damp=damp)
        B = bp_base.marginals()

    # plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(14, 5))

    axes[0].plot(entropy_history)
    axes[0].set_xlabel("Greedy step")
    axes[0].set_ylabel("Mean binary entropy at t=0")
    axes[0].set_title("Entropy over greedy steps")
    axes[0].grid(True)

    axes[1].plot(mov_history)
    axes[1].set_xlabel("Greedy step")
    axes[1].set_ylabel("MOV")
    axes[1].set_title("MOV over greedy steps")
    axes[1].grid(True)

    if gt is not None:
        axes[2].plot(ov_history)
        axes[2].set_xlabel("Greedy step")
        axes[2].set_ylabel("OV")
        axes[2].set_title("OV over greedy steps")
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/entropy_t0_greedy_rhomax{rho_max}.png")
    plt.close()

    return bp_base, current_obs, selected_nodes_history, ov_history, mov_history, entropy_history



def run_bp_greedy_entropy_prev(initial_obs, rho_max, N, T, contacts, delta, status_nodes,
                          max_iter=200, tol=1e-6, damp=0.5,
                          gt=None, print_progress=False):
    """
    Greedy sensor selection based on infection time entropy.
    At each step, selects the most uncertain node (highest entropy over B)
    from the current base BP run — no per-candidate BP calls needed.
    
    Args:
        initial_obs: initial observations array of shape (n_obs, 3)
        rho_max: target sensor density
        N: number of nodes
        T: time horizon
        contacts: list of contacts
        delta: prior infection probability
        status_nodes: ground truth states, shape (T+1, N)
        max_iter: max BP iterations for base convergence
        tol: convergence tolerance
        damp: damping factor
        gt: ground truth at t=0 for OV tracking; if None, skips OV tracking
        print_progress: whether to print progress
    """
    # 1) initialize
    initial_obs = np.array(initial_obs)
    if initial_obs.size == 0:
        already_selected = set()
        current_obs = np.empty((0, 3), dtype=int)
    else:
        already_selected = set(initial_obs[:, 0].astype(int))
        current_obs = initial_obs.copy()

    target_n = int(rho_max * N)
    selected_nodes_history = []
    ov_history = []
    mov_history = []
    entropy_history = []

    # 2) converge base BP from current_obs
    bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    B = bp_base.marginals()  # (N, T+2)

    print("Entering greedy entropy sensor selection loop...")

    # 3) greedy loop
    while len(already_selected) < target_n:
        k = len(already_selected)
        if k < 5 or k % 10 == 0:
            print(f"[Step {k}/{target_n}] selecting next sensor")
        if k >= N:
            break

        # compute infection time entropy for all nodes
        p = np.clip(B, 1e-10, 1 - 1e-10)
        H_per_node = -np.sum(p * np.log(p), axis=1)  # (N,)

        # select most uncertain unselected node
        remaining = list(set(range(N)) - already_selected)
        H_remaining = H_per_node[remaining]
        best_idx = np.argmax(H_remaining)
        best_node = remaining[best_idx]

        # track metrics before adding sensor
        Mt = get_Mt(B, t=0)
        mov = MOV(Mt)
        mov_history.append(mov)
        mean_entropy = np.mean(H_per_node)
        entropy_history.append(mean_entropy)

        if gt is not None:
            x_est = np.argmax(Mt, axis=0)
            ov = OV(x_est, gt)
            ov_history.append(ov)
        
        if print_progress:
            ov_str = f", OV={ov_history[-1]:.4f}" if gt is not None else ""
            print(f"  → selected node {best_node} (H={H_per_node[best_node]:.4f}), "
                  f"MOV={mov:.4f}{ov_str}, mean_H={mean_entropy:.4f}, "
                  f"rho={len(already_selected)/N:.3f}")

        # add best node
        new_rows = build_obs({best_node}, status_nodes)
        current_obs = np.vstack([current_obs, new_rows]) if current_obs.size else new_rows
        already_selected.add(best_node)
        selected_nodes_history.append(best_node)

        # reconverge base BP with best node added
        bp_base.reset_obs(current_obs)
        bp_base.update(maxit=max_iter, tol=tol, damp=damp)
        B = bp_base.marginals()

    # plot entropy and OV over greedy steps
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3 if gt is not None else 2, figsize=(14, 5))
    
    axes[0].plot(entropy_history)
    axes[0].set_xlabel("Greedy step")
    axes[0].set_ylabel("Mean posterior entropy")
    axes[0].set_title("Entropy over greedy steps")
    axes[0].grid(True)

    axes[1].plot(mov_history)
    axes[1].set_xlabel("Greedy step")
    axes[1].set_ylabel("MOV")
    axes[1].set_title("MOV over greedy steps")
    axes[1].grid(True)

    if gt is not None:
        axes[2].plot(ov_history)
        axes[2].set_xlabel("Greedy step")
        axes[2].set_ylabel("OV")
        axes[2].set_title("OV over greedy steps")
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/entropy_greedy_rhomax{rho_max}.png")
    plt.close()

    return bp_base, current_obs, selected_nodes_history, ov_history, mov_history, entropy_history

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
        for sensor in tqdm(list(best_subset), desc="Iterating through sensors"): # tqdm(list(best_subset), desc="Iterating through sensors"):  # iterate over copy since we may modify best_subset
            candidate_subset = set(nodes) - best_subset
            best_candidate = None
            best_candidate_reward = best_reward
            m=5
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


### ------------------- MCMC SAMPLE REPLACE ----------------------###
import numpy as np
import random
from tqdm import tqdm

def sampleReplaceSubset_MCMC(
    N, T, contacts, delta, status_nodes,
    rho,
    max_iter=50,
    tol=1e-6,
    damp=0.5,
    gt=False,
    m=None,
    n_steps=300,
    beta=1.0
):

    k = int(rho * N)
    nodes = list(range(N))

    # ------------------------------------------------------------
    # INITIAL SUBSET
    # ------------------------------------------------------------
    current_subset = set(np.random.choice(nodes, size=k, replace=False))

    # build initial BP state
    obs_rows = [
        (node, int(status_nodes[t, node]), t)
        for node in current_subset
        for t in range(status_nodes.shape[0])
    ]
    obs_array = np.array(obs_rows, dtype=int) if len(obs_rows) > 0 else np.empty((0, 3), dtype=int)

    bp_fg = fg.FactorGraph(N, T, contacts, obs_array, delta)
    bp_fg.update(maxit=max_iter, tol=tol, damp=damp)

    marg = bp_fg.marginals()
    Mt = get_Mt(marg, t=0)

    if gt:
        current_reward = OV(np.argmax(Mt, axis=0), status_nodes[0])
    else:
        current_reward = MOV(Mt)
        OV_reward = OV(np.argmax(Mt, axis=0), status_nodes[0])
        OV_MOV_comparison = [(OV_reward, current_reward)]

    best_subset = current_subset.copy()
    best_reward = current_reward

    history = [current_reward]

    # ------------------------------------------------------------
    # EARLY STOPPING VARIABLES
    # ------------------------------------------------------------
    best_seen = current_reward
    no_improve = 0

    burn_in = 50
    patience = 80
    eps = 1e-4
    window = 40

    # ------------------------------------------------------------
    # MCMC LOOP
    # ------------------------------------------------------------
    for step in tqdm(range(n_steps), desc="MCMC subset search"):

        # --- propose swap ---
        i = random.choice(list(current_subset))
        outside = list(set(nodes) - current_subset)

        if m is not None and len(outside) > m:
            outside = list(np.random.choice(outside, size=m, replace=False))

        j = random.choice(outside)

        proposed_subset = (current_subset - {i}) | {j}

        # --------------------------------------------------------
        # BUILD OBSERVATIONS
        # --------------------------------------------------------
        obs_rows = [
            (node, int(status_nodes[t, node]), t)
            for node in proposed_subset
            for t in range(status_nodes.shape[0])
        ]
        obs_array = np.array(obs_rows, dtype=int) if len(obs_rows) > 0 else np.empty((0, 3), dtype=int)

        # --------------------------------------------------------
        # WARM START BP
        # --------------------------------------------------------
        #bp_prop = bp_fg.copy()
        bp_prop = fg.FactorGraph(N, T, contacts, obs_array, delta)
        bp_prop.update(maxit=max_iter, tol=tol, damp=damp)
        bp_prop.reset_obs(obs_array)
        bp_prop.update(maxit=max_iter, tol=tol, damp=damp)

        marg = bp_prop.marginals()
        Mt = get_Mt(marg, t=0)

        if gt:
            proposed_reward = OV(np.argmax(Mt, axis=0), status_nodes[0])
        else:
            proposed_reward = MOV(Mt)
            OV_proposed_reward = OV(np.argmax(Mt, axis=0), status_nodes[0])
            OV_MOV_comparison.append((OV_proposed_reward, proposed_reward))

        # --------------------------------------------------------
        # METROPOLIS ACCEPT/REJECT
        # --------------------------------------------------------
        delta_E = proposed_reward - current_reward

        if delta_E > 0 or np.random.rand() < np.exp(beta * delta_E):
            current_subset = proposed_subset
            bp_fg = bp_prop
            current_reward = proposed_reward

            if current_reward > best_reward:
                best_reward = current_reward
                best_subset = current_subset.copy()

        history.append(current_reward)

        # --------------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------------
        if current_reward > best_seen + 1e-6:
            best_seen = current_reward
            no_improve = 0
        else:
            no_improve += 1

        # stagnation stop
        if step > burn_in and no_improve >= patience:
            break

        # stability stop
        if step > burn_in and len(history) >= window:
            if np.std(history[-window:]) < eps:
                break

    #return best_subset, best_reward, history
    # print comparison of OV vs MOV during MCMC search
    if not gt:
        # plot OV vs MOV during MCMC search
        import matplotlib.pyplot as plt
        OV_vals, MOV_vals = zip(*OV_MOV_comparison)
        plt.figure(figsize=(6, 6))
        plt.scatter(MOV_vals, OV_vals, alpha=0.6)
        plt.xlabel("MOV")
        plt.ylabel("OV")
        plt.title("OV vs MOV during MCMC search")
        plt.grid(True)
        plt.savefig("./results/OV_vs_MOV_MCMC_search.png")
        plt.close()
    return best_subset, best_reward, history, bp_fg

###------------------- GREEDY SENSOR SELECTION ALGORITHM: 1 sensor at a time chosen greedily ----------------------###

def run_bp_greedy_test(initial_obs, rho_max, N, T, contacts, delta, status_nodes, max_iter=200, tol=1e-6, damp=0.5, gt=None, print_progress=False, m=None):
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
    print("Entering greedy sensor selection loop...")
    while len(already_selected) < target_n:
        k = len(already_selected)

        if k < 5 or k % 10 == 0:
            print(f"[Step {k}/{target_n}] selecting next sensor")

        if k >= N:
            break

        best_mo = -np.inf
        best_ov = -np.inf
        best_node = None
        best_gain = -np.inf

        # try each node as candidate and compute gain in MO or OV compared to current_obs
        remaining = list(set(range(N)) - already_selected)
        if m is None or m >= len(remaining):
            candidates = remaining
        else:
            candidates = np.random.choice(remaining, size=m, replace=False)

        for candidate in tqdm(candidates, desc=f"Step {len(already_selected)+1}", leave=False):
            if candidate in already_selected:
                continue
            candidate_rows = [(candidate, int(status_nodes[t, candidate]), t) for t in range(status_nodes.shape[0])]
            candidate_obs = np.vstack([current_obs, np.array(candidate_rows, dtype=int)])

            # bp with candidate_obs
            bp_cand = bp_fg.copy()  # start from current bp factor graph to save time
            bp_cand.reset_observations(candidate_obs)  # update with candidate observations
            bp_cand.update(maxit=0.5*max_iter, tol=tol, damp=damp)
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