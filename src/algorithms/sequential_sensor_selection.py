import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from bpepi.Modules import fg_torch as fg #pytorch version
from src.utils.metrics import *

def get_candidates(remaining, m):
    if len(remaining) <= m:
        return remaining
    else:
        return np.random.choice(remaining, size=m, replace=False)

def sequential_sensor_selection(metric, bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta):
    target = int(rho_max * bp_base.size)
    sensor_set = set()
    sensor_order = []
    current_obs = np.empty((0, 3), dtype=int)  # node, state, time
    # converge initial BP with no sensors
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    saved_messages = bp_base.messages.values.clone()  # save base fixed point for warm-starting candidates
    # compute initial metric with no sensors
    metric_base = metric(bp_base.marginals(), status_nodes=status_nodes, delta=delta)
    overlap_base = OV(np.argmax(get_Mt(bp_base.marginals(), t=0), axis=0), status_nodes[0])
    print(f"Baseline metric with no sensors: {metric_base:.4f}, overlap: {overlap_base:.4f}")

    k = len(sensor_set)
    while k < target:
        # evaluate all candidates to find the one that maximizes the metric gain compared to current sensor set
        remaining = list(set(range(bp_base.size)) - sensor_set)
        candidates = get_candidates(remaining, m)
        #print(f"Evaluating candidates: {len(candidates)} remaining")
        best_candidate = eval_candidates(metric=metric, metric_base=metric_base, candidates=candidates, bp_base=bp_base, saved_messages=saved_messages, status_nodes=status_nodes, current_obs=current_obs, warm_iter=20, tol=tol, damp=damp, delta=delta, k=k)
        # reset current_obs and bp_base to base fixed point for next candidates
        bp_base.messages.values = torch.clone(saved_messages)
        bp_base.reset_obs(current_obs)
        # add best candidate to sensor set and update base BP with new observation
        #new_rows = build_obs({best_node}, status_nodes)
        #current_obs = np.vstack([current_obs, new_rows]) if current_obs.size else new_rows
        if k < 5 or k % 50 == 0:
            print(f"Selected candidate: {best_candidate}")
        prev_nb_sensors = len(sensor_set)
        sensor_set.add(best_candidate)
        if len(sensor_set) <= prev_nb_sensors:
            raise ValueError(
                f"No sensor added this iteration. Size stayed {len(sensor_set)}"
            )
        sensor_order.append(best_candidate)

        # add best candidate's full trajectory to observations
        current_obs = build_obs(sensor_set, status_nodes)
        contacts = bp_base.contacts
        N = bp_base.size
        T = bp_base.time
        bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
        bp_base.update(maxit=max_iter, tol=tol, damp=damp)
        saved_messages = bp_base.messages.values.clone()
        # bp_base.reset_obs(current_obs)
        # bp_base.update(maxit=200, tol=1e-6, damp=0.5)
        saved_messages = bp_base.messages.values.clone()  # update base fixed point for next candidates
        metric_value = metric(bp_base.marginals(), status_nodes=status_nodes, delta=delta)
        metric_base = metric_value  # update base metric for next candidates
        overlap = OV(np.argmax(get_Mt(bp_base.marginals(), t=0), axis=0), status_nodes[0])
        k = len(sensor_set)
        if k < 5 or k % 20 == 0:
            print(f"[Step {k}/{target}] selected sensor {best_candidate}, metric={metric_value:.4f}, overlap={overlap:.4f}, rho={(k)/bp_base.size:.3f}")
    return sensor_order


def update_cand_obs(bp_base, candidate, status_nodes, current_obs):
    candidate_rows = build_obs({candidate}, status_nodes)
    candidate_obs = np.vstack([current_obs, candidate_rows]) if current_obs.size else candidate_rows
    bp_base.reset_obs(candidate_obs)
    return 
    

def eval_candidates(metric, metric_base, candidates, bp_base, saved_messages, status_nodes, current_obs, warm_iter, tol, damp, delta, k):
    best_score = -np.inf
    best_candidate = None
    # ensure candidates is not empty
    if k * bp_base.size >= 0.5:
        damp = 0.7  # increase damping in later stages to help convergence with more sensors
    if len(candidates) == 0:
        raise ValueError("No candidates to evaluate")
    for candidate in tqdm(candidates):
        bp_base.messages.values = torch.clone(saved_messages)
        update_cand_obs(bp_base, candidate, status_nodes, current_obs) # updates bp_base in-place with candidate obs
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
        # if k < 5 or k % 30 == 0:
        #     print(f"convergence: {n_iter} iters, error={errors[1]:.2e}")
        marg = bp_base.marginals()
        # check for NaNs in marginals and handle them
        if np.isnan(marg).any():
            print(f"⚠️ NaN in marginals for candidate {candidate}, replacing with uniform distribution")
        marginals = np.nan_to_num(marg, nan=1.0/marg.shape[1])  # handle NaNs if BP fails to converge
        # check marginals clean now
        if np.isnan(marginals).any():
            print(f"⚠️ Still NaN in marginals for candidate {candidate} after replacement, check BP implementation")
        #score = metric(marginals, status_nodes=status_nodes, delta=delta)
        score = metric(marginals, status_nodes=status_nodes, delta=delta) - metric_base  # gain over current sensor set
        # check that score is a valid number
        if np.isnan(score) or np.isinf(score):
            print(f"⚠️ Invalid score {score} for candidate {candidate}, skipping")
        if score > best_score:
            best_score = score
            best_candidate = candidate
    return best_candidate


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


