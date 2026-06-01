import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from bpepi.Modules import fg_torch as fg #pytorch version
from src.utils.metrics import *
from src.utils.sensor_logger import SensorLogger, PInfLogger
from src.helpers.algo_helpers import update_cand_obs, build_obs


def get_candidates(remaining, m):
    if len(remaining) <= m:
        return remaining
    else:
        return np.random.choice(remaining, size=m, replace=False)

def sequential_sensor_selection(metric, bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta, logger=None, PinfLogger=None, G=None):
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
        if k < 5 or k % 50 == 0:
            print(f"Selected candidate: {best_candidate}")
        prev_nb_sensors = len(sensor_set)
        sensor_set.add(best_candidate)
        if len(sensor_set) <= prev_nb_sensors:
            raise ValueError(
                f"No sensor added this iteration. Size stayed {len(sensor_set)}"
            )
        sensor_order.append(best_candidate)
        if logger is not None:
            #print(f"Logging stats for selected sensor {best_candidate} at rho={(k+1)/bp_base.size:.3f}")
            logger.log_sensor_stats(selected_sensor=best_candidate, candidates=candidates, marginals=bp_base.marginals(), status_nodes=status_nodes, graph=G, rho=k/bp_base.size)  # log stats before updating BP with new sensor

        if PinfLogger is not None and len(candidates) > 0:
            infected_candidates = [c for c in candidates if int(status_nodes[0, c]) == 1]
            # log for a candidate with infected state for comparison
            if len(infected_candidates) > 0:
                inf_candidate = np.random.choice(infected_candidates, size=1)[0]
                inf_selected_state = int(status_nodes[0, inf_candidate])
                update_cand_obs(bp_base, inf_candidate, status_nodes, current_obs, T_max=bp_base.time)
                bp_base.update(maxit=20, tol=0.1*tol, damp=damp)
                PinfLogger.log_pinf_distribution(selected_state=inf_selected_state, marginals=bp_base.marginals(), status_nodes=status_nodes, graph=G, rho=k/bp_base.size)
                bp_base.reset_obs(current_obs)  # reset BP to current sensor set before next candidates
                bp_base.messages.values = torch.clone(saved_messages)

        # add best candidate's full trajectory to observations
        current_obs = update_cand_obs(bp_base, best_candidate, status_nodes, current_obs, T_max=bp_base.time)  # updates bp_base in-place with new candidate obs
        warm_iter=50
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=0.1*tol, damp=damp)
        marg = bp_base.marginals()

        if PinfLogger is not None:
            selected_state = int(status_nodes[0, best_candidate])
            PinfLogger.log_pinf_distribution(selected_state=selected_state, marginals=bp_base.marginals(), status_nodes=status_nodes, graph=G, rho=k/bp_base.size)

        saved_messages = bp_base.messages.values.clone()  # update base fixed point for next candidates
        metric_value = metric(marg, status_nodes=status_nodes, delta=delta)
        metric_base = metric_value  # update base metric for next candidates
        overlap = OV(np.argmax(get_Mt(marg, t=0), axis=0), status_nodes[0])
        k = len(sensor_set)
        if k < 5 or k % 50 == 0:
            print(f"[Step {k}/{target}] selected sensor {best_candidate}, metric={metric_value:.4f}, overlap={overlap:.4f}, rho={(k)/bp_base.size:.3f}")
            print(f"  Errors during BP convergence: {errors}, in iters: {n_iter}")
    return sensor_order
    

def eval_candidates(metric, metric_base, candidates, bp_base, saved_messages, status_nodes, current_obs, warm_iter, tol, damp, delta, k):
    best_score = -np.inf
    best_candidate = None
    # ensure candidates is not empty
    if k / bp_base.size >= 0.5:
        damp = damp*2  # increase damping in later stages to help convergence with more sensors
    if len(candidates) == 0:
        raise ValueError("No candidates to evaluate")
    for candidate in candidates:
        bp_base.messages.values = torch.clone(saved_messages)
        update_cand_obs(bp_base, candidate, status_nodes, current_obs, T_max=bp_base.time) # updates bp_base in-place with candidate obs
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

