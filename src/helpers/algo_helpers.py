import numpy as np
import networkx as nx
from src.utils.metrics import *
from src.helpers.pipeline_helpers import get_Mt
from collections import defaultdict


def build_obs(subset, status_nodes, T_max=None):
    """
    Builds sparse event-based observations for a subset of nodes.
    Matches the logic of generate_sensors_obs exactly:
      - Never infected        → (i, 0, T_max)
      - Infected at t_inf     → (i, 1, t_inf) + optionally (i, 0, t_inf-1)
      - Still infected at end → (i, 1, T_max)
      - Recovered at t_rec    → (i, 1, t_rec-1) + (i, 2, t_rec)
    """
    if T_max is None:
        T_max = status_nodes.shape[0] - 1

    obs_rows = []

    for node in subset:
        if node is None:
            continue
        t_inf_arr = np.where(status_nodes[:T_max+1, node] == 1)[0]
        t_rec_arr = np.where(status_nodes[:T_max+1, node] == 2)[0]
        if len(t_inf_arr) == 0:
            obs_rows.append((node, 0, T_max))
        else:
            t_inf = t_inf_arr[0]
            obs_rows.append((node, 1, t_inf))
            if t_inf > 0:
                obs_rows.append((node, 0, t_inf - 1))  # susceptible just before
            # Still infected at end (no recovery) — this was missing
            if len(t_rec_arr) == 0 and t_inf != T_max:
                obs_rows.append((node, 1, T_max))
            if len(t_rec_arr) > 0:
                t_rec = t_rec_arr[0]
                obs_rows.append((node, 1, t_rec - 1))
                obs_rows.append((node, 2, t_rec))

    return np.array(obs_rows, dtype=int) if obs_rows else np.empty((0, 3), dtype=int)


def update_cand_obs(bp_base, candidate, status_nodes, current_obs, T_max=None):
    """
    Adds observations for a new candidate node and updates BP.
    The observations generated are identical to if the candidate
    had been included in the original observed set from the start.
    """
    if T_max is None:
        T_max = status_nodes.shape[0] - 1
    candidate_rows = build_obs({candidate}, status_nodes, T_max=T_max)
    if current_obs.size:
        candidate_obs = np.vstack([current_obs, candidate_rows])
    else:
        candidate_obs = candidate_rows
    # Sort by time, matching generate_sensors_obs
    candidate_obs = candidate_obs[candidate_obs[:, 2].argsort()]
    bp_base.reset_obs(candidate_obs)
    return candidate_obs  # return so caller can update current_obs