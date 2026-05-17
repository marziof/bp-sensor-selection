
import pandas as pd 
from bpepi.Modules import fg_torch as fg #pytorch version
from src.helpers.sim_graph import *
from src.utils.metrics import *
from src.helpers.pipeline_helpers import *
import numpy as np
from collections import defaultdict
import itertools
import copy
from tqdm import tqdm
from src.Analysis.measures import *


### Get probas from marginal

def get_Mt(marginals, t):
    Mt = np.array([1 - marginals[:, 0],marginals[:,0],np.zeros_like(marginals[:,0])])
    return Mt


def compute_measures(marginals, status_nodes):
    T = status_nodes.shape[0] - 1
    M0 = get_Mt(marginals, t=0)
    ov0 = OV(status_nodes[0], np.argmax(M0,axis=0))
    ov0_rnd = OV_rnd(status_nodes[0], M0)
    ov_rescaled = (ov0 - ov0_rnd) / (1 - ov0_rnd)
    mov0 = MOV(M0)
    mov0_rnd = MOV_rnd(M0)
    mov_rescaled = (mov0 - mov0_rnd) / (1 - mov0_rnd)
    ti_str = ti_star(status_nodes, T_BP=T)
    ti_inf = ti_inferred(marginals)
    ti_rnd = ti_random(marginals)
    se = SE(ti_str, ti_inf)
    mse = MSE(marginals, ti_inf)
    measures = {}
    measures["Ov"] = ov0
    measures["MO"] = mov0
    measures["Ov_tilde"] = ov_rescaled
    measures["MO_tilde"] = mov_rescaled
    measures["SE"] = se
    measures["MSE"] = mse
    return measures
        

# ------------------------------------------------------------
# RANDOM BASELINE
# ------------------------------------------------------------

# def compute_bp_estimates(N, T_max, contacts, obs, delta):
#     bp_fg = fg.FactorGraph(N, T_max, contacts, obs, delta)
#     bp_fg.update(maxit=100, tol=1e-5, damp=0.5)
#     marg = bp_fg.marginals()
#     Mt = get_Mt(marg, t=0)
#     x_est = np.argmax(Mt, axis=0)
#     return x_est, Mt, marg

def compute_bp_estimates(bp_fg, obs):
    N = bp_fg.size
    T_max = bp_fg.time
    delta = bp_fg.delta
    contacts = bp_fg.contacts
    bp_fg_eval = fg.FactorGraph(N, T_max, contacts, obs, delta)
    bp_fg_eval.update(maxit=100, tol=1e-5, damp=0.5)
    marg = bp_fg_eval.marginals()
    return marg

#-----------
# BUILD OBSERVATIONS
# -------------------
def build_obs(subset, status_nodes):
    obs_rows = [
        (node, int(status_nodes[t, node]), t)
        for node in subset
        for t in range(status_nodes.shape[0])
    ]
    obs_array = np.array(obs_rows, dtype=int) if len(obs_rows) > 0 else np.empty((0, 3), dtype=int)
    return obs_array


def evaluate_sensors(selected_sensors, bp_fg, status_nodes):

    # selected_sensors = selection_method(metric = metric, bp_base = bp_fg, status_nodes = status_nodes, rho_max = rho, m = m, max_iter = 200, tol = 1e-6, damp = 0.5, delta = delta) #selection_method(bp_fg, G, N, T_max, rho, delta, lam)
    obs_array = build_obs(selected_sensors, status_nodes)
    # Run BP with these observations
    marg = compute_bp_estimates(bp_fg, obs_array)
    # random baseline:
    # compute metrics:
    s0 = status_nodes[0]
    if np.isnan(marg).any():
        print("⚠️ NaN in marginals")
    measures = compute_measures(marginals=marg, status_nodes=status_nodes)
    return {
                "O": measures["Ov"],
                "MO": measures["MO"],
                "O_tilde": measures["Ov_tilde"],
                "MO_tilde": measures["MO_tilde"],
                "SE": measures["SE"],
                "MSE": measures["MSE"]
            }

# def evaluate_sensors(selected_sensors, bp_fg, status_nodes, N, T_max, delta, x_rnd=None, Mt_rnd=None, logger=None, graph=None):

#     # selected_sensors = selection_method(metric = metric, bp_base = bp_fg, status_nodes = status_nodes, rho_max = rho, m = m, max_iter = 200, tol = 1e-6, damp = 0.5, delta = delta) #selection_method(bp_fg, G, N, T_max, rho, delta, lam)
#     obs_array = build_obs(selected_sensors, status_nodes)
#     # Run BP with these observations
#     x_est, Mt, marg = compute_bp_estimates(N, T_max, bp_fg.contacts, obs_array, delta)
#     # random baseline:
#     # compute metrics:
#     s0 = status_nodes[0]
#     if np.isnan(marg).any():
#         print("⚠️ NaN in marginals")
#     measures = compute_measures(marginals=marg, status_nodes=status_nodes, x_rnd=x_rnd, Mt_rnd=Mt_rnd)
#     rank = compute_rank(marg, s0)
#     precision, recall = compute_precision_recall(x_est, s0)
#     f1 = compute_f1(precision, recall)
#     # results_df.loc[len(results_df)] = [method, kind, rho, delta, lam, sim,
#     #             measures["Ov"], measures["MO"], measures["Ov_tilde"], measures["MO_tilde"], measures["SE"], measures["MSE"],
#     #             rank, precision, recall, f1
#     #         ]
#     #print("Overlap:", measures["Ov"], "Overlap tilde:", measures["Ov_tilde"])
#     return {
#                 "O": measures["Ov"],
#                 "MO": measures["MO"],
#                 "O_tilde": measures["Ov_tilde"],
#                 "MO_tilde": measures["MO_tilde"],
#                 "SE": measures["SE"],
#                 "MSE": measures["MSE"],
#                 "rank": rank,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1
#             }
            