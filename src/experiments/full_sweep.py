# access utils by moving to src directory
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd 
from bpepi.Modules import fg_torch as fg #pytorch version
from src.helpers.sim_graph import *
from src.utils.metrics import *
import numpy as np
from collections import defaultdict
import itertools
import copy
from tqdm import tqdm

# To compare different methods with a same metric


# ------------------------------------------------------------
# RANDOM BASELINE
# ------------------------------------------------------------
def compute_bp_estimates(N, T_max, contacts, obs, delta):
    bp_fg = fg.FactorGraph(N, T_max, contacts, obs, delta)
    bp_fg.update(maxit=100, tol=1e-5, damp=0.5)
    marg = bp_fg.marginals()
    Mt = get_Mt(marg, t=0)
    x_est = np.argmax(Mt, axis=0)
    return x_est, Mt, marg

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


def evaluate_sensors(selected_sensors, bp_fg, status_nodes, N, T_max, delta, x_rnd=None, Mt_rnd=None):

    # selected_sensors = selection_method(metric = metric, bp_base = bp_fg, status_nodes = status_nodes, rho_max = rho, m = m, max_iter = 200, tol = 1e-6, damp = 0.5, delta = delta) #selection_method(bp_fg, G, N, T_max, rho, delta, lam)
    obs_array = build_obs(selected_sensors, status_nodes)
    # Run BP with these observations
    x_est, Mt, marg = compute_bp_estimates(N, T_max, bp_fg.contacts, obs_array, delta)
    # random baseline:
    # compute metrics:
    s0 = status_nodes[0]
    if np.isnan(marg).any():
        print("⚠️ NaN in marginals")
    measures = compute_measures(marginals=marg, status_nodes=status_nodes, x_rnd=x_rnd, Mt_rnd=Mt_rnd)
    rank = compute_rank(marg, s0)
    precision, recall = compute_precision_recall(x_est, s0)
    f1 = compute_f1(precision, recall)
    # results_df.loc[len(results_df)] = [method, kind, rho, delta, lam, sim,
    #             measures["Ov"], measures["MO"], measures["Ov_tilde"], measures["MO_tilde"], measures["SE"], measures["MSE"],
    #             rank, precision, recall, f1
    #         ]
    #print("Overlap:", measures["Ov"], "Overlap tilde:", measures["Ov_tilde"])
    return {
                "O": measures["Ov"],
                "MO": measures["MO"],
                "O_tilde": measures["Ov_tilde"],
                "MO_tilde": measures["MO_tilde"],
                "SE": measures["SE"],
                "MSE": measures["MSE"],
                "rank": rank,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            

def run_full_sweep(methods, metrics, deltas, lambdas, rhos, Nsim, N, T_max, d, results_df, graph_type = "rrg"):

    for method_name, method in methods.items():
        print(f"Running method: {method_name}")

        for sim in tqdm(range(Nsim)):
            print(f"\n=== Sim {sim} ===")
            for delta, lam in itertools.product(deltas, lambdas):
                G, contacts, s0 = gen_graph_sim(N, d=d, lam=lam, T_max=T_max, delta=delta, kind=graph_type)
                # print fraction of initially infected nodes
                print(f"Initial infection fraction (delta): {s0.sum()/N:.3f}, lambda: {lam}")
                status_nodes = simulate_SI(G, s0, lam, T_max)
                # rnd baseline for metrics:
                x_rnd, Mt_rnd, _ = compute_bp_estimates(N, T_max, contacts, [], delta)
                rnd_overlap = OV(x_rnd, s0)
                print(f"Random baseline overlap: {rnd_overlap:.4f}")

                # if method is random, we can directly loop over rhos - no metrics needed for selection
                if method_name == "random":
                    bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)
                    for rho in rhos:
                        selected_sensors = method(bp_base = bp_fg, rho_max=rho, m=None)
                        result = evaluate_sensors(selected_sensors, bp_fg, status_nodes, N, T_max, delta, x_rnd=x_rnd, Mt_rnd=Mt_rnd)
                        result.update({
                            "method": method_name,
                            "metric": "N/A",
                            "delta": delta,
                            "lambda": lam,
                            "rho": rho,
                            "sim": sim,
                            "graph": graph_type
                        })
                        results_df.loc[len(results_df)] = result
                    # continue to next delta, lam since no metric loop for random method
                    continue

                for metric_name, metric in metrics.items():
                    print(f"  Running metric: {metric_name}")
                    bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)

                    is_seq = (method_name == "sequential")
                    if is_seq:
                        rho_max = max(rhos)
                        sensor_list = method(metric=metric, bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho_max, m=int(0.2 * N), max_iter=200, tol=1e-4, damp=0.5, delta=delta)
                        sensor_list = list(sensor_list)
                        # ordered list of sensors -> 

                        # now evaluate all rhos for this (delta, lam)
                        for rho in rhos:
                            k = int(rho * N)
                            subset = set(sensor_list[:k])
                            result = evaluate_sensors(selected_sensors=subset, bp_fg=bp_fg, status_nodes=status_nodes, N=N, T_max=T_max, delta=delta, x_rnd=x_rnd, Mt_rnd=Mt_rnd)
                            result.update({
                                "method": method_name,
                                "metric": metric_name,
                                "delta": delta,
                                "lambda": lam,
                                "rho": rho,
                                "sim": sim,
                                "graph": graph_type
                            })
                            results_df.loc[len(results_df)] = result
                        #print(f"  rho={rho} | O: {result['O']:.4f}, O_tilde: {result['O_tilde']:.4f}")

                    else:
                        # standard methods: rho loop normal
                        for rho in rhos:
                            selected_sensors = method(metric=metric, bp_base=bp_fg, rho_max=rho)
                            result = evaluate_sensors(selected_sensors, bp_fg, status_nodes, N, T_max, delta, x_rnd, Mt_rnd)
                            result.update({
                                "method": method_name,
                                "metric": metric_name,
                                "delta": delta,
                                "lambda": lam,
                                "rho": rho,
                                "sim": sim,
                                "graph": graph_type
                            })

                            results_df.loc[len(results_df)] = result
                            #print(f"  rho={rho} | O: {result['O']:.4f}, O_tilde: {result['O_tilde']:.4f}")

# def run_full_sweep(methods, metrics, param_grid, Nsim, N, T_max, d, results_df):
#     for method_name, method in methods.items():
#         print(f"Running method: {method_name}")
#         for delta, lam, rho in param_grid:

#                 # if ((results_df["delta"] == delta) &
#                 #     (results_df["lambda"] == lam) &
#                 #     (results_df["method"] == method_name) &
#                 #     (results_df["sim"] == sim)).any():
#                 #     continue

#                 bp_fg, status_nodes, G = make_instance(N, T_max, delta, d, lam)

#                 for metric_name, metric in metrics.items():
#                     print(f"  Running metric: {metric_name}")
#                     result = sim_pipeline(selection_method = method, metric=metric, bp_fg=bp_fg, status_nodes=status_nodes, G=G, N=N, T_max=T_max, rho=rho, m=int(0.2 * N), delta=delta, lam=lam)

#                     result.update({
#                         "method": method_name,
#                         "metric": metric_name,
#                         "delta": delta,
#                         "lambda": lam,
#                         "rho": rho,
#                         "sim": sim
#                     })
#                     #results_df = pd.DataFrame(columns=["method", "metric", "graph", "rho", "delta", "lambda", "sim", "O", "MO", "O_tilde", "MO_tilde", "SE", "MSE", "rank", "precision", "recall", "f1"])

#                     results_df.loc[len(results_df)] = result

#     # save check point
#     # methods_str = "_".join(methods.keys())
#     # save_name = f"full_sweep_{methods_str}_N{N}_T{T_max}_d{d}_Nsim{Nsim}.csv"
#     # save_results(results_df, params={"methods": list(methods.keys()), "N": N, "d": d, "T_max": T_max, "Nsim": Nsim}, suffix=save_name)
#     # print(f"Full sweep completed. Results saved to {save_name}")

