# access utils by moving to src directory
# import os
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

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
from src.utils.sensor_logger import SensorLogger

# To compare different methods with a same metric



def run_full_sweep(methods, metrics, deltas, lambdas, rhos, Nsim, N, T_max, d, results_df, graph_type = "rrg", logger=None):

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
                    if logger is not None:
                        logger.set_context(method_name=method_name, metric_name="N/A", delta=delta, lam=lam, sim=sim, graph_type=graph_type)
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
                        if len(selected_sensors) > 0 and logger is not None:
                            logger.log_sensor_stats(selected_sensor=list(selected_sensors)[0], candidates= list(set(range(N)) - set(selected_sensors)), marginals=bp_fg.marginals(), status_nodes=status_nodes, rho=rho, graph=G)
                    # continue to next delta, lam since no metric loop for random method
                    continue

                if method_name == "entropy":
                    rho_max = max(rhos)
                    bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)
                    #logger.set_context(method_name=method_name, metric_name="N/A", delta=delta, lam=lam, sim=sim, graph_type=graph_type)
                    selected_sensors = method(bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho_max, m=None, max_iter=200, tol=1e-4, damp=0.5, delta=delta, logger=None, G=G, alpha=0.5, beta=0.3, gamma=0.5)
                    sensor_list = list(selected_sensors)
                    # ordered list of sensors -> 
                    # now evaluate all rhos for this (delta, lam)
                    for rho in rhos:
                        k = int(rho * N)
                        subset = set(sensor_list[:k])
                        result = evaluate_sensors(selected_sensors=subset, bp_fg=bp_fg, status_nodes=status_nodes, N=N, T_max=T_max, delta=delta, x_rnd=x_rnd, Mt_rnd=Mt_rnd, graph=G)
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
                        #if len(selected_sensors) > 0:
                            #logger.log_sensor_stats(selected_sensor=list(selected_sensors)[0], candidates= list(set(range(N)) - set(selected_sensors)), marginals=bp_fg.marginals(), status_nodes=status_nodes, rho=rho, graph=G)
                    # continue to next delta, lam since no metric loop for random method
                    continue

                for metric_name, metric in metrics.items():
                    print(f"  Running metric: {metric_name}")
                    bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)

                    is_seq = (method_name == "sequential")
                    if is_seq:
                        rho_max = max(rhos)
                        logger.set_context(method_name=method_name, metric_name=metric_name, delta=delta, lam=lam, sim=sim, graph_type=graph_type)
                        sensor_list = method(metric=metric, bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho_max, m=int(0.2 * N), max_iter=200, tol=1e-4, damp=0.5, delta=delta, logger=logger, G=G)  # get ordered list of sensors selected by sequential method up to max rho
                        sensor_list = list(sensor_list)
                        # ordered list of sensors -> 

                        # now evaluate all rhos for this (delta, lam)
                        for rho in rhos:
                            k = int(rho * N)
                            subset = set(sensor_list[:k])
                            result = evaluate_sensors(selected_sensors=subset, bp_fg=bp_fg, status_nodes=status_nodes, N=N, T_max=T_max, delta=delta, x_rnd=x_rnd, Mt_rnd=Mt_rnd, graph=G)
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


