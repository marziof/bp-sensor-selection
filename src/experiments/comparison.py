

import pandas as pd 

from src.utils.bp_experiment_pipeline import make_instance, sim_pipeline

from collections import defaultdict
import copy

# To compare same methods with a different metric
def run_full_comparison(methods, metrics, param_grid, Nsim, N, T_max, d, results_df):
    for method_name, method in methods.items():
        # --- group rhos by (delta, lam)
        param_map = defaultdict(list)
        for delta, lam, rho in param_grid:
            param_map[(delta, lam)].append(rho)

        for sim in range(Nsim):
            print(f"\n=== Simulation {sim} ===")
            # --- FIX GRAPH ---
            G, contacts, s0 = gen_graph_sim(N, d, T_max=T_max)
            for (delta, lam), rhos in param_map.items():
                rhos = sorted(rhos)
                # --- FIX EPIDEMIC ---
                status_nodes = simulate_SI(G, s0, lam, T_max)
                for metric_name, metric in metrics.items():
                    print(f"\nMethod: {method_name} | delta={delta}, lam={lam}")
                    is_sequential = (method_name == "sequential_sensor_selection")
                    # fresh BP object per method
                    bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)
                    if is_sequential:
                        # --- run ONCE with max rho ---
                        rho_max = max(rhos)
                        sensor_list = method(metric=metric, bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho_max,
                                            m=int(0.2 * N), max_iter=200, tol=1e-6, damp=0.5, delta=delta
                        )
                        # ensure ordered
                        sensor_list = list(sensor_list)
                        for rho in rhos:
                            k = int(rho * N)
                            subset = set(sensor_list[:k])
                            # --- evaluate once per subset ---
                            result_base = sim_pipeline_fixed_subset(subset=subset, bp_fg=copy.deepcopy(bp_fg), status_nodes=status_nodes, N=N, T_max=T_max, delta=delta)

                            # --- evaluate all metrics ---
                            for metric_name, metric in metrics.items():
                                result = result_base.copy()
                                result_metric = metric(result_base["marginals"], status_nodes=status_nodes, delta=delta)

                                result.update({
                                    "method": method_name,
                                    "metric": metric_name,
                                    "delta": delta,
                                    "lambda": lam,
                                    "rho": rho,
                                    "sim": sim,
                                    "score": result_metric
                                })

                                results_df.loc[len(results_df)] = result

                    else:
                        # --- standard methods ---
                        for rho in rhos:
                            subset = method(metric=metric, bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho,
                                            m=int(0.2 * N), max_iter=200, tol=1e-6, damp=0.5, delta=delta)

                            result_base = sim_pipeline_fixed_subset(subset=subset, bp_fg=copy.deepcopy(bp_fg), status_nodes=status_nodes, N=N, T_max=T_max, delta=delta)

                            for metric_name, metric in metrics.items():
                                result = result_base.copy()
                                result_metric = metric(result_base["marginals"], status_nodes=status_nodes, delta=delta)

                                result.update({
                                    "method": method_name,
                                    "metric": metric_name,
                                    "delta": delta,
                                    "lambda": lam,
                                    "rho": rho,
                                    "sim": sim,
                                    "score": result_metric
                                })

                                results_df.loc[len(results_df)] = result


def sim_pipeline_fixed_subset(subset, bp_fg, status_nodes, N, T_max, delta):

    obs_array = build_obs(subset, status_nodes)
    x_est, Mt, marg = compute_bp_estimates(N, T_max, bp_fg.contacts, obs_array, delta)
    x_rnd, Mt_rnd, _ = compute_bp_estimates(N, T_max, bp_fg.contacts, [], delta)

    s0 = status_nodes[0]

    measures = compute_measures(marginals=marg, status_nodes=status_nodes, x_rnd=x_rnd, Mt_rnd=Mt_rnd)

    rank = compute_rank(marg, s0)
    precision, recall = compute_precision_recall(x_est, s0)
    f1 = compute_f1(precision, recall)

    return {
        "Ov": measures["Ov"],
        "MO": measures["MO"],
        "rank": rank,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "marginals": marg   # needed for metric evaluation
    }