from bpepi.Modules import fg_torch as fg #pytorch version
import numpy as np
import networkx as nx
import pandas as pd
from sim_helpers import *
from sensor_selection import *
from metrics import *
from tqdm import tqdm

from greedy_algo import *
#from SensorSelection.Outdated.BO_optimization import *

#################################################################
#--------------------- FULL SIMULATION PIPELINE -----------------
#################################################################

def full_sim(params, Gfixed = False, results_df = None, save=True):

    # initialize results dataframe if not already provided
    if results_df is None:
        results_df = pd.DataFrame(columns=["method", "graph", "rho", "delta", "lambda", "sim", "O", "MO", "O_tilde", "MO_tilde", "SE", "MSE", "rank", "precision", "recall", "f1"])

    if params["track_sensor"]:
        sensors_df = pd.DataFrame(columns=["sim", "method", "graph_kind", "N", "d", "delta", "lam", "rho", "k", "subset_size", "mean_pairwise_distance", "boundary_size", "density", "mean_degree", "degree_bias"])
    else:
        sensors_df = None
        
    for method in params["methods"]:
        # find correct pipeline for given method, and add results to results_df
        print(f"Running method {method}...")
        if method in ["random", "page_rank", "betweenness", "degree", "katz", "eigenvec"]:
            static_bp_sim_pipeline(params["param_list"], params["Nsim"], params["T_max"], params["N"], params["d"], results_df, method, Gfixed=Gfixed, kind=params["graph_kind"])

        elif method in ["greedyOV", "greedyMOV"]:
            greedy_bp_sim_pipeline(params["param_list"], params["Nsim"], params["T_max"], params["N"], params["d"], results_df, method, Gfixed=Gfixed, kind=params["graph_kind"], sensors_df=sensors_df)

        elif method in ["greedySubsetOV", "greedySubsetMOV", "greedySampleReplaceOV", "greedySampleReplaceMOV"]:
            lam = params["param_list"][0][1] # assuming lambda is fixed across param_list
            delta = params["param_list"][0][0] # assuming delta is fixed across param_list
            rho_list = sorted(set([rho for _, _, rho in params["param_list"]])) # get unique rho values from param_list
            greedy_subset_bp_sim_pipeline(lam, delta, rho_list, params["Nsim"], params["T_max"], params["N"], params["d"], results_df, method, kind=params["graph_kind"], sensors_df=sensors_df)
        
        elif method in ["RL"]:
            lam = params["param_list"][0][1] # assuming lambda is fixed across param_list
            delta = params["param_list"][0][0] # assuming delta is fixed across param_list
            rho_list = sorted(set([rho for _, _, rho in params["param_list"]])) # get unique rho values from param_list
            reinforce_bp_sim_pipeline(lam, delta, rho_list, params["Nsim"], params["T_max"], params["N"], params["d"], results_df, method, kind = params["graph_kind"], Gfixed=Gfixed, baseline_decay=0.95, lr=0.1, entropy_coef=0.01)

        else: 
            raise ValueError(f"Unknown method(s) {params['method']}")
    
    # save results
    methods_str = "_".join(params["methods"])
    sim_name = f"methods_{methods_str}_N{params['N']}_d{params['d']}_T{params['T_max']}_Nsim{params['Nsim']}"
    results_df.to_csv(f"results/{sim_name}.csv", index=False)
    print(f"Results saved to results/{sim_name}.csv")
    sensors_df.to_csv(f"results/sensors_{sim_name}.csv", index=False)
    return results_df, sensors_df



#---------------------- Static sensor selection: random or by centrality ----------------------###

def static_bp_sim_pipeline(param_list, Nsim, T_max, N, d, results_df, method, Gfixed = False, kind="rrg"):
    for delta, lam, rho in tqdm(param_list, desc="Simulations"):
        if Gfixed:
            G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
        for sim in range(Nsim):
            # check if results already exist for this combination of parameters
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["rho"] == rho) & (results_df["method"] == method) & (results_df["sim"] == sim)).sum() == Nsim:
                continue

            # 1) Generate dynamic process simulation
            if not Gfixed:
                G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # 2) Generate sensor observations (random and selected)
            obs_selected = gen_selected_sensor_obs(G, rho, status_nodes, method=method)

            # 3) Inference with BP for both random and selected sensors
            bp_fg_rnd = fg.FactorGraph(N,T_max,contacts,[],delta) # for x_rnd: BP with no obs
            bp_fg_rnd.update(maxit=10, print_iter=None)
            marg_rnd = bp_fg_rnd.marginals()
            Mt_rnd = get_Mt(marg_rnd, t=0)
            x_rnd = np.argmax(Mt_rnd, axis=0)

            bp_fg_selected = fg.FactorGraph(N,T_max,contacts,obs_selected,delta)
            it_selected, _ = bp_fg_selected.update(maxit=10, print_iter=None)
            marg_selected = bp_fg_selected.marginals()

            # 4) Compute performance metrics:
            measures_selected, x_est_selected = compute_measures(marg_selected, s0, delta, status_nodes, x_rnd, Mt_rnd)
            rank_selected = compute_rank(marg_selected, s0)
            precision_selected, recall_selected = compute_precision_recall(x_est_selected, s0)
            f1_selected = compute_f1(precision_selected, recall_selected)

            # 5) Store results in results df
            results_df.loc[len(results_df)] = [
                method, kind, rho, delta, lam, sim, measures_selected["Ov"], measures_selected["MO"], measures_selected["Ov_tilde"], measures_selected["MO_tilde"], 
                measures_selected["SE"], measures_selected["MSE"], rank_selected, precision_selected, recall_selected, f1_selected
                ]


#---------------------- Greedy sensor selection: by OV or MOV ----------------------###

def greedy_bp_sim_pipeline(param_list, Nsim, T_max, N, d, results_df, method, kind="rrg", Gfixed=False, sensors_df=None):
    rho_list = np.arange(0, 1.1, 0.1)
    for delta, lam, rho in tqdm(param_list, desc="Simulations"):
        if Gfixed:
            G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
        for sim in range(Nsim):
            # Skip if already computed
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["method"] == method) & (results_df["sim"] == sim)).any():
                continue

            # 1) Generate graph + epidemic
            if not Gfixed:
                G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # 2) Run GREEDY ONCE up to rho=1: if gt: with OV, else with MOV
            gt = s0 if method == "greedyOV" else None
            bp_fg, full_obs, selected_nodes_history, ov_history = run_bp_greedy(
                [], rho_max=1.0, N=N, T=T_max,contacts=contacts, delta=delta,
                status_nodes=status_nodes, max_iter=200, tol=1e-6, damp=0.5, gt=gt, m = int(0.1 * N)
            )
            #bp_fg, current_obs, selected_nodes_history, ov_history =  run_bp_entropy([], G, rho_max=1.0, N=N, T=T_max, contacts=contacts, delta=delta, status_nodes=status_nodes,
            #       max_iter=200, tol=1e-6, damp=0.5, gt=None, beta0=1, new=True)
            if sensors_df is not None:
                #update_sensor_df(sensors_df, G=G, sim_id=sim, selected_nodes_history=selected_nodes_history, OV_values=None)
                for rho_eval in rho_list:
                    k = int(rho_eval * N)
                    subset = selected_nodes_history[:k]

                    features = compute_subset_features(G, subset)

                    row = {
                        "sim": sim,
                        "method": method,
                        "graph_kind": kind,
                        "N": N,
                        "d": d,
                        "delta": delta,
                        "lam": lam,
                        "rho": rho_eval,
                        "k": k
                    }
                    row.update(features)
                    sensors_df.loc[len(sensors_df)] = row

            # 3) Random baseline BP (no observations)
            bp_fg_rnd = fg.FactorGraph(N, T_max, contacts, [], delta)
            marg_rnd = bp_fg_rnd.marginals()
            Mt_rnd = get_Mt(marg_rnd, t=0)
            x_rnd = np.argmax(Mt_rnd, axis=0)

            # 4) Evaluate for each rho
            for rho_eval in rho_list:
                k = int(rho_eval * N)
                if k == 0:
                    current_nodes = []
                else:
                    current_nodes = selected_nodes_history[:k]
                # Build observations for these nodes
                obs_rows = []
                for node in current_nodes:
                    for t in range(T_max + 1):
                        obs_rows.append((node, int(status_nodes[t, node]), t))
                # Convert to array
                if len(obs_rows) > 0:
                    obs_array = np.array(obs_rows, dtype=int)
                else:
                    obs_array = np.empty((0, 3), dtype=int)

                # Run BP with these observations
                bp_fg = fg.FactorGraph(N, T_max, contacts, obs_array, delta)
                bp_fg.update(maxit=200, tol=1e-6, damp=0.5)
                marg = bp_fg.marginals()

                # Metrics
                measures, x_est = compute_measures(marg, s0, delta, status_nodes, x_rnd, Mt_rnd)
                rank = compute_rank(marg, s0)
                precision, recall = compute_precision_recall(x_est, s0)
                f1 = compute_f1(precision, recall)

                # Store results
                results_df.loc[len(results_df)] = [
                method, kind, rho_eval, delta, lam, sim, measures["Ov"], measures["MO"], measures["Ov_tilde"], 
                measures["MO_tilde"], measures["SE"], measures["MSE"], rank, precision, recall, f1
                ]



###---------------------- BO: Greedy sensor subset selection ----------------------###


def greedy_subset_bp_sim_pipeline(lam, delta, rho_list, Nsim, T_max, N, d, results_df, method, kind="rrg", Gfixed = False, print_progress=False, sensors_df=None):
    if method == "greedySubsetOV" or method == "greedySampleReplaceOV":
        gt = True
    else:        
        gt = False
    if method == "greedySampleReplaceOV" or method == "greedySampleReplaceMOV":
        sampleReplace=True
    else:
        sampleReplace=False

    ov_bayes_list = []
    if Gfixed:
        G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
    for rho in rho_list:
        ov_bayes_list = []
        k = int(rho * N)
        for sim in tqdm(range(Nsim)):
            print(f"Bayes-optimal selection, rho={rho}, sim={sim}") if print_progress else None
            # (assumes G, contacts, s0 already fixed outside loop OR regenerate if needed)
            #G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta)
            if not Gfixed:
                G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # --- Bayes-optimal subset (brute force approx) ---
            if sampleReplace:
                best_subset, best_reward, all_overlaps = sampleReplaceSubset(
                    N=N, T=T_max, contacts=contacts, delta=delta, status_nodes=status_nodes,
                    rho=rho, max_iter=20, tol=1e-6, damp=0.5, gt=gt, m=int(0.1*(1-rho)*N)
                )

                # if sim == 5:
                #     # plot all overlaps
                #     sns.lineplot(x=range(len(all_overlaps)), y=all_overlaps)
                #     plt.xlabel("Iteration")
                #     plt.ylabel("Reward (OV or MOV)")
                #     plt.title("Reward Progression during Sample & Replace")
                #     plt.show()
            else:
                best_subset, _ = bayes_optimal_subset(
                    N=N, T=T_max, contacts=contacts, delta=delta, status_nodes=status_nodes,
                    rho=rho, max_iter=20, tol=1e-6, damp=0.5, gt=(method == "greedySubsetOV")
                )

            if sensors_df is not None:
                features = compute_subset_features(G, best_subset)

                row = {
                    "sim": sim,
                    "method": method,
                    "graph_kind": kind,
                    "N": N,
                    "d": d,
                    "delta": delta,
                    "lam": lam,
                    "rho": rho,
                    "k": int(rho * N)
                }
                row.update(features)

                sensors_df.loc[len(sensors_df)] = row

            # --- build observations from selected subset ---
            obs_rows = []
            for node in best_subset:
                for t in range(status_nodes.shape[0]):
                    obs_rows.append((node, int(status_nodes[t, node]), t))

            if len(obs_rows) > 0:
                obs_array = np.array(obs_rows, dtype=int)
            else:
                obs_array = np.empty((0, 3), dtype=int)

            # --- BP inference ---
            bp_fg = fg.FactorGraph(N, T_max, contacts, obs_array, delta)
            bp_fg.update(maxit=20, tol=1e-6, damp=0.5)

            marg = bp_fg.marginals()
            Mt = get_Mt(marg, t=0)

            x_est = np.argmax(Mt, axis=0)
            if gt:
                ov = OV(x_est, status_nodes[0])
            else:
                ov = MOV(Mt) 
            ov_bayes_list.append(ov)

            # 3) Random baseline BP (no observations)
            bp_fg_rnd = fg.FactorGraph(N, T_max, contacts, [], delta)
            marg_rnd = bp_fg_rnd.marginals()
            Mt_rnd = get_Mt(marg_rnd, t=0)
            x_rnd = np.argmax(Mt_rnd, axis=0)

            # Metrics
            measures, x_est = compute_measures(marg, s0, delta, status_nodes, x_rnd, Mt_rnd)
            rank = compute_rank(marg, s0)
            precision, recall = compute_precision_recall(x_est, s0)
            f1 = compute_f1(precision, recall)

            # Store results
            results_df.loc[len(results_df)] = [
            method, kind, rho, delta, lam, sim, measures["Ov"], measures["MO"], measures["Ov_tilde"], 
            measures["MO_tilde"], measures["SE"], measures["MSE"], rank, precision, recall, f1
            ]

        if print_progress:
            print(f"Bayes-opt OVs (rho={rho}):", ov_bayes_list)
            print(f"Average OV (rho={rho}):", np.mean(ov_bayes_list))



### -------------------- BO: learning sensor distribution with REINFORCE ----------------------###


def reinforce_bp_sim_pipeline(lam, delta, rho_list, Nsim, T_max, N, d, results_df, method, kind = "rrg", Gfixed=True, baseline_decay=0.95, lr=0.05, entropy_coef=0.01):

    if Gfixed:
        G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)

    # 1) Train sensor selector for each rho in rho_list (using same graph and epidemic for all rhos)
    selectors = {}
    reward_histories = {}

    for rho in tqdm(rho_list):
        rho = float(rho)
        if not Gfixed:
            G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)

        selector, history = train_sensor_selector(
            G=G, s0=s0, lam=lam, N=N, T=T_max, contacts=contacts, delta=delta,
            rho=rho, lr=lr, baseline_decay=baseline_decay, iterations=100
        )

        selectors[rho] = selector
        reward_histories[rho] = history

    # 2) Evaluate each selector on Nsim new epidemic realizations, and store results in results_df
    for rho in rho_list:
        selector = selectors[rho]
        for sim in tqdm(range(Nsim)):
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["rho"] == rho) & (results_df["method"] == method) & (results_df["sim"] == sim)).any():
                continue
            if not Gfixed:
                G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # Build observations from selector
            probs = selector.get_probs()
            subset = selector.sample_subset(probs)
            #subset = np.argsort(-probs)[:selector.k] # take top-k nodes by probability

            obs_rows = []
            for node in subset:
                for t in range(status_nodes.shape[0]):
                    obs_rows.append((node, int(status_nodes[t, node]), t))

            if len(obs_rows) > 0:
                obs_array = np.array(obs_rows, dtype=int)
            else:
                obs_array = np.empty((0, 3), dtype=int)

            # BP inference
            bp_fg = fg.FactorGraph(N, T_max, contacts, obs_array, delta)
            bp_fg.update(maxit=20, tol=1e-6, damp=0.5)

            marg = bp_fg.marginals()
            Mt = get_Mt(marg, t=0)

            x_est = np.argmax(Mt, axis=0)

            # 3) Random baseline BP (no observations)
            bp_fg_rnd = fg.FactorGraph(N, T_max, contacts, [], delta)
            marg_rnd = bp_fg_rnd.marginals()
            Mt_rnd = get_Mt(marg_rnd, t=0)
            x_rnd = np.argmax(Mt_rnd, axis=0)

            # Metrics
            measures, x_est = compute_measures(marg, s0, delta, status_nodes, x_rnd, Mt_rnd)
            rank = compute_rank(marg, s0)
            precision, recall = compute_precision_recall(x_est, s0)
            f1 = compute_f1(precision, recall)

            # Store results
            results_df.loc[len(results_df)] = [
                method, kind, rho, delta, lam, sim,
                measures["Ov"], measures["MO"], measures["Ov_tilde"], 
                measures["MO_tilde"], measures["SE"], measures["MSE"], rank, precision, recall, f1
            ]