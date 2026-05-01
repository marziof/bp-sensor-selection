
from bpepi.Modules import fg_torch as fg #pytorch version
from src.helpers.sim_graph import gen_graph_sim, simulate_SI
from src.utils.metrics import *


def make_instance(N, T_max, delta, d, lam, kind="rrg"):
    G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta, kind=kind)
    status_nodes = simulate_SI(G, s0, lam, T_max)
    bp_fg = fg.FactorGraph(N,T_max,contacts,[],delta)
    return bp_fg, status_nodes, G



# ------------------------------------------------------------
# RANDOM BASELINE
# ------------------------------------------------------------
def compute_bp_estimates(N, T_max, contacts, obs, delta):
    bp_fg = fg.FactorGraph(N, T_max, contacts, obs, delta)
    bp_fg.update(maxit=20, tol=1e-6, damp=0.5)
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


def evaluate_subset(selected_sensors, metric, bp_fg, status_nodes, G, N, T_max, rho, m, delta, lam):

    # selected_sensors = selection_method(metric = metric, bp_base = bp_fg, status_nodes = status_nodes, rho_max = rho, m = m, max_iter = 200, tol = 1e-6, damp = 0.5, delta = delta) #selection_method(bp_fg, G, N, T_max, rho, delta, lam)
    obs_array = build_obs(selected_sensors, status_nodes)
    # Run BP with these observations
    x_est, Mt, marg = compute_bp_estimates(N, T_max, bp_fg.contacts, obs_array, delta)
    # random baseline:
    x_rnd, Mt_rnd, _ = compute_bp_estimates(N, T_max, bp_fg.contacts, [], delta)
    # compute metrics:
    s0 = status_nodes[0]
    measures = compute_measures(marginals=marg, status_nodes=status_nodes, x_rnd=x_rnd, Mt_rnd=Mt_rnd)
    rank = compute_rank(marg, s0)
    precision, recall = compute_precision_recall(x_est, s0)
    f1 = compute_f1(precision, recall)
    # results_df.loc[len(results_df)] = [method, kind, rho, delta, lam, sim,
    #             measures["Ov"], measures["MO"], measures["Ov_tilde"], measures["MO_tilde"], measures["SE"], measures["MSE"],
    #             rank, precision, recall, f1
    #         ]
    return {
                "Ov": measures["Ov"],
                "MO": measures["MO"],
                "Ov_tilde": measures["Ov_tilde"],
                "MO_tilde": measures["MO_tilde"],
                "SE": measures["SE"],
                "MSE": measures["MSE"],
                "rank": rank,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }


def full_method_pipeline(selection_method, param_list, Nsim, T_max, N, d, results_df, method, kind="rrg", Gfixed=False):
    print(f"Running full_method_pipeline with method={method}...")
    for delta, lam, rho in tqdm(param_list, desc="Simulations"):
        for sim in range(Nsim):
            # Skip if already computed
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["method"] == method) & (results_df["sim"] == sim)).any():
                continue

            # Run method to select sensors and evaluate with BP and metrics
            sim_pipeline(selection_method, G, N, T_max, rho, m=int(0.2 * N), delta=delta, lam=lam)
            
            return {
                "Ov": measures["Ov"],
                "MO": measures["MO"],
                "rank": rank,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }