

def bp_sim_pipeline(param_list, Nsim, T_max, N, d, results_df, method, div=False, alpha=0.7, with_random=True):
    for delta, lam, rho in tqdm(param_list, desc="Simulations"):
        for sim in range(Nsim):
            # check if results already exist for this combination of parameters
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["rho"] == rho) & (results_df["sensor_type"] == method) & (results_df["sim"] == sim)).sum() == Nsim:
                continue

            # 1) Generate dynamic process simulation
            G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # 2) Generate sensor observations (random and selected)
            obs_random = gen_selected_sensor_obs(G, rho, status_nodes, method="random") if with_random else None
            #obs_selected = gen_selected_sensor_obs(G, rho, status_nodes, method=method)
            if div:
                obs_selected = gen_selected_sensor_obs_div(G, rho, status_nodes, method=method, alpha=alpha)
            else:
                obs_selected = gen_selected_sensor_obs(G, rho, status_nodes, method=method)

            # 3) Inference with BP for both random and selected sensors
            bp_fg_rnd = fg.FactorGraph(N,T_max,contacts,[],delta) # for x_rnd: BP with no obs
            bp_fg_rnd.update(maxit=10, print_iter=None)
            marg_rnd = bp_fg_rnd.marginals()
            Mt_rnd = get_Mt(marg_rnd, t=0)
            x_rnd = np.argmax(Mt_rnd, axis=0)

            bp_fg_random = fg.FactorGraph(N,T_max,contacts,obs_random,delta) if with_random else None
            it_rnd, _ =  bp_fg_random.update(maxit=10, print_iter=None) if with_random else None
            marg = bp_fg_random.marginals() if with_random else None

            bp_fg_selected = fg.FactorGraph(N,T_max,contacts,obs_selected,delta)
            it_selected, _ = bp_fg_selected.update(maxit=10, print_iter=None)
            marg_selected = bp_fg_selected.marginals()

            # 4) Compute performance metrics:
            measures_random, x_est_random = compute_measures(marg, s0, delta, status_nodes, x_rnd) if with_random else (None, None)
            measures_selected, x_est_selected = compute_measures(marg_selected, s0, delta, status_nodes, x_rnd)
            rank_random = compute_rank(marg, s0) if with_random else None
            rank_selected = compute_rank(marg_selected, s0)
            precision_random, recall_random = compute_precision_recall(x_est_random, s0) if with_random else (None, None)
            precision_selected, recall_selected = compute_precision_recall(x_est_selected, s0)
            f1_random = compute_f1(precision_random, recall_random) if with_random else None
            f1_selected = compute_f1(precision_selected, recall_selected)

            # 5) Store results in results df
            results_df.loc[len(results_df)] = ["random", rho, delta, lam, measures_random["Ov"], measures_random["Ov_tilde"], rank_random, precision_random, recall_random, f1_random, measures_random["SE"], measures_random["MSE"], sim] if with_random else None
            results_df.loc[len(results_df)] = [method, rho, delta, lam, measures_selected["Ov"], measures_selected["Ov_tilde"], rank_selected, precision_selected, recall_selected, f1_selected, measures_selected["SE"], measures_selected["MSE"], sim]




############### Dynamic sensors pipeline #####


def bp_sim_pipeline_dyn(param_list, Nsim, T_max, N, d, results_df, method):
    for delta, lam, rho in tqdm(param_list, desc="Simulations"):
        for sim in range(Nsim):
            # check if results already exist for this combination of parameters
            if ((results_df["delta"] == delta) & (results_df["lambda"] == lam) & (results_df["rho"] == rho) & (results_df["sensor_type"] == method)).sum() == Nsim:
                continue

            # 1) Generate dynamic process simulation
            G, contacts, s0 = gen_graph_sim(N, d, lam=lam, T_max=T_max, delta=delta)
            status_nodes = simulate_SI(G, s0, lam, T_max)

            # 2) Generate sensor observations (random and selected)
            initial_obs = gen_sensor_obs(rho/2, status_nodes)

            # 3) Inference with BP for both random and selected sensors
            bp_fg_dyn = fg.FactorGraph(N,T_max,contacts,initial_obs,delta)
            run_bp(bp_fg_dyn, initial_obs, rho_max=rho/2, N=N)
            marg = bp_fg_dyn.marginals()

            # 4) Compute performance metrics:
            # Ov_random = compute_overlap_t0(marg, s0)
            # Ov_selected = compute_overlap_t0(marg_selected, s0)
            # Otilde_random = compute_O_tilde(Ov_random, s0, delta)
            # Otilde_selected = compute_O_tilde(Ov_selected, s0, delta)
            # xpred_random = compute_xest_t0(marg)
            # xpred_selected = compute_xest_t0(marg_selected)
            measures_dyn, x_est_dyn = compute_measures(marg, s0, delta, status_nodes)
            rank_random = compute_rank(marg, s0)
            precision_dyn, recall_dyn = compute_precision_recall(x_est_dyn, s0)
            f1_dyn= compute_f1(precision_dyn, recall_dyn) 

            # 5) Store results in results df
            results_df.loc[len(results_df)] = [method, rho, delta, lam, measures_dyn["Ov"], measures_dyn["Ov_tilde"], rank_random, precision_dyn, recall_dyn, f1_dyn, measures_dyn["SE"], measures_dyn["MSE"]]



from SensorSelection.Outdated.dynamic_selection import *

def run_bp(bp_fg, initial_obs, rho_max, N):
    rho = 0 # initial sensor density
    while rho < rho_max:
        bp_fg.iterate()
        marg = bp_fg.marginals()
        obs_selected = max_entropy_selection(marg, bp_fg.status_nodes)
        updated_obs = initial_obs + [obs_selected]
        bp_fg.reset_obs(updated_obs)
        rho += len(obs_selected) / N
    return marg