from sim_pipeline import *
from plot_helpers import *

import numpy as np



graph_kind = "rrg" # or er
lambdas=[0.3]
T_max = 10
N=500
d=3
deltas= np.arange(0.1, 0.5, 0.1)
rho_list = np.arange(0, 1.1, 0.1)
#rho_list = [0.3, 0.6]
Nsim = 10
track_sensor=True


param_list = [(delta, lam, rho) for delta in deltas for lam in lambdas for rho in rho_list]
# sensor selection method

methods = ["random", "greedyOV", "greedyMOV"]# ["random", "page_rank", "betweenness", "degree"] # ["random", "greedyOV", "greedySampleReplaceOV"] # ["random", "betweenness"] # ["random", "greedyOV"] #["random", "page_rank", "betweenness", "degree"] # greedyOV, greedyMOV, random, RL...
params = {"N": N, "d": d, "T_max": T_max, "Nsim": Nsim, "methods": methods, "param_list": param_list, "graph_kind": graph_kind, "track_sensor": track_sensor}

results_df = full_sim(params, Gfixed=False)
