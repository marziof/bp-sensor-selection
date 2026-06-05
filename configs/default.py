
import itertools
import numpy as np
from src.algorithms.sequential_sensor_selection import sequential_sensor_selection, ov_metric, mov_metric, mov_constrained_metric
from src.algorithms.static_selection import *
from src.algorithms.non_oracle_selection import path_weight_sensor_selection, max_entropy_selection, max_pinf_selection

# -------------------
# PARAM GRID
# -------------------
deltas = [0.05, 0.1, 0.2]
lambdas = [0.3]
rhos = np.arange(0, 1.1, 0.1)

N = 1000
T_max = 20
d = 3 # gamma for powerlaw graph, degree for rrg
graph_type = "rrg" #"er" # "rrg" # "er" # "rrg" #"rrg" #"er"
Nsim = 20

SIM_NAME = "non_oracle" #"static_er" #"non_oracle" # "Oracle" #"PathWeight" #"Oracle" #"PathWeight" #"static_selection"

# -------------------
# METHODS
# -------------------
# methods = {
#     #"entropy": entropy_sensor_selection,
#     "random": random_selection,
#     "deg_centrality": deg_centrality_selection,
#     "betweenness_centrality": betweenness_centrality_selection,
#     "page_rank": page_rank_selection
#     # "closeness": closeness_selection
#     }

methods = {
    "path_weight": path_weight_sensor_selection,
    "max_pinf": max_pinf_selection,
    "max_entropy": max_entropy_selection,
    "random": random_selection
}

# methods = {
#     #"random": random_selection,
#     "sequential": sequential_sensor_selection
# }

# -------------------
# METRICS
# -------------------
metrics = {
    #"ov": ov_metric
    #"mov": mov_metric
    "c_mov": mov_constrained_metric
}

# save_dir and title
save_dir = "results"
# methods as rnd_seq and metrics as ov_mov_c_mov for filename
metrics_str = "_".join(metrics.keys())
delta_str = "_".join([str(dd) for dd in deltas])
# remove all "." from delta_str for filename
delta_str = delta_str.replace(".", "")
#save_title = f"full_sweep_logger_rnd_seq_{metrics_str}_{graph_type}_N{N}_T{T_max}_d{d}_Nsim{Nsim}_del{delta_str}.csv"
save_title = f"full_sweep_log_{SIM_NAME}_{metrics_str}_{graph_type}_N{N}_T{T_max}_d{d}_Nsim{Nsim}_del{delta_str}.csv"