
import itertools
import numpy as np
from src.algorithms.sequential_sensor_selection import sequential_sensor_selection, ov_metric, mov_metric, mov_constrained_metric
from src.algorithms.static_selection import random_selection


# -------------------
# PARAM GRID
# -------------------
deltas = [0.05, 0.2, 0.4]
lambdas = [0.3]
rhos = np.arange(0, 1.1, 0.1)

N = 300
T_max = 10
d = 3
graph_type = "rrg"
Nsim = 5


# -------------------
# METHODS
# -------------------
methods = {
    "random": random_selection,
    "sequential": sequential_sensor_selection
}

# -------------------
# METRICS
# -------------------
metrics = {
    "ov": ov_metric,
    "mov": mov_metric,
    "c_mov": mov_constrained_metric
}

# save_dir and title
save_dir = "results_new"
# methods as rnd_seq and metrics as ov_mov_c_mov for filename
metrics_str = "_".join(metrics.keys())
delta_str = "_".join([str(dd) for dd in deltas])
save_title = f"full_sweep_rnd_seq_{metrics_str}_{graph_type}_N{N}_T{T_max}_d{d}_Nsim{Nsim}_del{delta_str}.csv"