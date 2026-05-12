
import pandas as pd

from configs.default import *
from src.experiments.full_sweep import run_full_sweep 
from src.utils.sensor_logger import SensorLogger

# -------------------
# RESULTS STORAGE
# -------------------
results_df = pd.DataFrame(columns=["method", "metric", "graph", "rho", "delta", "lambda", "sim", "O", "MO", "O_tilde", "MO_tilde", "SE", "MSE", "rank", "precision", "recall", "f1"])


# -------------------
# Logger
# -------------------
logger = SensorLogger(
    sensor_df=pd.DataFrame(columns=[
        # context
        "method", "metric", "delta", "lambda", "rho", "sim", "graph",

        # selection
        "selected_sensor",

        # p_inf stats
        "p_inf_selected",
        "cand_mean_p_inf",
        "cand_std_p_inf",
        "rank_in_candidates",

        # neighbor stats
        "neigh_mean_p_inf",
        "neigh_std_p_inf",

        # ground truth
        "true_state_t0",
        # entropy stats        
        "entropy_selected",
        "entropy_cand_mean",
        "entropy_neigh_mean",

        # frontier stats (NEW)
        "frontier_selected",
        "frontier_rank",
        "frontier_cand_mean"
    ])
)
# -------------------
# RUN EXPERIMENTS
# -------------------
run_full_sweep(
    methods=methods,
    metrics=metrics,
    deltas=deltas,
    lambdas=lambdas,
    rhos=rhos,
    Nsim=Nsim,
    N=N,
    T_max=T_max,
    d=d,
    results_df=results_df,
    graph_type=graph_type,
    logger=logger
)

# print(results_df["delta"].describe())
# print(results_df["O"].describe())
# print(results_df["O_tilde"].describe())
# print("O values:" , results_df["O"].values)
# print("O_tilde values:" , results_df["O_tilde"].values)

# -------------------
# SAVE
# -------------------
# create save directory if it doesn't exist
import os
os.makedirs(save_dir, exist_ok=True)
results_df.to_csv(f"{save_dir}/{save_title}", index=False)
logger.sensor_df.to_csv(f"{save_dir}/sensor_stats_{save_title}", index=False)
print(f"Saved {save_dir}/{save_title}")