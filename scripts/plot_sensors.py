import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers.plot_helpers import *
from src.helpers.plot_sensor_stats import *

FILE_DIR = "results_new"
SENSOR_FILE_NAME = "sensor_stats_full_sweep_logger_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del01_02_03.csv" #"sensor_stats_full_sweep_logger_test_rnd_seq_ov_mov_c_mov_rrg_N30_T10_d3_Nsim3_del0.3.csv"


SAVE_DIR = "results_new/plots"

SENSOR_PATH = f"{FILE_DIR}/{SENSOR_FILE_NAME}"
sensor_df = pd.read_csv(SENSOR_PATH)

delta = 0.2
# print(sensor_df.head())
# # check how many methods
# print(sensor_df["method"].unique())

SAVE_TITLE = f"pinf_comparison_{delta}.png"
plot_pinf_vs_rho(sensor_df, delta=delta, save=True, save_path=f"{SAVE_DIR}/pinf_vs_rho_{SAVE_TITLE}")

SAVE_TITLE = f"rank_comparison_{delta}.png"
plot_rank_vs_rho(sensor_df, delta=delta, save=True, save_path=f"{SAVE_DIR}/rank_vs_rho_{SAVE_TITLE}")

SAVE_TITLE = f"true_infected_comparison_{delta}.png"
plot_true_infected_vs_rho(sensor_df, delta=delta, save=True, save_path=f"{SAVE_DIR}/true_infected_vs_rho_{SAVE_TITLE}")

SAVE_TITLE = f"neighbor_pinf_comparison_{delta}.png"
plot_neighbor_pinf_vs_rho(sensor_df, delta=delta, save=True, save_path=f"{SAVE_DIR}/neighbor_pinf_vs_rho_{SAVE_TITLE}")  

SAVE_TITLE = f"entropy_cand_comparison_{delta}.png"
plot_entropy_vs_rho(sensor_df, delta=delta, save=True, save_path=f"{SAVE_DIR}/entropy_vs_rho_{SAVE_TITLE}")
