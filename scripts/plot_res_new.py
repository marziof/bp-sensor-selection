import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers.plot_helpers import *
from src.helpers.plot_sensor_stats import *

FILE_DIR = "results_new"
FILE_NAME = "full_sweep_log_static_selection_ov_rrg_N100_T20_d3_Nsim10_del005_01_02_03_04.csv" #"full_sweep_log_static_selection_ov_rrg_N300_T10_d3_Nsim10_del005_01_02_03_04.csv" # "full_sweep_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del0.3.csv" #"full_sweep_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del0.3.csv"  
PATH = f"{FILE_DIR}/{FILE_NAME}" 
results_df = pd.read_csv(PATH)
print(len(results_df))
PLOT_NAME = "NEW_test" #"OpracticalVsStatic_N300_T10_d3_Nsim10" #"static_rrg_N300_T10_d3_Nsim10" # "seqOvOracle_N300_T10_Nsim5" #"practicalVsOracle_N300_T10_d3_Nsim105"

results_df2 = pd.read_csv(f"{FILE_DIR}/full_sweep_log_cone3modComb_rnd_seq_ov_rrg_N300_T10_d3_Nsim10_del005_01_02_03_04.csv")

# #results_df = results_df[results_df["metric"] != "mov"] # remove mov from plot
#results_df = results_df[results_df["method"].isin(["closeness", "random"])] # remove mov from plot
# # #results_df = results_df[results_df["method"] != "random"] # remove random from plot
results_df2 = results_df2[results_df2["method"] != "random"] # remove random from plot

# # results_df2 = pd.read_csv(f"{FILE_DIR}/full_sweep_log_cone2mod_rnd_seq_ov_rrg_N300_T10_d3_Nsim10_del03.csv")
# # results_df3 = pd.read_csv(f"{FILE_DIR}/full_sweep_log_cone3mod_rnd_seq_ov_rrg_N300_T10_d3_Nsim10_del04.csv")
# # results_df4 = pd.read_csv(f"{FILE_DIR}/full_sweep_log_cone2mod_rnd_seq_ov_rrg_N300_T10_d3_Nsim10_del01.csv")

#results_df = pd.concat([results_df, results_df2, results_df3, results_df4], ignore_index=True)

# # FILE_NAME2 = "full_sweep_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del0.05_0.2_0.4.csv"
# # PATH2 = f"{FILE_DIR}/{FILE_NAME2}"
# # results_df2 = pd.read_csv(PATH2)

#results_df = pd.concat([results_df, results_df2], ignore_index=True)

delta = 0.4
# print nb of entries in df for delta=0.1
#print(f"Number of entries for delta={delta}: {len(results_df[results_df['delta'] == delta])}")
delta_str = str(delta).replace(".", "")
SAVE_DIR = "results_new/plots"
SAVE_TITLE = f"{PLOT_NAME}_del{delta_str}"#f"N100_T10_d3_Nsim5_del{delta_str}_cone" #overlap_comparison_N300_T10_d3_Nsim5_del0.3.png"
SAVE_PATH = f"{SAVE_DIR}/{SAVE_TITLE}"
plot_comparison(results_df, eval_metric="O_tilde", delta=delta, save=True, title=None, save_path=SAVE_PATH)

# for delta in [0.1, 0.2, 0.3, 0.4]:
#     delta_str = str(delta).replace(".", "")
#     SAVE_DIR = "results_new/plots"
#     SAVE_TITLE = f"N300_T10_d3_Nsim10_del{delta_str}_cone3mod" #overlap_comparison_N300_T10_d3_Nsim10_del0.3.png"
#     SAVE_PATH = f"{SAVE_DIR}/{SAVE_TITLE}"

#     plot_comparison(results_df, eval_metric="O_tilde", delta=delta, save=True, title=None, save_path=SAVE_PATH)

# method_metric="entropy"
# plot_delta_comparison(results_df, eval_metric="O_tilde", method_metric=method_metric, save=True, title=None, save_path=f"{SAVE_DIR}/delta_comparison_O_tilde_{method_metric}")






# plot_metrics_comparison(results_df, metric1="O_tilde", metric2="MO_tilde", delta=delta, save=True, title=None, save_path=f"{SAVE_DIR}/overlap_comparison_O_tilde_vs_MO_tilde_{SAVE_TITLE}")

# plot_metrics_comparison(results_df, metric1="SE", metric2="MSE", delta=delta, save=True, title=None, save_path=f"{SAVE_DIR}/overlap_comparison_SE_vs_MSE_{SAVE_TITLE}")

# SENSOR_FILE_NAME = "sensor_stats_full_sweep_logger_test_rnd_seq_ov_mov_c_mov_rrg_N30_T10_d3_Nsim3_del0.3.csv"
# SENSOR_PATH = f"{FILE_DIR}/{SENSOR_FILE_NAME}"
# sensor_df = pd.read_csv(SENSOR_PATH)

# print(sensor_df.head())
# # check how many methods
# print(sensor_df["method"].unique())

# SAVE_TITLE = "pinf_comparison.png"
# plot_pinf_vs_rho(sensor_df, save=True, save_path=f"{SAVE_DIR}/pinf_vs_rho_{SAVE_TITLE}")

# SAVE_TITLE = "rank_comparison.png"
# plot_rank_vs_rho(sensor_df, save=True, save_path=f"{SAVE_DIR}/rank_vs_rho_{SAVE_TITLE}")

# SAVE_TITLE = "true_infected_comparison.png"
# plot_true_infected_vs_rho(sensor_df, save=True, save_path=f"{SAVE_DIR}/true_infected_vs_rho_{SAVE_TITLE}")

# SAVE_TITLE = "neighbor_pinf_comparison.png"
# plot_neighbor_pinf_vs_rho(sensor_df, save=True, save_path=f"{SAVE_DIR}/neighbor_pinf_vs_rho_{SAVE_TITLE}")   

