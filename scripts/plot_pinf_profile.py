import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers.plot_helpers import *
from src.helpers.plot_sensor_stats import *

FILE_DIR = "results_new"
#FILE_NAME = "p_inf_stats_full_sweep_log_COMP_Oracle_mov_rrg_N300_T10_d3_Nsim5_del02.csv" #"sensor_stats_full_sweep_logger_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del01_02_03.csv" #"sensor_stats_full_sweep_logger_test_rnd_seq_ov_mov_c_mov_rrg_N30_T10_d3_Nsim3_del0.3.csv"
FILE_NAME = "p_inf_stats_full_sweep_log_COMP_Oracle_c_mov_rrg_N300_T10_d3_Nsim5_del02_04.csv"

SAVE_DIR = "results_new/plots"

SENSOR_PATH = f"{FILE_DIR}/{FILE_NAME}"
pinf_df = pd.read_csv(SENSOR_PATH)

print(pinf_df.head())

delta = 0.2

rhos = [0.0] #[0.0, 0.1, 0.2, 0.5, 0.8]

m=0.2


# select fraction m of nodes, and rhos in rhos
pinf_df = pinf_df[pinf_df["rho"].isin(rhos)]
print(pinf_df.head())
# nb of nodes
num_nodes = len(pinf_df["node"].unique())
print(f"Number of nodes: {num_nodes}")
# select fraction m of nodes
num_selected = int(m * num_nodes)
print(f"Number of nodes to select: {num_selected}")
# randomly select num_selected nodes
selected_nodes = pinf_df["node"].unique()[:num_selected]
print(f"Selected nodes: {selected_nodes}")
pinf_df = pinf_df[pinf_df["node"].isin(selected_nodes)]

# plot pinf profile with hue on rho values
plt.figure(figsize=(10, 6))
# x axis should just rep the selected nodes (no T in pinf_df, so we can just use the index of the selected nodes for x axis) 
sns.lineplot(data=pinf_df, x="node", y="p_inf", hue="true_state", marker="o")
plt.title(f"P_inf Profile for delta={delta} and rhos={rhos}")
plt.xlabel("Nodes")
plt.ylabel("P_inf")
plt.legend(title="Rho")
plt.grid()
plt.savefig(f"{SAVE_DIR}/p_inf_profile_CMOV_COMP_delta{delta}_rhos{rhos}.png")
plt.show()  
