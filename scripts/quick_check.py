import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

FILE_NAME = "sensor_stats_full_sweep_logger_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del01_02_03.csv"
FILE_DIR = "results_new"
PATH = f"{FILE_DIR}/{FILE_NAME}"
sensor_df = pd.read_csv(PATH)

# print values of "true infected" for rho< 0.2

# plot p_inf vs rho for different methods, delta=0.3

delta = 0.3

def filter_df(df, method=None, metric=None, delta=None, lam=None, sim=None):
    dff = df.copy()

    dff["method_metric"] = np.where(
        dff["method"] == "random",
        "rnd",
        dff["method"].astype(str) + "_" + dff["metric"].astype(str)
    )
    return dff 

dff = filter_df(sensor_df)
if delta is not None:
    dff = dff[dff["delta"] == delta]
fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
# print unique values of true_state_t0 
print("Unique values of true_state_t0:", dff["true_state_t0"].unique())
sns.lineplot(data=dff, x="rho", y="true_state_t0", hue="method_metric", palette="Set2", ax=ax)
ax.set_xlabel("rho")
ax.set_ylabel("State of selected node at t=0 (1=infected, 0=not infected)")
plt.tight_layout()
plt.show()
plt.savefig(f"{FILE_DIR}/aaap_inf_vs_rho_comparison.png")
plt.close(fig)  # prevent bleed into next plot
# save figure
