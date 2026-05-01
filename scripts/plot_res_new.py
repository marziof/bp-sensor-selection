import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.helpers.plot_helpers import *

FILE_DIR = "results_new"
FILE_NAME = "full_sweep_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del0.3.csv"
PATH = f"{FILE_DIR}/{FILE_NAME}" 
results_df1 = pd.read_csv(PATH)

FILE_NAME2 = "full_sweep_rnd_seq_ov_mov_c_mov_rrg_N300_T10_d3_Nsim5_del0.05_0.2_0.4.csv"
PATH2 = f"{FILE_DIR}/{FILE_NAME2}"
results_df2 = pd.read_csv(PATH2)

results_df = pd.concat([results_df1, results_df2], ignore_index=True)

SAVE_DIR = "results_new/plots"
SAVE_TITLE = "overlap_comparison_N300_T10_d3_Nsim5_del0.3.png"
SAVE_PATH = f"{SAVE_DIR}/{SAVE_TITLE}"

#print(results_df.head())

plot_comparison(results_df, eval_metric="O_tilde", delta=0.3, save=True, title="overlap_comparison", save_path=SAVE_PATH)

