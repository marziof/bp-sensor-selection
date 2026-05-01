from plot_helpers import *
import pandas as pd


PATH = "results/checkpoints/sensors_greedyWarmStartMOV_N1000_d3_T10_Nsim5_checkpoint_tadapt_sim0.csv" 

#PATH = "results/results_dfs/random_greedyWarmStartMOV_N500_d3_T10_Nsim5_final.csv" 
#PATH = "results/checkpoints/greedyWarmStartMOV_N1000_d3_T10_Nsim5_checkpoint_tadapt_sim0.csv"
results_df = pd.read_csv(PATH)

# PATH2 = "results/results_dfs/greedyWarmStartOV_greedyOV_N500_d3_T10_Nsim3_tadapt.csv"
# results_df2 = pd.read_csv(PATH2)

# PATH3 = "results/results_dfs/greedyWarmStartMOV_N500_d3_T10_Nsim3_plainMOV.csv"
# results_df3 = pd.read_csv(PATH3)
# # rename method in df3 to "greedyWarmstartMOV_plainMOV"
# results_df3["method"] = "greedyWarmstartMOV_plainMOV"

# results_df = pd.concat([results_df, results_df2, results_df3], ignore_index=True)

#x = results_df[results_df["method"] == "greedyMOV"]
#y = results_df2[results_df2["method"] == "greedyOV"]
#plot_side_by_side(x,y, metric="O_tilde", save=True, title="overlap_comparison_sides")


import re
import pandas as pd

def parse_ov_rho(filepath, method="unknown", graph="unknown", delta=None, lam=None, sim=None):
    pattern = re.compile(
        r"OV=(?P<ov>\d+\.\d+).*rho=(?P<rho>\d+\.\d+)"
    )

    rows = []

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ov = float(match.group("ov"))
                rho = float(match.group("rho"))

                rows.append({
                    "method": method,
                    "graph": graph,
                    "rho": rho,
                    "delta": delta,
                    "lambda": lam,
                    "sim": sim,
                    "O": ov,              # OV
                    "MO": None,
                    "O_tilde": None,
                    "MO_tilde": None,
                    "SE": None,
                    "MSE": None,
                    "rank": None,
                    "precision": None,
                    "recall": None,
                    "f1": None
                })

    results_df = pd.DataFrame(rows)
    return results_df

# results_df = df = parse_ov_rho("./gen_results.out", method="greedy_soft", graph="formica", delta=0.3)
# results_df = pd.concat([results_df, results_df2], ignore_index=True)
plot_comparison(results_df, metric="O_tilde", delta=0.3, save=True, title="overlap_comparison")




# # sensor plots:

# SENSOR_PATH = "results/sensors_dfs/sensors_greedyWarmStartOV_greedyOV_N500_d3_T10_Nsim3_tadapt.csv" 
# sensors_df = pd.read_csv(SENSOR_PATH)

# SENSOR_PATH2 = "results/sensors_dfs/sensors_random_greedyWarmStartMOV_N500_d3_T10_Nsim5_tadapt.csv"
# sensors_df2 = pd.read_csv(SENSOR_PATH2)

# SENSOR_PATH3 = "results/sensors_dfs/sensors_greedyWarmStartMOV_N500_d3_T10_Nsim3_plainMOV.csv"
# sensors_df3 = pd.read_csv(SENSOR_PATH3)
# sensors_df3["method"] = "greedyWarmstartMOV_plainMOV"

# sensors_df = pd.concat([sensors_df, sensors_df2, sensors_df3], ignore_index=True)

# plot_sensor_properties_vs_rho(sensors_df, property_name="boundary_size", save=True, title="boundary_size_vs_rho")

# plot_sensor_properties(sensors_df, property_name="boundary_size", save=True, title="boundary_size_comparison")