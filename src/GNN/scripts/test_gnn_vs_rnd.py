from src.GNN.gnn_selection import gnn_sensor_selection
from src.GNN.model import SensorSelectorGNN, SensorSelectorMLP
import pandas as pd 
from bpepi.Modules import fg_torch as fg #pytorch version
from src.helpers.sim_graph import *
from src.utils.metrics import *
from src.helpers.pipeline_helpers import *
import numpy as np
from collections import defaultdict
import itertools
import copy
from tqdm import tqdm
from src.algorithms.static_selection import random_selection
import torch
import os


# params
N = 50
T_max = 10
d = 3
deltas = [0.3]
lambdas = [0.3]
rhos = np.arange(0, 1.1, 0.1)
Nsim = 5
graph_type = "rrg"

# load GNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SensorSelectorMLP(in_channels=T_max+3, hidden=64).to(device)
#model = SensorSelectorGNN(in_channels=13, hidden_channels=64).to(device)
model.load_state_dict(torch.load("src/GNN/models/sensor_gnn_final_att_500_var_cmov_mlp.pth", map_location=device))
model.eval()
print("Model loaded and set to eval mode.")

#results_df = pd.DataFrame(columns=["method", "sim_id", "delta", "lam", "rho", "overlap"])

results = []

for sim in tqdm(range(Nsim)):
    for delta, lam in itertools.product(deltas, lambdas):
        G, contacts, s0 = gen_graph_sim(N, d=d, lam=lam, T_max=T_max, delta=delta, kind=graph_type)
        status_nodes = simulate_SI(G, s0, lam, T_max)
        bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)


        # random baseline for normalization (passed to evaluate_sensors)
        rnd_order = np.random.permutation(N).tolist()
        rnd_obs = build_obs(set(rnd_order[:int(0.3 * N)]), status_nodes)
        x_rnd, Mt_rnd, _ = compute_bp_estimates(N, T_max, contacts, rnd_obs, delta)
        print(f"Random baseline computed for sim={sim}, delta={delta}, lam={lam}")
        # GNN order
        gnn_order = gnn_sensor_selection(model, bp_fg, G, contacts, status_nodes, rho_max=1.0, max_iter=100, tol=1e-4, damp=0.5, delta=delta, device=device)
        print(f"GNN selection completed for sim={sim}, delta={delta}, lam={lam}")
        for k in range(1, N+1):
            rho = k / N
            # GNN Selection
            metrics = evaluate_sensors(
                selected_sensors=set(gnn_order[:k]),
                bp_fg=fg.FactorGraph(N, T_max, contacts, [], delta),
                status_nodes=status_nodes,
                N=N, T_max=T_max, delta=delta,
                x_rnd=x_rnd, Mt_rnd=Mt_rnd
            )
            metrics.update({"method": "gnn", "sim": sim, "delta": delta, "lam": lam, "rho": rho})
            results.append(metrics)
            # Random Selection
            metrics_rnd = evaluate_sensors(
                selected_sensors=set(rnd_order[:k]),
                bp_fg=fg.FactorGraph(N, T_max, contacts, [], delta),
                status_nodes=status_nodes,
                N=N, T_max=T_max, delta=delta,
                x_rnd=x_rnd, Mt_rnd=Mt_rnd
            )
            metrics_rnd.update({"method": "random", "sim": sim, "delta": delta, "lam": lam, "rho": rho})
            results.append(metrics_rnd)
    
# ── save and print ─────────────────────────────────────────────────────────────
os.makedirs("src/GNN/results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("src/GNN/results/gnn_vs_random_N50_att_500_var_cmov_mlp.csv", index=False)

summary = results_df.groupby(["method", "rho"])["O"].mean().unstack("method")
print(summary)