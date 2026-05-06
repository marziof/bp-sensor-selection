# import torch
# import numpy as np
# from scipy.stats import spearmanr
# # print current location to debug
# import os
# print(f"Current working directory: {os.getcwd()}")
# from src.GNN.model import SensorSelectorGNN
# from src.GNN.dataloader import get_full_dataloader
# from tqdm import tqdm

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # load model
# model = SensorSelectorGNN(in_channels=13, hidden_channels=64).to(device)
# model.load_state_dict(torch.load("src/GNN/models/sensor_gnn_final_att.pth", map_location=device))
# loader = get_full_dataloader(save_dir="src/GNN/data2", batch_size=32)
# model.eval()
# correlations = []
# with torch.no_grad():
#     for i, batch in tqdm(enumerate(loader), total=len(loader)):
#         if i >= 10:  # only check 10 batches
#             break
#         batch = batch.to(device)
#         out = model(batch.x, batch.edge_index)
        
#         for graph_id in batch.batch.unique():
#             mask = (batch.batch == graph_id) & batch.mask
#             if mask.sum() < 2:
#                 continue
#             scores = out[mask].cpu().numpy()
#             gains = batch.y[mask].cpu().numpy()
#             rho, _ = spearmanr(scores, gains)
#             correlations.append(rho)

# correlations = np.array(correlations)
# print(f"Spearman rho: mean={correlations.mean():.4f}, std={correlations.std():.4f}")
# print(f"Positive correlations: {(correlations > 0).mean():.2%}")
# print(f"Strong correlations (rho>0.3): {(correlations > 0.3).mean():.2%}")

import torch
import numpy as np
from scipy.stats import spearmanr
import os
print(f"Current working directory: {os.getcwd()}")

from src.GNN.model import SensorSelectorGNN
from src.helpers.sim_graph import gen_graph_sim, simulate_SI
from src.utils.metrics import ov_metric, get_Mt
from src.GNN.sequential_selector import build_obs
from bpepi.Modules import fg_torch as fg
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = SensorSelectorGNN(in_channels=13, hidden_channels=64).to(device)
model.load_state_dict(torch.load("src/GNN/models/sensor_gnn_final_att_50_var.pth", map_location=device))
model.eval()

# config — same as training
N, T_max, d = 50, 10, 3
delta, lam = 0.3, 0.3
graph_type = "rrg"
n_test_graphs = 50

correlations = []

with torch.no_grad():
    for _ in tqdm(range(n_test_graphs)):
        # fresh graph never seen during training
        G, contacts, s0 = gen_graph_sim(N, d=d, lam=lam, T_max=T_max, delta=delta, kind=graph_type)
        status_nodes = simulate_SI(G, s0, lam, T_max)

        bp = fg.FactorGraph(N, T_max, contacts, [], delta)
        bp.update(maxit=200, tol=1e-4, damp=0.5)
        marg = bp.marginals()

        # GNN scores
        x = torch.tensor(marg, dtype=torch.float).to(device)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        observed = torch.zeros(N, dtype=torch.float).to(device)  # no sensors yet
        x = torch.cat([x, observed.unsqueeze(1)], dim=1)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(device)
        scores = model(x, edge_index).cpu().numpy()

        # CMO oracle gains for same graph — one step
        saved = bp.messages.values.clone()
        gains = []
        candidates = list(range(N))
        metric_base = ov_metric(marg, status_nodes=status_nodes, delta=delta)
        for cand in candidates:
            bp.messages.values = saved.clone()
            obs = build_obs({cand}, status_nodes)
            bp.reset_obs(obs)
            bp.update(maxit=20, tol=1e-4, damp=0.5)
            score = ov_metric(bp.marginals(), status_nodes=status_nodes, delta=delta)
            gains.append(score - metric_base)

        gains = np.array(gains)
        rho, _ = spearmanr(scores, gains)
        correlations.append(rho)

correlations = np.array(correlations)
print(f"Spearman rho: mean={correlations.mean():.4f}, std={correlations.std():.4f}")
print(f"Positive correlations: {(correlations > 0).mean():.2%}")
print(f"Strong correlations (rho>0.3): {(correlations > 0.3).mean():.2%}")


from src.GNN.dataloader import get_full_dataloader
loader = get_full_dataloader(save_dir="src/GNN/data_var", batch_size=32)
shuffled_correlations = []
model.eval()
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out_original = model(batch.x, batch.edge_index)
        
        x_shuffled = batch.x[torch.randperm(batch.x.size(0))]
        out_shuffled = model(x_shuffled, batch.edge_index)
        
        corr, _ = spearmanr(
            out_original.cpu().numpy().flatten(), 
            out_shuffled.cpu().numpy().flatten()
        )
        shuffled_correlations.append(corr)
        print(f"Correlation original vs shuffled: {corr:.4f}")
        break

# print these in a file "evaluation_results.txt" in resuts directory
with open("src/GNN/results/evaluation_results.txt", "w") as f:
    f.write(f"Spearman rho: mean={correlations.mean():.4f}, std={correlations.std():.4f}\n")
    f.write(f"Positive correlations: {(correlations > 0).mean():.2%}\n")
    f.write(f"Strong correlations (rho>0.3): {(correlations > 0.3).mean():.2%}\n")
    f.write(f"Correlation original vs shuffled: {np.mean(shuffled_correlations):.4f} ± {np.std(shuffled_correlations):.4f}\n")
