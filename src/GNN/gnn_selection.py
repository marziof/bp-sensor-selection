# test_gnn_selector.py
from src.GNN.model import SensorSelectorGNN
from src.helpers.sim_graph import gen_graph_sim, simulate_SI
from bpepi.Modules import fg_torch as fg
from src.utils.metrics import *
from src.helpers.pipeline_helpers import build_obs
import numpy as np
import torch


def gnn_sensor_selection(model, bp_base, G, contacts, status_nodes, rho_max, max_iter, tol, damp, delta, device):
    target = int(rho_max * bp_base.size)
    sensor_set = set()
    sensor_order = []

    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    saved_messages = bp_base.messages.values.clone()
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(device)

    model.eval()
    for k in range(target):
        remaining = list(set(range(bp_base.size)) - sensor_set)

        marg = bp_base.marginals()
        observed = torch.zeros(bp_base.size, dtype=torch.float).to(device)
        for node in sensor_set:
            observed[node] = 1.0

        x = torch.tensor(marg, dtype=torch.float).to(device)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        x = torch.cat([x, observed.unsqueeze(1)], dim=1)  # (N, T+3)

        with torch.no_grad():
            scores = model(x, edge_index).cpu().numpy()

        # mask out already selected nodes
        scores[list(sensor_set)] = -np.inf
        best = np.argmax(scores)
        sensor_set.add(best)
        sensor_order.append(best)

        N, T = bp_base.size, bp_base.time
        current_obs = build_obs(sensor_set, status_nodes)
        bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
        bp_base.messages.values = saved_messages.clone()
        bp_base.update(maxit=20, tol=tol, damp=damp)
        saved_messages = bp_base.messages.values.clone()

        if k < 5 or k % 10 == 0:
            overlap = OV(np.argmax(get_Mt(bp_base.marginals(), t=0), axis=0), status_nodes[0])
            print(f"[GNN] k={k+1}/{target}, overlap={overlap:.4f}")

    return sensor_order

# def gnn_sensor_selection(model, bp_base, G, contacts, status_nodes, rho_max, max_iter, tol, damp, delta, device):
#     target = int(rho_max * bp_base.size)
#     sensor_set = set()
#     sensor_order = []
#     current_obs = np.empty((0, 3), dtype=int)

#     #bp_base.update(maxit=max_iter, tol=tol, damp=damp)
#     # save messages after first full convergence
#     bp_base.update(maxit=max_iter, tol=tol, damp=damp)
#     saved_messages = bp_base.messages.values.clone()

#     for k in range(target):
#         remaining = list(set(range(bp_base.size)) - sensor_set)

#         # get GNN scores
#         marg = bp_base.marginals()
#         x = torch.tensor(marg, dtype=torch.float).to(device)
#         x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
#         edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous().to(device)

#         model.eval()
#         with torch.no_grad():
#             scores = model(x, edge_index).cpu().numpy()  # (N,)

#         best = remaining[np.argmax(scores[remaining])]
#         sensor_set.add(best)
#         sensor_order.append(best)

#         N, T = bp_base.size, bp_base.time
#         current_obs = build_obs(sensor_set, status_nodes)
#         bp_base = fg.FactorGraph(N, T, contacts, current_obs, delta)
#         bp_base.messages.values = saved_messages.clone()  # warm start
#         bp_base.update(maxit=20, tol=tol, damp=damp)      # 20 instead of 200
#         saved_messages = bp_base.messages.values.clone()

#         if k < 5 or k % 10 == 0:
#             overlap = OV(np.argmax(get_Mt(bp_base.marginals(), t=0), axis=0), status_nodes[0])
#             print(f"[GNN] k={k+1}/{target}, overlap={overlap:.4f}")

#     return sensor_order