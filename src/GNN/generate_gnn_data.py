# scripts/generate_gnn_data.py
from src.helpers.sim_graph import *
from bpepi.Modules import fg_torch as fg
from src.GNN.data_collection import GNNTrainingCollector

import pandas as pd 
from src.utils.metrics import *
from src.helpers.pipeline_helpers import *
import numpy as np
from collections import defaultdict
import itertools
import copy
from tqdm import tqdm

from src.GNN.sequential_selector import sequential_sensor_selector


def generate_gnn_dataset(deltas, lambdas, rho_max, Nsim, N, T_max, d, graph_type, collector):
    for sim in tqdm(range(Nsim)):
        for delta, lam in itertools.product(deltas, lambdas):
            # draw random rho_max from 0.1-1
            rho_max = np.random.uniform(0.02, 0.3) 
            print(f"\n=== Sim {sim}, delta {delta}, lambda {lam} ===, target rho_max: {rho_max:.2f}")
            # 1. Generate Environment
            G, contacts, s0 = gen_graph_sim(N, d=d, lam=lam, T_max=T_max, delta=delta, kind=graph_type)
            status_nodes = simulate_SI(G, s0, lam, T_max)
            bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)

            # 2. Sequential Selection (Collector is passed as the logger proxy)
            # This will internally call collector.log_candidate and collector.commit_step
            _ = sequential_sensor_selector(
                metric=ov_metric,#mov_constrained_metric,
                bp_base=bp_fg,
                status_nodes=status_nodes,
                rho_max=rho_max,
                m=int(0.2 * N),
                collector=collector,
                max_iter=200,
                tol=1e-4,
                damp=0.5,
                G=G,
                delta=delta,
                lam=lam,
                sim_id=sim
            )
        # Checkpoint every 5 simulations
        if (sim + 1) % 5 == 0:
            collector.save_checkpoint(sim + 1)
    
    #collector.save_dataset()


# def generate_gnn_dataset(deltas, lambdas, rhos, Nsim, N, T_max, d, graph_type = "rrg", collector=None):

#     for sim in tqdm(range(Nsim)):
#         print(f"\n=== Sim {sim} ===")
#         for delta, lam in itertools.product(deltas, lambdas):
#             G, contacts, s0 = gen_graph_sim(N, d=d, lam=lam, T_max=T_max, delta=delta, kind=graph_type)
#             print(f"Initial infection fraction (delta): {s0.sum()/N:.3f}, lambda: {lam}")

#             # add G to collector, along with info of delta, lam, sim id, graph type for completeness
#             collector.add_graph(G, delta=delta, lam=lam, sim_id=sim, graph_type=graph_type)
#             status_nodes = simulate_SI(G, s0, lam, T_max)

#             bp_fg = fg.FactorGraph(N, T_max, contacts, [], delta)

#             # Run sequential method to get sensor order and log candidate stats for GNN training
#             sensor_list = sequential_sensor_selector(metric=ov_metric, bp_base=bp_fg, status_nodes=status_nodes, rho_max=rho_max, m=int(0.2 * N), max_iter=200, tol=1e-4, damp=0.5, delta=delta, collector=collector, G=G)  # get ordered list of sensors selected by sequential method up to max rho

                



