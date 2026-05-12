import pandas as pd
import numpy as np
import networkx as nx
from src.utils.metrics import get_Mt


class SensorLogger:
    def __init__(self, sensor_df):
        self.sensor_df = sensor_df

    def set_context(self, method_name, metric_name, delta, lam, sim, graph_type):
        self.context = {
            "method": method_name,
            "metric": metric_name,
            "delta": delta,
            "lambda": lam,
            "sim": sim,
            "graph": graph_type,
        }

    # log sensor p_inf + stats (mean, std) of p_inf among candidates, rank of selected sensor in terms of p_inf
    # mean + std of p_inf for neighbors of selected sensor
    # true state of selected sensor at t=0
    def log_sensor_stats(self, selected_sensor, candidates, marginals, status_nodes, rho, graph=None):
        Mt = get_Mt(marginals, t=0)
        p_inf = Mt[1]

        p_sel = p_inf[selected_sensor]
        cand_p = p_inf[candidates]
        mean_cand = np.mean(cand_p)
        std_cand = np.std(cand_p)
        rank = np.sum(cand_p > p_sel) + 1

        if graph is not None:
            neighbors = list(graph.neighbors(selected_sensor))
            if len(neighbors) > 0:
                neigh_p = p_inf[neighbors]
                mean_neigh = np.mean(neigh_p)
                std_neigh = np.std(neigh_p)
                # ADD: frontier score for selected sensor
                frontier_selected = np.mean(np.abs(p_sel - neigh_p))
            else:
                mean_neigh = np.nan
                std_neigh = np.nan
                frontier_selected = np.nan
            
            # ADD: frontier score for all candidates
            frontier_cands = []
            for c in candidates:
                c_neighbors = list(graph.neighbors(c))
                if len(c_neighbors) > 0:
                    c_neigh_p = p_inf[c_neighbors]
                    frontier_cands.append(np.mean(np.abs(p_inf[c] - c_neigh_p)))
                else:
                    frontier_cands.append(0.0)
            frontier_cands = np.array(frontier_cands)
            
            # rank of selected sensor by frontier score
            frontier_rank = np.sum(frontier_cands > frontier_selected) + 1
        else:
            mean_neigh = np.nan
            std_neigh = np.nan
            frontier_selected = np.nan
            frontier_rank = np.nan

        true_state = int(status_nodes[0, selected_sensor])

        row = {**self.context,
            "rho": rho,
            "selected_sensor": selected_sensor,
            "p_inf_selected": p_sel,
            "cand_mean_p_inf": mean_cand,
            "cand_std_p_inf": std_cand,
            "rank_in_candidates": rank,
            "neigh_mean_p_inf": mean_neigh,
            "neigh_std_p_inf": std_neigh,
            "true_state_t0": true_state,
            "entropy_selected": compute_entropy_nodes(marginals, selected_sensor),
            "entropy_cand_mean": np.mean([compute_entropy_nodes(marginals, c) for c in candidates]),
            "entropy_neigh_mean": np.mean([compute_entropy_nodes(marginals, n) for n in graph.neighbors(selected_sensor)]) if graph is not None else np.nan,
            # NEW
            "frontier_selected": frontier_selected,
            "frontier_rank": frontier_rank,
            "frontier_cand_mean": np.mean(frontier_cands) if graph is not None else np.nan,
        }
        self.sensor_df.loc[len(self.sensor_df)] = row
    # def log_sensor_stats(self, selected_sensor, candidates, marginals, status_nodes, rho, graph=None):
    #     """
    #     selected_sensor: int
    #     candidates: list[int]
    #     marginals: np.array of shape N, T+2 -> to get Mt (num_states, N) at t=0
    #     status_nodes: array (T, N)
    #     graph: networkx graph (optional, for neighbor stats)
    #     """
    #     Mt = get_Mt(marginals, t=0)  # shape (num_states, N)

    #     # --- p_inf for all nodes ---
    #     p_inf = Mt[1]  # adjust index if "infected" is not state 1

    #     # --- selected sensor stats ---
    #     p_sel = p_inf[selected_sensor]

    #     # --- candidate stats ---
    #     cand_p = p_inf[candidates]
    #     mean_cand = np.mean(cand_p)
    #     std_cand = np.std(cand_p)

    #     # rank (higher p_inf = better rank)
    #     rank = np.sum(cand_p > p_sel) + 1  # 1 = best

    #     # --- neighbor stats ---
    #     if graph is not None:
    #         neighbors = list(graph.neighbors(selected_sensor))
    #         if len(neighbors) > 0:
    #             neigh_p = p_inf[neighbors]
    #             mean_neigh = np.mean(neigh_p)
    #             std_neigh = np.std(neigh_p)
    #         else:
    #             mean_neigh = np.nan
    #             std_neigh = np.nan
    #     else:
    #         mean_neigh = np.nan
    #         std_neigh = np.nan

    #     # --- true state at t=0 ---
    #     true_state = int(status_nodes[0, selected_sensor])

    #     # --- log everything ---
    #     row = {**self.context, 
    #         "rho": rho,
    #         "selected_sensor": selected_sensor,
    #         "p_inf_selected": p_sel,
    #         "cand_mean_p_inf": mean_cand,
    #         "cand_std_p_inf": std_cand,
    #         "rank_in_candidates": rank,
    #         "neigh_mean_p_inf": mean_neigh,
    #         "neigh_std_p_inf": std_neigh,
    #         "true_state_t0": true_state,
    #         "entropy_selected": compute_entropy_nodes(marginals, selected_sensor),
    #         "entropy_cand_mean": np.mean([compute_entropy_nodes(marginals, c) for c in candidates]),
    #         "entropy_neigh_mean": np.mean([compute_entropy_nodes(marginals, n) for n in graph.neighbors(selected_sensor)]) if graph is not None else np.nan
    #         # NEW
    #         "frontier_selected": frontier_selected,
    #         "frontier_rank": frontier_rank,
    #         "frontier_cand_mean": np.mean(frontier_cands) if graph is not None else np.nan
    #         }
    #     self.sensor_df.loc[len(self.sensor_df)] = row


def compute_entropy_nodes(marginals, node):
    # marginals: shape N, T+2
    # entropy of infection time distribution for each node
    p = marginals[node, :]  # shape (num_states,)
    p = p[p > 0]  # avoid log(0)
    entropy = -np.sum(p * np.log(p))
    return entropy


def frontier_score(marg, G, t=0):
    """
    Score nodes by how much their infection probability contrasts with neighbors.
    High score = node is on the infection frontier.
    """
    Mt = get_Mt(marg, t=0)
    p_inf = Mt[1]  # P(infected) per node, shape (N,)
    
    scores = np.zeros(len(p_inf))
    for i in range(len(p_inf)):
        neighbors = list(G.neighbors(i))
        if len(neighbors) == 0:
            continue
        # contrast = how different is node i from its neighbors
        neighbor_p = np.array([p_inf[j] for j in neighbors])
        scores[i] = np.mean(np.abs(p_inf[i] - neighbor_p))
    
    return scores  # (N,)