
import numpy as np
import torch

from src.Analysis.measures import OV, MOV, SE, MSE, OV_rnd, MOV_rnd
from src.helpers.pipeline_helpers import get_Mt

###--------------------- BAYESIAN INFERENCE METRICS ----------------------###


### For greedy selection: ###

def ov_metric(marginals, status_nodes, **kwargs):
    Mt = get_Mt(marginals, t=0)
    x_est = np.argmax(Mt, axis=0)
    return OV(x_est, status_nodes[0])

def mov_metric(marginals, status_nodes=None, **kwargs):
    Mt = get_Mt(marginals, t=0)
    return MOV(Mt)

def mov_constrained_metric(marginals, status_nodes=None, delta=0, alpha=0.2, **kwargs):
    #print("Running constrained MOV metric with delta =", delta)
    N = marginals.shape[0]
    k = int(delta * N)
    Mt = get_Mt(marginals, t=0)
    p_base = Mt[1]  # keep true probabilistic meaning
    time_scores = time_score_from_b(marginals)
    time_scores = (time_scores - time_scores.min()) / (time_scores.max() - time_scores.min() + 1e-12)
    p_inf = p_base + alpha * time_scores
    # ---------- top-k selection ----------
    top_k = np.argsort(-p_inf)[:k]
    x_est = np.zeros(N, dtype=int)
    x_est[top_k] = 1
    # ---------- evaluation uses TRUE probability only ----------
    conf = np.where(x_est == 1, Mt[1], 1 - Mt[1])
    return np.mean(conf)

METRICS = {
    "ov": ov_metric,
    "mov": mov_metric,
    "c_mov": mov_constrained_metric
}

def metric(name, *args, **kwargs):
    return METRICS[name](*args, **kwargs)


def time_score_from_b(marginals, decay=0.3):
    """
    marginals: (N, T+2)
        - last time index is t = inf (absorbing state)
    returns: (N,)
    """

    marginals = np.asarray(marginals)  # (N, T+2)

    N, T_full = marginals.shape
    T = T_full - 1  # exclude inf state

    t = np.arange(T)

    weights = np.exp(-decay * t)
    weights = weights / weights.sum()

    # weighted sum over time axis (axis=1)
    # result: (N,)
    return np.sum(marginals[:, :T] * weights[None, :], axis=1)


###--------------------- Accuracy metrics ----------------------###
    
def compute_rank(marginals, x_true):
    p = marginals[:, 0]  # P(t_i=0)
    sorted_idx = np.argsort(-p)  # descending

    true_indices = np.where(x_true == 1)[0]

    ranks = []
    for i in true_indices:
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(rank)

    return np.mean(ranks)

def compute_normalized_rank(marginals, x_true):
    rank = compute_rank(marginals, x_true)
    N = len(x_true)
    return 1 - (rank - 1) / (N - 1)


def compute_precision_recall(x_pred, x_true):
    TP = np.sum((x_pred == 1) & (x_true == 1))
    FP = np.sum((x_pred == 1) & (x_true == 0))
    FN = np.sum((x_pred == 0) & (x_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def compute_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)
