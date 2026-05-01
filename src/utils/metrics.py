
import numpy as np


###--------------------- BAYESIAN INFERENCE METRICS ----------------------###


def compute_measures(marginals, status_nodes, x_rnd, Mt_rnd):
    """
    Receives: marginals: array of shape (N, T+2) containing the beliefs estimated by BP, s0: array of shape (N,) containing the ground truth initial states, delta: initial probability of being infected, status_nodes: array of shape (T+1, N) containing the ground truth states of the system from t=0 to t=T
    Returns: a dictionary containing the computed measures
    """
    s0 = status_nodes[0]
    measures = {}
    Mt = get_Mt(marginals, t=0)
    x_est = np.argmax(Mt, axis=0)
    ti_star_vec = ti_star(status_nodes, T_BP=marginals.shape[1]-2)
    ti_inf_vec = ti_inferred(marginals)
    ti_rnd_vec = ti_random(marginals)
    measures["Ov"] = OV(x_est, s0)
    measures["Ov_rnd"] = OV(x_rnd, s0)  # OV_rnd(s0, Mt)
    measures["Ov_tilde"] = (measures["Ov"] - measures["Ov_rnd"]) / (1 - measures["Ov_rnd"]) if measures["Ov_rnd"] < 1 else 1.0
    measures["MO"] = MOV(Mt)
    measures["MO_rnd"] = MOV(Mt_rnd)  # MOV_rnd(Mt)
    measures["MO_tilde"] = (measures["MO"] - measures["MO_rnd"]) / (1 - measures["MO_rnd"]) if measures["MO_rnd"] < 1 else 1.0
    measures["SE"] = SE(ti_star_vec, ti_inf_vec)
    measures["MSE"] = MSE(marginals, ti_inf_vec)
    return measures


def get_Mt(B, t=0):
    """
    Receives: B: shape (N, T+2), infection time distribution
    Returns: Mt: shape (2, N) with [P(S), P(I)] at time t
    """
    #assert np.allclose(B.sum(axis=1), 1), "Beliefs do not sum to 1; B entry that does not sum to 1: {}".format(B[np.where(~np.isclose(B.sum(axis=1), 1))[0][0]])
    p_inf = np.sum(B[:, :t+1], axis=1)   # P(infected by time t)
    p_sus = 1 - p_inf                   
    return np.array([p_sus, p_inf])

def ti_star(status_nodes,T_BP):
    """Function to compute the ground truth vector of times of infection
    Args:
        status_nodes (array): Array of shape (T+1) x N contaning the ground-truth states of the system from t=0 to t=T
        T_BP (int): maximum time step considered by BP (T_BP <= T)
    Returns:
        ti (array): Array of size N containing the ground-truth vector of times of infection
    """
    N = len(status_nodes[0])
    ti = np.zeros(N)
    for i in range(N):
        t_inf = np.nonzero(status_nodes[:, i] == 1)[0]
        if len(t_inf) == 0:
            ti[i] = T_BP
        else:
            ti[i] = t_inf[0] - 1
    return ti

def ti_inferred(B):
    """Function to compute the MMSE-estimation of the times of infection
    Args: B (array): Array of shape N x (T+2) containing the beliefs estimated by BP
    Returns: ti_infer (array): Array of size N contaning the MMSE-estimation of the times of infection
    """
    ti_infer = np.array([np.array([(t - 1) * bt for t, bt in enumerate(b)]).sum() for b in B])
    return ti_infer

def ti_random(B):
    """Function to compute the RND-estimation of the times of infection
    Args: B (array): Array of shape N x (T+2) containing the beliefs estimated by BP
    Returns: ti_rnd (array): Array of size N contaning the RND-estimation of the times of infection
    """
    N = B.shape[0]
    T = len(B[0])-2
    b_mean = B.mean(axis=0)
    ti_rnd =  np.array([ t * b_mean[i]  for i,t  in enumerate(range(-1,T+1))]).sum()
    return np.full(N,ti_rnd)


def x_est_t(marginals, t=0):
    Mt = get_Mt(marginals, t)
    x_est = np.argmax(Mt, axis=0)   
    assert x_est.shape[0] == marginals.shape[0], "x_est should have the same number of nodes as marginals"
    return x_est


def OV(conf1, conf2):
    """Function to compute the overlap between two arrays of the same shape
    Args: conf1 (array): First array, conf2 (array): Second array
    Returns: ov (float): Overlap
    """
    # assertion to check that conf1 and conf2 have the same shape
    assert conf1.shape == conf2.shape, "Input arrays must have the same shape"
    ov = np.mean(conf1 == conf2)
    return ov


# def MOV(Mt):
#     """Function to compute the MMO-Mean overlap, given the array of marginals
#     Args:
#         Mt (array): Array of marginals
#     Returns:
#         mov (float): MMO-Mean overlap
#     """
#     M0 = np.maximum(Mt[0],Mt[1])
#     mov = np.mean(np.maximum(M0, Mt[2]))
#     return mov

def MOV(Mt):
    # Mt shape: (2, N)
    M0 = np.maximum(Mt[0], Mt[1])
    mov = np.mean(M0)
    return mov

def OV_rnd(conf, Mt):
    """Function to compute the RND-overlap, given the array of marginals
    Args: conf (array): Array of configurations Mt (array): Array of marginals
    Returns: mov_rnd (float): RND-overlap
    """
    x = np.argmax(np.mean(Mt, axis=1))
    ov_rnd = np.mean(conf == x)
    return ov_rnd

def MOV_rnd(Mt):
    """Function to compute the RND-Mean overlap, given the array of marginals
    Args: Mt (array): Array of marginals
    Returns: mov_rnd (float): RND-Mean overlap
    """
    m1 = np.maximum(np.mean(Mt[0]), np.mean(Mt[1]))
    mov_rnd = np.maximum(m1, np.mean(Mt[2]))
    return mov_rnd


def SE(ti_star, ti_inferred):
    """Function to compute the SE, given the ground truth vector of times of infection and an estimation of it
    Args:
        ti_star (array): Array of size N containing the ground-truth vector of times of infection
        ti_inferred (array):  Array of size N contaning an estimation of the times of infection
    Returns: se (float): SE
    """
    se = np.array([(t - ti_inferred[i]) ** 2 for i, t in enumerate(ti_star)]).mean()
    return se


def MSE(B, ti_inferred):
    """Function to compute the MSE, given the ground truth vector of times of infection and an estimation of it

    Args:
        B (array): Array of shape N x (T+2) containing the beliefs estimated by BP
        ti_inferred (array):  Array of size N contaning an estimation of the times of infection

    Returns:
        mse (float): MSE
    """
    mse = np.array(
        [
            np.array([b * (ti - (t - 1)) ** 2 for t, b in enumerate(B[i])]).sum()
            for i, ti in enumerate(ti_inferred)
        ]
    ).mean()
    return mse


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


## OLD
# def constrained_MOV_alt(Mt, delta, N, marginals=None, alpha=0.2):
#     """
#     MOV with constrained number of predicted infections = delta*N.

#     Uses time-aware scores ONLY for ranking (not for probability/confidence).
#     Falls back to original MOV when marginals is None.
#     """
#     k = int(delta * N)
#     if marginals is None:
#         p_inf = Mt[1]  # standard marginal infection probability
#     else:
#         p_base = Mt[1]  # keep true probabilistic meaning
#         time_scores = time_score_from_b(marginals)
#         time_scores = (time_scores - time_scores.min()) / (time_scores.max() - time_scores.min() + 1e-12)
#         p_inf = p_base + alpha * time_scores
#     # ---------- top-k selection ----------
#     top_k = np.argsort(-p_inf)[:k]
#     x_est = np.zeros(N, dtype=int)
#     x_est[top_k] = 1
#     # ---------- evaluation uses TRUE probability only ----------
#     conf = np.where(x_est == 1, Mt[1], 1 - Mt[1])
#     return np.mean(conf)

# def ov_mimic_metric_alt(Mt, delta, N, current_s_nb, marginals=None):
#     # w = soft_topk_weights(Mt, delta, N)
#     # p = Mt[1]

#     # # expected correctness under soft assignment
#     # conf = np.mean(w * p + (1 - w) * (1 - p))

#     # return conf
#     return constrained_MOV(Mt, delta, N, marginals)
#     # if current_s_nb/N < delta:
#     #     return constrained_MOV_weighted(Mt, delta, N)
#     # else: 
#     #     return ov_mimic_metric_old(Mt, delta, N)


# ### Previous attempts:


# def constrained_MOV_alt(Mt, delta, N):
#     """
#     MOV with constrained number of predicted infections.
#     """
#     p_inf = Mt[1]
#     k = int(delta * N)
#     top_k = np.argsort(-p_inf)[:k]
#     x_est = np.zeros(N, dtype=int)
#     x_est[top_k] = 1
#     # confidence = p_inf for predicted I, 1-p_inf for predicted S
#     conf = np.where(x_est == 1, p_inf, 1 - p_inf)
#     return np.mean(conf)

# def constrained_MOV_soft(Mt, delta, N, lam=1.0):
#     p_inf = Mt[1]
#     mov = np.mean(np.maximum(Mt[0], Mt[1]))
    
#     # normalize penalty to [0,1] range — max possible deviation is delta or (1-delta)
#     max_deviation = max(delta, 1 - delta)
#     prevalence_penalty = ((np.mean(p_inf) - delta) / max_deviation) ** 2
    
#     return mov - lam * prevalence_penalty

# def constrained_MOV_gaussian(Mt, delta, N, n_samples=10):
#     """
#     Soft constrained MOV averaged over k ~ N(delta*N, delta*(1-delta)*N).
#     """
#     p_inf = Mt[1]
#     mu = delta * N
#     sigma = np.sqrt(delta * (1 - delta) * N)
    
#     # sample k values from the distribution
#     k_samples = np.random.normal(mu, sigma, n_samples)
#     k_samples = np.clip(k_samples, 1, N-1).astype(int)
    
#     scores = []
#     for k in k_samples:
#         top_k = np.argsort(-p_inf)[:k]
#         x_est = np.zeros(N, dtype=int)
#         x_est[top_k] = 1
#         conf = np.where(x_est == 1, p_inf, 1 - p_inf)
#         scores.append(np.mean(conf))
    
#     return np.mean(scores)

# def constrained_MOV_weighted(Mt, delta, N):
#     """
#     Constrained MOV with confidence weighted by p_inf rank.
#     Top-k nodes predicted as I, but weighted by their p_inf.
#     """
#     p_inf = Mt[1]
#     k = int(delta * N)
    
#     # hard top-k assignment
#     top_k = np.argsort(-p_inf)[:k]
#     x_est = np.zeros(N, dtype=int)
#     x_est[top_k] = 1
    
#     # for predicted I: weight by p_inf (high confidence = high weight)
#     # for predicted S: weight by 1-p_inf (high confidence = high weight)
#     conf = np.where(x_est == 1, p_inf ** 2, (1 - p_inf) ** 2)
    
#     return np.mean(conf)

# def ov_mimic_metric_old(Mt, delta, N):
#     p_inf = Mt[1]
#     sorted_p = np.sort(p_inf)[::-1]
    
#     # 1. We expect k to be around delta * N, with a 'swing' 
#     # determined by the binomial standard deviation.
#     expected_k = delta * N
#     std_k = np.sqrt(N * delta * (1 - delta))
    
#     # 2. Search for a cutoff k* that 'makes sense'
#     # We look in the range [expected_k - 2*std, expected_k + 2*std]
#     search_min = max(1, int(expected_k - 2*std_k))
#     search_max = min(N-1, int(expected_k + 2*std_k))
    
#     # We pick the k* that maximizes the GAP (the 'cliff')
#     # This is the point where the model says: "The infection likely ends here."
#     gaps = sorted_p[search_min:search_max] - sorted_p[search_min+1:search_max+1]
#     k_star = np.argmax(gaps) + search_min
    
#     # 3. Informative Score
#     # How confident are we in this specific cluster of size k*?
#     conf_I = np.mean(sorted_p[:k_star])
#     conf_S = np.mean(1 - sorted_p[k_star:])
    
#     # 4. The H0 Penalty (Likelihood)
#     # Penalize the choice of k* if it strays too far from delta * N
#     # This keeps the 'dynamic' choice grounded in your prior.
#     z_score = (k_star - expected_k) / std_k
#     h0_penalty = 0.5 * (z_score**2)
    
#     return (0.5 * conf_I + 0.5 * conf_S) - (0.1 * h0_penalty)


# def clustering_confidence_metric(Mt, delta, N):
#     p = Mt[1]
#     k = int(delta * N)

#     # sort probabilities
#     idx = np.argsort(-p)
#     p_sorted = p[idx]

#     # --- 1. define clusters (top-k vs rest) ---
#     p_I = p_sorted[:k]
#     p_S = p_sorted[k:]

#     # --- 2. intra-cluster confidence ---
#     conf_I = np.mean(p_I) if k > 0 else 0.0
#     conf_S = np.mean(1 - p_S) if k < N else 0.0
#     intra_conf = 0.5 * (conf_I + conf_S)

#     # --- 3. inter-cluster separation ---
#     if k > 0 and k < N:
#         sep = np.mean(p_I) - np.mean(p_S)
#     else:
#         sep = 0.0

#     # --- 4. boundary sharpness ---
#     if 0 < k < N:
#         gap = p_sorted[k-1] - p_sorted[k]
#     else:
#         gap = 0.0

#     # --- 5. entropy penalty (uncertainty inside clusters) ---
#     eps = 1e-9
#     entropy = -np.mean(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
#     sharpness = 1 - entropy  # higher = better

#     return intra_conf + 0.5 * sep + 0.5 * gap + 0.2 * sharpness


# def soft_topk_weights(Mt, delta, N):
#     p = Mt[1]
#     k = int(delta * N)
#     tau = 0.05 * N  # temperature for softness, scaled with N
#     idx = np.argsort(-p)
#     ranks = np.arange(len(p))

#     # soft cutoff around k
#     w = 1 / (1 + np.exp((ranks - k) / tau))
#     w = w[idx]

#     return w


# def time_score_from_b(marginals, decay=0.3):
#     """
#     marginals: (N, T+2)
#         - last time index is t = inf (absorbing state)

#     returns: (N,)
#     """

#     marginals = np.asarray(marginals)  # (N, T+2)

#     N, T_full = marginals.shape
#     T = T_full - 1  # exclude inf state

#     t = np.arange(T)

#     weights = np.exp(-decay * t)
#     weights = weights / weights.sum()

#     # weighted sum over time axis (axis=1)
#     # result: (N,)
#     return np.sum(marginals[:, :T] * weights[None, :], axis=1)

# def constrained_MOV_final(Mt, delta, N, marginals):
#     """
#     MOV with constrained number of predicted infections = delta*N.
#     Best ground-truth-free proxy for OV found so far.
#     """
#     if marginals is not None:
#         time_scores = time_score_from_b(marginals)
#         # reweight p_inf by time scores to prioritize early infections
#         p_inf = time_scores
#     else:
#         p_inf = Mt[1]
#     k = int(delta * N)
#     top_k = np.argsort(-p_inf)[:k]
#     x_est = np.zeros(N, dtype=int)
#     x_est[top_k] = 1
#     conf = np.where(x_est == 1, p_inf, 1 - p_inf)
#     return np.mean(conf)