
import numpy as np


###--------------------- BAYESIAN INFERENCE METRICS ----------------------###


def compute_measures(marginals, s0, delta, status_nodes, x_rnd, Mt_rnd):
    """
    Receives: marginals: array of shape (N, T+2) containing the beliefs estimated by BP, s0: array of shape (N,) containing the ground truth initial states, delta: initial probability of being infected, status_nodes: array of shape (T+1, N) containing the ground truth states of the system from t=0 to t=T
    Returns: a dictionary containing the computed measures
    """
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
    return measures, x_est


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