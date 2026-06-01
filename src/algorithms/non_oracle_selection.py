import numpy as np
import networkx as nx
from src.utils.metrics import *
from src.helpers.pipeline_helpers import get_Mt
from tqdm import tqdm
from collections import defaultdict
from math import comb
from src.helpers.algo_helpers import update_cand_obs, build_obs
# methods that do not implicitely use info of status nodes for selection

#-----------------
# 1. Entropy based selection
#-----------------


def compute_si_backscore(G, candidates, observations, marginals, lam):
    """
    observations: array of (node, state, time); we only use (node, time)
    """

    # build shortest path distances from ALL candidates at once
    dist_map = {}
    for i in candidates:
        dist_map[i] = nx.single_source_shortest_path_length(G, i)

    Mt = get_Mt(marginals, t=0)
    p_inf = Mt[1]   # P(node infected at t=0)

    beta = 10.0
    scores = {i: 0.0 for i in candidates}

    for (j, _, t_j) in observations:

        # --- compute raw explanation weights for THIS observation ---
        local_weights = {}

        for i in candidates:
            d = dist_map[i].get(j, np.inf)

            if d <= t_j:
                #w = (lam ** d) * ((1 - lam) ** (t_j - d))
                w = comb(t_j - 1, d - 1) * (lam ** d) * ((1 - lam) ** (t_j - d))
                local_weights[i] = w

        if len(local_weights) == 0:
            continue

        # --- local softmax normalization ---
        vals = np.array(list(local_weights.values()))
        vals = vals - vals.max()

        exp_vals = np.exp(beta * vals)
        Z = exp_vals.sum() + 1e-12

        soft_weights = exp_vals / Z

        # --- multiplicative BP modulation (your requested change) ---
        for idx, i in enumerate(local_weights.keys()):
            scores[i] += p_inf[i] * soft_weights[idx]

    return scores

def select_best_candidate_reverse_si(G, candidates, current_obs, marginals, lam, alpha=0.5):
    """
    Returns argmax source under backward SI likelihood approximation.
    """

    scores = compute_si_backscore(
        G=G,
        candidates=candidates,
        observations=current_obs,
        marginals=marginals,
        lam=lam
    )

    return max(scores, key=scores.get)


def path_weight_sensor_selection(bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta, logger=None, G=None, alpha=0.5, beta=0.3, gamma=0.5):
    """
    Fast sensor selection using ONE BP run + entropy scoring.
    """
    target = int(rho_max * bp_base.size)
    sensor_set = set()
    sensor_order = []
    current_obs = np.empty((0, 3), dtype=int)
    # --- run BP once ---
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    marginals = bp_base.marginals()

    for k in tqdm(range(target + 1)):
        remaining = list(set(range(bp_base.size)) - sensor_set)
        if len(remaining) == 0:
            break
        # --- candidate scoring (NO BP reruns!) ---
        #marginals = bp_base.marginals()
        #Mt = get_Mt(marginals, t=0)
        #saved_messages = torch.clone(bp_base.messages.values)
        best_candidate = select_best_candidate_reverse_si(G=G, candidates=remaining, current_obs=current_obs, marginals=marginals, lam=0.3) #select_sensor_light_cone(G, candidates= remaining, observed_nodes=set(current_obs[:, 0].tolist()), current_obs=current_obs, lam=0.3, max_depth=5, eps=1e-12)
        sensor_set.add(best_candidate)
        sensor_order.append(best_candidate)
        # --- update obs---
        T_max = bp_base.time
        current_obs = update_cand_obs(bp_base, best_candidate, status_nodes, current_obs, T_max=T_max)
        warm_iter=50
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=0.1*tol, damp=damp)
        marg = bp_base.marginals()
        cmov = mov_constrained_metric(marg, delta=delta)
        print("Added candidate:", best_candidate, ", true initial state is:", status_nodes[0, best_candidate])

        if k < 5 or k % 20 == 0:
            overlap = OV(np.argmax(get_Mt(marg, t=0), axis=0), status_nodes[0])
            print(f"Overlap after adding candidate: {overlap:.4f}, error: {errors[-1]:.4f}, BP iters: {n_iter}, rho ={len(sensor_set)/bp_base.size:.2f}")

        if k/bp_base.size == delta:
            print(f"Reached delta={delta:.2f} at k={k}, current overlap: {overlap:.4f}, proportion of true sources in selected set: {np.sum(status_nodes[0, list(sensor_set)]) / len(sensor_set):.4f}")
        #k += 1

    return sensor_order


def max_pinf_selection(bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta, logger=None, G=None, alpha=0.5, beta=0.3, gamma=0.5):
    """
    Fast sensor selection using ONE BP run + pinf scoring.
    """
    target = int(rho_max * bp_base.size)
    sensor_set = set()
    sensor_order = []
    current_obs = np.empty((0, 3), dtype=int)
    # --- run BP once ---
    bp_base.update(maxit=max_iter, tol=tol, damp=damp)
    marginals = bp_base.marginals()

    for k in tqdm(range(target + 1)):
        remaining = list(set(range(bp_base.size)) - sensor_set)
        if len(remaining) == 0:
            break
        # --- candidate scoring (NO BP reruns!) ---
        marginals = bp_base.marginals()
        Mt = get_Mt(marginals, t=0)
        p_inf = Mt[1]
        #saved_messages = torch.clone(bp_base.messages.values)
        best_candidate = remaining[np.argmax(p_inf[remaining])]
        sensor_set.add(best_candidate)
        sensor_order.append(best_candidate)
        # --- update obs---
        T_max = bp_base.time
        current_obs = update_cand_obs(bp_base, best_candidate, status_nodes, current_obs, T_max=T_max)
        warm_iter=50
        n_iter, errors = bp_base.update(maxit=warm_iter, tol=0.1*tol, damp=damp)
        marg = bp_base.marginals()
        cmov = mov_constrained_metric(marg, delta=delta)
        print("Added candidate:", best_candidate, ", true initial state is:", status_nodes[0, best_candidate])

        if k < 5 or k % 20 == 0:
            overlap = OV(np.argmax(get_Mt(marg, t=0), axis=0), status_nodes[0])
            print(f"Overlap after adding candidate: {overlap:.4f}, error: {errors[-1]:.4f}, BP iters: {n_iter}, rho ={len(sensor_set)/bp_base.size:.2f}")

        if k/bp_base.size == delta:
            print(f"Reached delta={delta:.2f} at k={k}, current overlap: {overlap:.4f}, proportion of true sources in selected set: {np.sum(status_nodes[0, list(sensor_set)]) / len(sensor_set):.4f}")
        #k += 1

    return sensor_order

def max_entropy_selection(bp_base, status_nodes, rho_max, m, max_iter, tol, damp, delta, logger=None, G=None, alpha=0.5, beta=0.3, gamma=0.5):
    """
    Sensor selection using entropy over infection time distribution.
    """
    target = int(rho_max * bp_base.size)
    sensor_set = set()
    sensor_order = []
    current_obs = np.empty((0, 3), dtype=int)

    bp_base.update(maxit=max_iter, tol=tol, damp=damp)

    for k in tqdm(range(target + 1)):
        remaining = list(set(range(bp_base.size)) - sensor_set)
        if len(remaining) == 0:
            break

        marginals = bp_base.marginals()  # shape (N, T+2)

        # --- entropy over infection time distribution ---
        # p = marginals  # p[i, t] = p_inf(node i, time t)
        # p_clipped = np.clip(p, 1e-10, 1)  # avoid log(0)
        # entropy = -np.sum(p_clipped * np.log(p_clipped), axis=1)  # shape (N,)

        p = marginals[:, 0]  # p_inf at t=0, shape (N,)
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        entropy = -(p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped))  # shape (N,)
        best_candidate = remaining[np.argmax(entropy[remaining])]

        best_candidate = remaining[np.argmax(entropy[remaining])]
        sensor_set.add(best_candidate)
        sensor_order.append(best_candidate)

        T_max = bp_base.time
        current_obs = update_cand_obs(bp_base, best_candidate, status_nodes, current_obs, T_max=T_max)
        n_iter, errors = bp_base.update(maxit=50, tol=0.1*tol, damp=damp)
        marg = bp_base.marginals()

        print("Added candidate:", best_candidate, ", true initial state is:", status_nodes[0, best_candidate])

        if k < 5 or k % 20 == 0:
            overlap = OV(np.argmax(get_Mt(marg, t=0), axis=0), status_nodes[0])
            print(f"Overlap: {overlap:.4f}, error: {errors[-1]:.4f}, BP iters: {n_iter}, rho={len(sensor_set)/bp_base.size:.2f}")

        if k/bp_base.size == delta:
            print(f"Reached delta={delta:.2f} at k={k}, overlap: {overlap:.4f}")

    return sensor_order




## Light cone

# import networkx as nx
# from collections import deque, defaultdict

# def backward_light_cone(G, source_node, t_obs, lam=0.3, max_depth=5):

#     N = G.number_of_nodes()
#     weights = np.zeros(N)

#     q = deque([(source_node, 0, 1.0)])

#     visited_mass = {}  # track accumulated probability per node

#     while q:
#         node, depth, prob = q.popleft()

#         if depth >= max_depth:
#             continue

#         for neigh in G.neighbors(node):

#             new_prob = prob * lam * (1 - lam) ** depth

#             weights[neigh] += new_prob

#             visited_mass[neigh] = visited_mass.get(neigh, 0.0) + new_prob

#             q.append((neigh, depth + 1, new_prob))

#     total = weights.sum()

#     if total > 0:
#         weights /= total

#     # ---------- diagnostics ----------
#     support = np.sum(weights > 1e-6)
#     entropy = -np.sum(weights * np.log(weights + 1e-12))
#     max_w = np.max(weights)
#     top_node = np.argmax(weights)

#     # print("\n[LIGHT CONE DEBUG]")
#     # print(f"node={source_node}  t_obs={t_obs}")
#     # print(f"  total_mass={total:.4f}")
#     # print(f"  support_size={support}/{N}")
#     # print(f"  entropy={entropy:.4f}")
#     # print(f"  max_weight={max_w:.4f} (node {top_node})")

#     # sanity flags
#     if support > N * 0.8:
#         print("  ⚠️ WARNING: cone too diffuse (almost uniform)")
#     if entropy > np.log(N) * 0.9:
#         print("  ⚠️ WARNING: near-max entropy (random-like)")
#     if max_w < 0.05:
#         print("  ⚠️ WARNING: no dominant causal region")

#     return weights

# def infection_times_from_obs(obs, cand):
#     #print("Cand obs: ", obs[obs[:, 0] == cand]) # this is a list of a list ex: [[0, 0, 1, 0, 0...]]
#     infected = obs[(obs[:,0] == cand) & (obs[:,1] == 1)]

#     return infected[0,2] if len(infected) else obs[:,2].max() + 1

# # def infection_times_from_obs(obs, cand):
# #     # ensure consistent dtype comparison
# #     infected = obs[(obs[:, 0] == cand) & (obs[:, 1] == 1)]
# #     return infected[0, 2] if len(infected) else obs[:, 2].max() + 1

# def compute_global_light_cone_score(G, observed_nodes, current_obs, lam=0.3, max_depth=5):

#     N = G.number_of_nodes()
#     global_score = np.zeros(N)

#     # print("\n==============================")
#     # print(" LIGHT CONE GLOBAL DEBUG")
#     # print("==============================")

#     #observed_nodes = set(current_obs[current_obs[:, 1] == 1, 0].tolist())


#     for i in observed_nodes:

#         t_i = infection_times_from_obs(current_obs, i)
#         #print("Infection time for node", i, "is", t_i)

#         #print(f"\n--- SENSOR {i} (t={t_i}) ---")

#         weights = backward_light_cone(G, i, t_i, lam, max_depth)

#         nonzero = np.sum(weights > 1e-4)
#         top5 = np.argsort(-weights)[:5]

#         # print(f"  top-5 candidates: {top5}")
#         # print(f"  nonzero nodes: {nonzero}")
#         # print(f"  weight mass check: {weights.sum():.4f}")

#         time_weight = 1.0 / (t_i + 1)
#         global_score += time_weight * weights

#     #print("\n==============================\n")

#     return global_score

# def select_sensor_light_cone(G, candidates, observed_nodes, current_obs, lam=0.3, max_depth=5, eps=1e-12):
#     """
#     Select next sensor by maximizing reduction in uncertainty
#     of backward light-cone source distribution.
#     """

#     if current_obs.size == 0:
#         print("No current observations, selecting random candidate.")
#         return np.random.choice(candidates)
#     # else:
#     #     print("Already observed ", len(observed_nodes), "nodes. Evaluating candidates with light cone scoring.")

#     # --- base belief ---
#     base_score = compute_global_light_cone_score(G, observed_nodes, current_obs, lam=lam, max_depth=max_depth)
#     base_score = np.clip(base_score, eps, 1.0)
#     base_H = -np.sum(base_score * np.log(base_score))

#     best_node = None
#     best_gain = -np.inf

#     # print("\n[SENSOR SELECTION DEBUG]")

#     for c in candidates:
#         # simulate adding candidate as observed sensor
#         new_obs = observed_nodes | {c}  # add c to set of observed nodes
#         new_score = compute_global_light_cone_score(G, new_obs, current_obs, lam=lam, max_depth=max_depth)
#         new_score = np.clip(new_score, eps, 1.0)
#         new_score /= new_score.sum()

#         new_H = -np.sum(new_score * np.log(new_score))

#         gain = base_H - new_H

#         #print(f"candidate={c:4d} | gain={gain:.6f}")

#         if gain > best_gain:
#             best_gain = gain
#             best_node = c

#     print(f"\nSELECTED SENSOR: {best_node} | gain={best_gain:.6f}\n")
#     return best_node

# ## EARLY MASS

# def early_mass(p_i, tau=2):
#     return np.sum(p_i[:tau+1])

# def temporal_variance(p_i):
#     t = np.arange(len(p_i))
#     mean = np.sum(t * p_i)
#     var = np.sum(p_i * (t - mean) ** 2)
#     return var

# def score_node(i, marginals, neighbors_i, tau=2, alpha=1.0, beta=1.0, gamma=0.5):

#     p_i = marginals[i]

#     # 1. early infection mass
#     e_i = early_mass(p_i, tau)

#     # 2. neighborhood contrast
#     if len(neighbors_i) > 0:
#         e_neighbors = np.mean([early_mass(marginals[j], tau) for j in neighbors_i])
#     else:
#         e_neighbors = 0.0

#     contrast = e_i - e_neighbors

#     # 3. temporal sharpness
#     sharpness = -temporal_variance(p_i)

#     return alpha * e_i + beta * contrast + gamma * sharpness


# def select_best_candidate(candidates, marginals, G):

#     scores = {}

#     for i in candidates:
#         neighbors_i = list(G.neighbors(i))
#         scores[i] = score_node(i, marginals, neighbors_i)

#     return max(scores, key=scores.get) #, scores

# # ## KL_DIV

# # def source_template(T):
# #     q = np.zeros(T)
# #     q[0] = 1.0
# #     return q

# # def neighbor_template(T, lam):
# #     q = np.zeros(T)
# #     for t in range(1, T):
# #         q[t] = lam * (1 - lam) ** (t - 1)
# #     return q

# # def kl(p, q, eps=1e-12):
# #     p = np.clip(p, eps, 1)
# #     q = np.clip(q, eps, 1)
# #     return np.sum(p * np.log(p / q))

# # def score_node(i, marginals, neighbors, lam=0.3, alpha=1.0, beta=1.0):

# #     T = marginals.shape[1]

# #     # self fit
# #     p_i = marginals[i]
# #     q_src = source_template(T)
# #     self_score = -kl(p_i, q_src)

# #     # neighbor fit
# #     q_nbr = neighbor_template(T, lam)
# #     neigh_score = 0.0
# #     for j in neighbors:
# #         p_j = marginals[j]
# #         neigh_score += -kl(p_j, q_nbr)

# #     neigh_score /= max(len(neighbors), 1)

# #     return alpha * self_score + beta * neigh_score

# # ## CMOV for t=0 vs t>0

# # def metric_obs_t(candidate, time, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20):
# #     """
# #     Hypothesize candidate as source, measure entropy reduction.
# #     No ground truth needed.
# #     """
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     # add fake obs that candidate is infected at time t, see how much entropy reduces compared to no obs
# #     if time == 0:
# #         fake_obs = np.array([[candidate, 1, 0]], dtype=int)
# #     else:
# #         fake_obs = np.array([[candidate, 0, time-1], [candidate, 1, time]], dtype=int)
# #     combined = np.vstack([current_obs, fake_obs]) if current_obs.size else fake_obs
# #     bp_base.reset_obs(combined)
# #     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
# #     marg = bp_base.marginals()
    
# #     cmov_obs_t = mov_constrained_metric(marg, delta=bp_base.delta)
# #     # reset to clean state
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     bp_base.reset_obs(current_obs)
# #     return cmov_obs_t

# # def expected_gain(candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20):

# #     bp_base.messages.values = torch.clone(saved_messages)
# #     base_marg = bp_base.marginals()
# #     probs = base_marg[candidate]
# #     T = bp_base.time
# #     expected = 0.0
# #     max_metric = 0.0

# #     for t in range(T):
# #         # calc metric for obs of candidate infected at time t
# #         cmov_obs_t = metric_obs_t(candidate, t, bp_base, saved_messages, current_obs, tol, damp, warm_iter)
# #         # S(i) = U(ti=0) - <U(ti>0)> 
# #         if t == 0:
# #             # term U(ti=0)
# #             expected += cmov_obs_t #* probs[t]
# #         else:
# #             # term <U(ti>0)>
# #             #expected -= cmov_obs_t * probs[t]
# #             cand_metric = cmov_obs_t #* probs[t]
# #             if cand_metric > max_metric:
# #                 max_metric = cand_metric

# #     return expected - max_metric

# # def select_best_candidate(candidates, G, bp_base, saved_messages, current_obs, delta, tol, damp, warm_iter=20):
# #     """
# #     counterfactual BP to pick best.
# #     """
# #     # base CMO before any candidate
# #     base_score = mov_constrained_metric(bp_base.marginals(), delta=delta)

# #     # counterfactual BP for each candidate
# #     scores = {}
# #     for candidate in candidates:
# #         neighbors = list(G.neighbors(candidate))
# #         #score = evaluate_candidate_counterfactual(candidate, bp_base, status_nodes, saved_messages, current_obs, delta, tol, damp, warm_iter)
# #         score = score_node(candidate, bp_base.marginals(), neighbors, lam=0.3, alpha=1.0, beta=1.0) #expected_gain(candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter)
# #         scores[candidate] = score #- base_score

# #     # reset bp to saved state
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     bp_base.reset_obs(current_obs)

# #     print("Best candidate vs avge gain:", max(scores, key=scores.get), "with gain", scores[max(scores, key=scores.get)], "average gain among candidates:", np.mean(list(scores.values())))
# #     return max(scores, key=scores.get)


# # def evaluate_candidate_counterfactual(candidate, bp_base, status_nodes, saved_messages, current_obs, delta, tol, damp, warm_iter=20):
# #     """
# #     Hypothesize candidate as source, measure CMO gain.
# #     No ground truth needed.
# #     """
# #     bp_base.messages.values = torch.clone(saved_messages)
    
# #     # inject source hypothesis: infected at t=0
# #     fake_obs = np.array([[candidate, 1, 0]], dtype=int)
# #     combined_obs = np.vstack([current_obs, fake_obs]) if current_obs.size else fake_obs
# #     bp_base.reset_obs(combined_obs)
# #     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
    
# #     marg = bp_base.marginals()
# #     return ov_metric(marg, status_nodes) #mov_constrained_metric(marg, delta=delta)

# # def entropy_gain_t(candidate, time, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20):
# #     """
# #     Hypothesize candidate as source, measure entropy reduction.
# #     No ground truth needed.
# #     """
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     # add fake obs that candidate is infected at time t, see how much entropy reduces compared to no obs
# #     if time == 0:
# #         fake_obs = np.array([[candidate, 1, 0]], dtype=int)
# #     else:
# #         fake_obs = np.array([[candidate, 0, time-1], [candidate, 1, time]], dtype=int)
# #     combined = np.vstack([current_obs, fake_obs]) if current_obs.size else fake_obs
# #     bp_base.reset_obs(combined)
# #     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
    
# #     marg = bp_base.marginals()
# #     H_after = compute_tau_entropy(marg).sum()  # total entropy after adding candidate
# #     Mt = get_Mt(marg, t=0)
# #     p_inf = Mt[1]
# #     N0_after = np.sum(p_inf)
# #     N_exp = bp_base.size * bp_base.delta
    
# #     # reset to clean state
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     bp_base.reset_obs(current_obs)
# #     penalty = (N_exp - N0_after) ** 2
# #     return H_after, penalty

# # def expected_entropy_gain(candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20, lambda_mass=1.0):

# #     bp_base.messages.values = torch.clone(saved_messages)
# #     base_marg = bp_base.marginals()
# #     probs = base_marg[candidate]
# #     H_before = compute_tau_entropy(base_marg).sum()
# #     T = bp_base.time
# #     expected = 0.0

# #     for t in range(T):
# #         p_t = probs[t]
# #         if p_t < 1e-8:
# #             continue

# #         H_after, penalty = entropy_gain_t(candidate, t, bp_base, saved_messages, current_obs, tol, damp, warm_iter)
# #         gain_t = (H_before - H_after) - lambda_mass * penalty

# #         expected += p_t * gain_t

# #     return expected

# # def select_best_candidate(candidates, bp_base, saved_messages, current_obs, delta, tol, damp, warm_iter=20):
# #     """
# #     counterfactual BP to pick best.
# #     """
# #     # base CMO before any candidate
# #     base_score = mov_constrained_metric(bp_base.marginals(), delta=delta)

# #     # counterfactual BP for each candidate
# #     cf_gains = {}
# #     for candidate in candidates:
# #         #score = evaluate_candidate_counterfactual(candidate, bp_base, status_nodes, saved_messages, current_obs, delta, tol, damp, warm_iter)
# #         score = expected_entropy_gain(candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter, lambda_mass=1.0)
# #         cf_gains[candidate] = score #- base_score

# #     # reset bp to saved state
# #     bp_base.messages.values = torch.clone(saved_messages)
# #     bp_base.reset_obs(current_obs)

# #     print("Best candidate vs avge gain:", max(cf_gains, key=cf_gains.get), "with gain", cf_gains[max(cf_gains, key=cf_gains.get)], "average gain among candidates:", np.mean(list(cf_gains.values())))
# #     return max(cf_gains, key=cf_gains.get)

# # def build_obs(subset, status_nodes):
# #     obs_rows = []
# #     for node in subset:
# #         if node is None:
# #             continue
# #         for t in range(status_nodes.shape[0]):
# #             # ensure status_nodes[t, node] is int (0 or 1) for obs array
# #             val = status_nodes[t, node]
# #             if isinstance(val, np.ndarray):
# #                 print(status_nodes.shape)
# #                 print("ARRAY FOUND:", val, val.shape, type(val))
# #                 print("found for node", node, "at time", t)
# #             obs_rows.append((node, int(status_nodes[t, node]), t))
# #     return np.array(obs_rows, dtype=int) if obs_rows else np.empty((0, 3), dtype=int)

# def adaptive_score(marg, Mt, G, alpha=0.5, decay=0.3):
#     N = marg.shape[0]
    
#     p_source = Mt[1]                          # P(source) — direct signal
#     time_scores = time_score_from_b(marg, decay)
    
#     # entropy over infection time — high = BP is uncertain about this node
#     H = -np.sum(marg * np.log(marg + 1e-12), axis=1)  # (N,)
#     H /= H.max() + 1e-12
    
#     # current global confidence — how peaked is the posterior overall?
#     # when this is high, we trust p_source more; when low, trust entropy more
#     global_confidence = 1 - H.mean()         # 0 = maximally uncertain, 1 = certain
    
#     score = (
#         global_confidence       * p_source      # late phase: confirm infected
#         + (1 - global_confidence) * H           # early phase: reduce uncertainty
#         + alpha                 * time_scores   # always: bias toward early infected
#     )
#     return score

# def entropy(p, eps=1e-12):
#     p = np.clip(p, eps, 1.0)
#     return -np.sum(p * np.log(p))


# def compute_tau_entropy(marginals):
#     """
#     marginals: (N, T) or (N, T+2) infection-time probabilities
#     returns: entropy per node
#     """
#     H = np.zeros(marginals.shape[0])
#     for i in range(marginals.shape[0]):
#         H[i] = entropy(marginals[i])
#     return H

# def early_bias(marginals, gamma=0.5):
#     """
#     favors nodes with early infection probability
#     """
#     T = marginals.shape[1]
#     t = np.arange(T)
#     weights = np.exp(-gamma * t)

#     return marginals @ weights


# def neighbor_entropy(graph, H_nodes):
#     """
#     aggregates uncertainty in neighborhood
#     """
#     neigh_H = np.zeros_like(H_nodes)
#     for i in range(len(H_nodes)):
#         neigh = list(graph.neighbors(i))
#         if len(neigh) > 0:
#             neigh_H[i] = np.mean(H_nodes[neigh])
#         else:
#             neigh_H[i] = 0.0
#     return neigh_H

# def evaluate_candidate_loglik(candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20):
#     """
#     Hypothesize candidate as source, measure log-likelihood gain.
#     No ground truth needed.
#     """
#     bp_base.messages.values = torch.clone(saved_messages)
    
#     fake_obs = np.array([[candidate, 1, 0]], dtype=int)
#     combined = np.vstack([current_obs, fake_obs]) if current_obs.size else fake_obs
#     bp_base.reset_obs(combined)
#     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
    
#     return bp_base.loglikelihood().item()


# def select_best_candidate_loglik(candidates, bp_base, saved_messages, current_obs, tol, damp, warm_iter=20):
#     ll_scores = {}
#     for candidate in candidates:
#         ll_scores[candidate] = evaluate_candidate_loglik(
#             candidate, bp_base, saved_messages, current_obs, tol, damp, warm_iter
#         )
    
#     # reset to clean state
#     bp_base.messages.values = torch.clone(saved_messages)
#     bp_base.reset_obs(current_obs)
    
#     # print mean and std of scores for debugging
#     scores = np.array(list(ll_scores.values()))
#     print(f"Log-likelihood scores for candidates: mean={scores.mean():.4f}, std={scores.std():.4f}")
#     return max(ll_scores, key=ll_scores.get)




# # KL DIV TO SEE WHICH OBS CHANGES THE MOST

# def evaluate_candidate_influence(
#     candidate,
#     bp_base,
#     saved_messages,
#     current_obs,
#     tol,
#     damp,
#     warm_iter=20,
#     metric="kl"   # or "ov"
# ):
#     """
#     Counterfactual BP influence:
#     compare posterior before vs after observing candidate.
#     """

#     # --- restore base state ---
#     bp_base.messages.values = torch.clone(saved_messages)

#     bp_base.reset_obs(current_obs)
#     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
#     base_marg = bp_base.marginals()

#     # --- counterfactual observation ---
#     fake_obs = np.array([[candidate, 1, 0]], dtype=int)
#     combined = np.vstack([current_obs, fake_obs]) if current_obs.size else fake_obs

#     bp_base.reset_obs(combined)
#     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)
#     new_marg = bp_base.marginals()

#     # --- compare distributions ---
#     if metric == "kl":
#         eps = 1e-12
#         base = base_marg + eps
#         new = new_marg + eps
#         kl = np.sum(base * np.log(base / new), axis=1)
#         return np.mean(kl)

#     elif metric == "ov":
#         x1 = np.argmax(get_Mt(base_marg, t=0), axis=0)
#         x2 = np.argmax(get_Mt(new_marg, t=0), axis=0)
#         return OV(x1, x2)

#     else:
#         raise ValueError("Unknown metric")


# def select_best_candidate_influence(
#     candidates,
#     bp_base,
#     saved_messages,
#     current_obs,
#     tol,
#     damp,
#     delta,              # <-- added (needed for CMOV)
#     warm_iter=20,
#     metric="kl"
# ):
#     scores = {}

#     # --- run base BP once (needed for CMOV term) ---
#     bp_base.messages.values = torch.clone(saved_messages)
#     bp_base.reset_obs(current_obs)
#     bp_base.update(maxit=warm_iter, tol=tol, damp=damp)

#     marginals = bp_base.marginals()
#     Mt = get_Mt(marginals, t=0)
#     p_inf = Mt[1]   # P(node infected at t=0)

#     # CMOV consistency term
#     cmov = 1.0 - np.abs(p_inf - delta)

#     for c in candidates:
#         influence = evaluate_candidate_influence(
#             c, bp_base, saved_messages, current_obs,
#             tol, damp, warm_iter, metric
#         )

#         # --- MULTIPLICATIVE CMOV MODULATION ---
#         scores[c] = influence * (1 + 0.3 *(cmov[c]-cmov.mean()))  # boost candidates that align with CMOV prior

#     return max(scores, key=scores.get)


# ## Light cone

# import numpy as np
# import networkx as nx
# from collections import defaultdict
# from math import comb

# # def compute_si_backscore(G, candidates, observations, marginals, lam, alpha=0.5):
# #     """
# #     observations: array of (node, state, time)
# #                  we only use (node, time)
# #     """

# #     # build shortest path distances from ALL candidates at once
# #     # (much faster than repeated BFS)
# #     dist_map = {}

# #     for i in candidates:
# #         dist_map[i] = nx.single_source_shortest_path_length(G, i)

# #     # scores = {}

# #     Mt = get_Mt(marginals, t=0)
# #     p_inf = Mt[1]   # P(node infected at t=0)

# #     # for i in candidates:
# #     #     s = 0.0
# #     #     dist_i = dist_map[i]

# #     #     for (j, _, t_j) in observations:
# #     #         d = dist_i.get(j, np.inf)

# #     #         if d <= t_j:
# #     #             s += (lam ** d) * ((1 - lam) ** (t_j - d))
# #     #             #s += d*np.log(lam) + (t_j - d)*np.log(1 - lam)

# #     #     scores[i] = s + alpha * np.log(p_inf[i] + 1e-12)

# #     # return scores
# #     # temperature for local competition
# #     beta = 10.0

# #     scores = {i: 0.0 for i in candidates}


# #     for (j, _, t_j) in observations:

# #         # --- compute raw explanation weights for THIS observation ---
# #         local_weights = {}

# #         for i in candidates:

# #             d = dist_map[i].get(j, np.inf)

# #             if d <= t_j:
# #                 w = (lam ** d) * ((1 - lam) ** (t_j - d))
# #                 local_weights[i] = w

# #         # skip impossible obs
# #         if len(local_weights) == 0:
# #             continue

# #         # --- local softmax normalization ---
# #         vals = np.array(list(local_weights.values()))

# #         # stability trick
# #         vals = vals - vals.max()

# #         exp_vals = np.exp(beta * vals)
# #         Z = exp_vals.sum() + 1e-12

# #         soft_weights = exp_vals / Z

# #         # --- accumulate competitive attribution ---
# #         for idx, i in enumerate(local_weights.keys()):
# #             scores[i] += soft_weights[idx]

# #     # weak BP prior ONLY
# #     for i in candidates:
# #         scores[i] += alpha * p_inf[i]

# #     return scores






# # def compute_si_backscore(G, candidates, observations, marginals, lam, alpha=0.5):

# #     dist_map = {
# #         i: nx.single_source_shortest_path_length(G, i)
# #         for i in candidates
# #     }

# #     Mt = get_Mt(marginals, t=0)
# #     p_inf = Mt[1]

# #     beta = 10.0
# #     eps = 1e-12

# #     # log-score for cross-observation coupling
# #     log_scores = {i: 0.0 for i in candidates}

# #     for (j, _, t_j) in observations:

# #         # ---------- compute ALL local weights once ----------
# #         local_weights = {}

# #         for i in candidates:
# #             d = dist_map[i].get(j, np.inf)

# #             if d <= t_j:
# #                 local_weights[i] = (lam ** d) * ((1 - lam) ** (t_j - d))

# #         if len(local_weights) == 0:
# #             continue

# #         # ---------- softmax over candidates ----------
# #         vals = np.array(list(local_weights.values()))
# #         vals = vals - vals.max()

# #         exp_vals = np.exp(beta * vals)
# #         Z = exp_vals.sum() + eps

# #         keys = list(local_weights.keys())
# #         probs = exp_vals / Z

# #         # ---------- cross-observation coupling ----------
# #         for idx, i in enumerate(keys):
# #             log_scores[i] += np.log(probs[idx] + eps)

# #     # ---------- final scoring ----------
# #     scores = {
# #         i: p_inf[i] * np.exp(log_scores[i])
# #         for i in candidates
# #     }

# #     return scores

# def select_farthest_node(candidates, selected, G):
#     """
#     Pick candidate maximally far from current selected set
#     using shortest-path distance.
#     """

#     if len(selected) == 0:
#         return np.random.choice(list(candidates))

#     # precompute shortest path lengths from selected nodes
#     max_min_dist = -1
#     best_node = None

#     for i in candidates:
#         # distance to closest selected node
#         min_dist = min(
#             nx.shortest_path_length(G, source=i, target=s)
#             for s in selected
#         )

#         if min_dist > max_min_dist:
#             max_min_dist = min_dist
#             best_node = i

#     return best_node




# def update_cand_obs(bp_base, candidate, status_nodes, current_obs):
#     candidate_rows = build_obs({candidate}, status_nodes)
#     candidate_obs = np.vstack([current_obs, candidate_rows]) if current_obs.size else candidate_rows
#     #print("shape of candidate obs:", candidate_obs.shape)
#     bp_base.reset_obs(candidate_obs)
#     #print("Updated BP observations with candidate:", candidate, "new obs shape:", bp_base.observations.shape)
#     return 

# def build_obs(subset, status_nodes):
#     obs_rows = []

#     T = status_nodes.shape[0]

#     for node in subset:
#         if node is None:
#             continue

#         # find first infection time
#         infected_times = np.where(status_nodes[:, node] == 1)[0]

#         if len(infected_times) > 0:
#             t_inf = infected_times[0]
#             obs_rows.append((node, 1, t_inf))
#             if t_inf > 0:
#                 obs_rows.append((node, 0, t_inf - 1))
#         else:
#             # optional: never infected
#             obs_rows.append((node, 0, T - 1))
#     #print("Built obs rows for candidate:", obs_rows)
#     #print("Shape of obs rows:", len(obs_rows), "x 3")
#     return np.array(obs_rows, dtype=int)
