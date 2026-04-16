from pyexpat import features

import numpy as np
from bpepi.Modules import fg_torch as fg #pytorch version
from sim_helpers import simulate_SI
from metrics import OV, get_Mt


class SensorSelectorREINFORCE:
    def __init__(self, N, rho, lr=0.01, baseline_decay=0.95, entropy_coef=0.01):
        self.N = N
        self.k = int(rho * N)
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.entropy_coef = entropy_coef

        self.w =  np.zeros(4) #np.zeros(N)
        self.baseline = 0.0

    # def get_probs(self):
    #     exp_w = np.exp(self.w - np.max(self.w))
    #     return exp_w / np.sum(exp_w)

    def sample_subset(self, probs):
        return np.random.choice(self.N, size=self.k, replace=False, p=probs)

    # def _grad_logp(self, subset, probs):
    #     grad = -probs.copy()
    #     grad[subset] += 1.0
    #     return grad
    
    def _grad_logp(self, subset, probs, features):
        grad = np.zeros_like(self.w)

        # expected part
        for i in range(self.N):
            grad -= probs[i] * features[:, i]

        # selected nodes
        for i in subset:
            grad += features[:, i]

        return grad
    

    def update(self, reward, subset, probs):
        # baseline update
        self.baseline = (
            self.baseline_decay * self.baseline +
            (1 - self.baseline_decay) * reward
        )

        #adv = reward - self.baseline
        adv = (reward - baseline) / (np.std(batch_rewards) + 1e-8)

        # REINFORCE gradient
        grad = self._grad_logp(subset, probs)

        # entropy regularization (encourage exploration)
        entropy_grad = -np.log(probs + 1e-12) - 1.0

        total_grad = adv * grad + self.entropy_coef * entropy_grad

        # normalize gradient (stability)
        total_grad /= (np.linalg.norm(total_grad) + 1e-8)

        # update
        self.w += self.lr * total_grad


def train_sensor_selector(
    G, s0, lam,
    N, T, contacts, delta,
    rho,
    lr=0.01,
    baseline_decay=0.95,
    entropy_coef=0.01,
    iterations=200,
    bp_iter=20,
    max_iter_bp=20,
    tol=1e-6,
    damp=0.5,
    batch_size=20,
    grad_clip=1.0
):

    selector = SensorSelectorREINFORCE(N, rho, lr=lr, baseline_decay=baseline_decay, entropy_coef=entropy_coef)

    reward_history = []

    for it in range(iterations):

        # sample a new epidemic realization
        status_nodes = simulate_SI(G, s0, lam, T)

        probs = selector.get_probs()

        batch_rewards = []
        batch_grads = []

        for _ in range(batch_size):

            subset = selector.sample_subset(probs)

            # build observations
            obs_rows = []
            for node in subset:
                for t in range(status_nodes.shape[0]):
                    obs_rows.append((node, int(status_nodes[t, node]), t))

            obs_array = (
                np.array(obs_rows, dtype=int)
                if len(obs_rows) > 0
                else np.empty((0, 3), dtype=int)
            )

            # BP inference
            bp_fg = fg.FactorGraph(N, T, contacts, obs_array, delta)
            bp_fg.update(maxit=max_iter_bp, tol=tol, damp=damp)

            marg = bp_fg.marginals()
            Mt = get_Mt(marg, t=0)

            x_est = np.argmax(Mt, axis=0)

            reward = OV(x_est, status_nodes[0])

            adv = reward - selector.baseline
            grad = selector._grad_logp(subset, probs)

            batch_rewards.append(reward)
            batch_grads.append(adv * grad)

        # aggregate
        avg_grad = np.mean(batch_grads, axis=0)

        # gradient clipping
        norm = np.linalg.norm(avg_grad)
        if norm > grad_clip:
            avg_grad = avg_grad / norm * grad_clip

        avg_reward = np.mean(batch_rewards)

        # baseline update
        selector.baseline = (
            selector.baseline_decay * selector.baseline
            + (1 - selector.baseline_decay) * avg_reward
        )

        # parameter update
        selector.w += selector.lr * avg_grad

        reward_history.append(avg_reward)

        if it % 10 == 0:
            print(f"iter={it}, reward={avg_reward:.4f}, baseline={selector.baseline:.4f}")

    return selector, reward_history