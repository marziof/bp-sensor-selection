
import numpy as np

def random_selection(bp_base, rho_max, m=None, **kwargs):
    k = int(rho_max * bp_base.size)
    selected_sensors = set(np.random.choice(bp_base.size, size=k, replace=False))
    return selected_sensors