from bpepi.Modules import fg_torch as fg #pytorch version
import numpy as np
import networkx as nx
from metrics import *

# select sensors to maximize overlap:

def max_ov_selection(marginals, status_nodes):
    """
       Receives: marginals
       Returns: sensor obs to maximize overlap (assuming ground truth is known)
    """
    selected = np.argmax(entropy)
    for t in range(T_plus1):
        state = int(status_nodes[t, selected])
        obs = (selected, state, t)
    return obs

def max_mov_selection(marginals, status_nodes):
    """
       Receives: marginals, status_nodes (shape (T_max+1, N))
       Returns: sensor obs to maximize mov (assuming ground truth is unknown)
    """
    T_plus1, N = status_nodes.shape
    Mt = get_Mt(marginals, t=0)
    selected = np.argmax(Mt)
    for t in range(T_plus1):
        state = int(status_nodes[t, selected])
        obs = (selected, state, t)  
    return obs

def max_entropy_selection(marginals, status_nodes):
    """
       Receives: marginals, status_nodes (shape (T_max+1, N))
       Returns: sensor obs with max entropy
    """
    T_plus1, N = status_nodes.shape
    entropy = np.zeros(len(marginals))
    for i in range(len(marginals)):
        entropy[i] = -np.sum(marginals[i] * np.log(marginals[i] + 1e-8))
    selected = np.argmax(entropy)
    for t in range(T_plus1):
        state = int(status_nodes[t, selected])
        obs = (selected, state, t)
    return obs

