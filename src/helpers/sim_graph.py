from bpepi.Modules import fg_torch as fg #pytorch version
import numpy as np
import networkx as nx



import networkx as nx
import numpy as np

def powerlaw_graph(N, gamma=2.5, d=3, seed=None, max_tries=100):
    """
    Generate a simple power-law graph with approximately
    average degree d and no isolated vertices.
    """

    rng = np.random.default_rng(seed)

    for _ in range(max_tries):

        # Sample power-law degree sequence
        deg = np.array(
            np.round(
                nx.utils.powerlaw_sequence(
                    N,
                    exponent=gamma,
                    seed=int(rng.integers(2**31))
                )
            ),
            dtype=int
        )

        deg[deg < 1] = 1

        # Rescale to target mean degree
        deg = np.maximum(
            1,
            np.round(deg * d / deg.mean())
        ).astype(int)

        # Degree sum must be even
        if deg.sum() % 2:
            deg[rng.integers(N)] += 1

        # Build graph
        G = nx.configuration_model(
            deg,
            seed=int(rng.integers(2**31))
        )

        # Convert to simple graph
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        # Require every node to have at least one neighbor
        if nx.number_of_isolates(G) == 0:
            return G

    raise RuntimeError(
        f"Failed to generate a power-law graph without isolates "
        f"after {max_tries} attempts."
    )

### --------------- GRAPH GENERATION AND HELPERS ---------------###
def generate_graph(N, d, kind="rrg"):
    """
    Generate a random graph (rrg or er) with N nodes and average degree d.
    """
    if kind == "rrg":
        G = nx.random_regular_graph(d, N)
    elif kind == "er":
        p = d / (N - 1)
        G = nx.erdos_renyi_graph(N, p)
        # regenerate if there are isolated nodes
        while nx.number_of_isolates(G) > 0:
            G = nx.erdos_renyi_graph(N, p)
    elif kind == "powerlaw":
        #G = nx.powerlaw_cluster_graph(N, d, 0.2)  # powerlaw with clustering, m=d, p=0.1
        G = powerlaw_graph(N, gamma=4, d=d)
    else:
        raise ValueError("Unsupported graph type. Use 'rrg' or 'er'.")
    return G

def add_infection_proba(G, lam):
    """ Add infection probability lam to each edge in the graph G as an attribute 'lambda'."""
    for (u, v) in G.edges():
        G.edges[u, v]["lambda"] = lam
    return G

def get_N(G):
    N = G.number_of_nodes()
    return N

def get_contact_list(G, lam, T_max):
    """
    Receives: G: a graph, lam: infection probability, T_max: maximum number of time steps
    Returns: contact_list: a list of contacts as [i,j,time, lambda]
    """
    contact_list = []
    for t in range(T_max):
        for i, j in G.edges():
            contact_list.append([i, j, t, lam])
    return contact_list


### --------------- SIMULATION HELPERS ---------------###
def simulate_SI(G, s0, lam, T_max=100): 
    """
    Receives: G: a graph, s0: initial state of the nodes (0 for susceptible, 1 for infected), lam: infection probability, T_max: maximum number of time steps
    Returns: status_nodes: a matrix of shape (T_max+1, N) where each row corresponds to the state of the nodes at a given time step
    SI mechanism: at each time step, each infected node can infect its susceptible neighbors with probability lam; each infected node remains infected for the next time step
    """
    N=G.number_of_nodes()
    s0 = s0.tolist()
    status_nodes = np.zeros((T_max+1, N), dtype=int)
    status_nodes[0] = np.array(s0)
    for t in range(T_max):
        status_nodes[t+1] = status_nodes[t].copy()
        infected_nodes = np.where(status_nodes[t] == 1)[0]
        for i in infected_nodes:
            for j in G.neighbors(i):
                if status_nodes[t,j] == 0:   # susceptible
                    if np.random.rand() < lam:
                        status_nodes[t+1,j] = 1
    #status_nodes[T_max+1, :] = 1 # all infected at the end 
    return status_nodes


def gen_s0(N, delta):
    """
    Receives: N: number of nodes, delta: initial probability of being infected (same for each node)
    Returns: s0: initial state of the nodes (0 for susceptible, 1 for infected) (array of shape (N,))
    """
    return np.random.choice([0, 1], size=N, p=[1-delta, delta])


def gen_graph_sim(N, d=5, lam=0.1, T_max=10, delta=0.1, kind="rrg"):
    s0 = np.zeros(N, dtype=int)
    while s0.sum() == 0:  # if no initial infection, redo this simulation
        G = generate_graph(N, d, kind)
        G = add_infection_proba(G, lam)
        contacts = get_contact_list(G, lam, T_max)
        s0 = gen_s0(N, delta)
        if not nx.is_connected(G):
            s0 = np.zeros(N, dtype=int)  # reset s0 to trigger regeneration
    return G, contacts, s0

