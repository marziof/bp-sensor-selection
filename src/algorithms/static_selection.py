
import numpy as np
import networkx as nx

def random_selection(bp_base, rho_max, m=None, **kwargs):
    k = int(rho_max * bp_base.size)
    selected_sensors = set(np.random.choice(bp_base.size, size=k, replace=False))
    return selected_sensors

def deg_centrality_selection(bp_base, rho_max, m=None, G=None, **kwargs):
    if G is None:
        raise ValueError("Graph G must be provided for degree centrality selection.")
    centrality = nx.degree_centrality(G)
    items = list(centrality.items())
    np.random.shuffle(items)  # FIX: break symmetry

    sorted_nodes = [n for n, _ in sorted(items, key=lambda x: x[1], reverse=True)]
    
    #sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    k = int(rho_max * bp_base.size)
    selected_sensors = set(sorted_nodes[:k])
    return selected_sensors

def betweenness_centrality_selection(bp_base, rho_max, m=None, G=None, **kwargs):
    if G is None:
        raise ValueError("Graph G must be provided for betweenness centrality selection.")
    centrality = nx.betweenness_centrality(G)

    items = list(centrality.items())
    np.random.shuffle(items)  # FIX: break symmetry

    sorted_nodes = [n for n, _ in sorted(items, key=lambda x: x[1], reverse=True)]

    #sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    k = int(rho_max * bp_base.size)
    selected_sensors = set(sorted_nodes[:k])
    return selected_sensors

def page_rank_selection(bp_base, rho_max, m=None, G=None, **kwargs):
    if G is None:
        raise ValueError("Graph G must be provided for PageRank selection.")
    centrality = nx.pagerank(G)

    items = list(centrality.items())
    np.random.shuffle(items)  # FIX: break symmetry

    sorted_nodes = [n for n, _ in sorted(items, key=lambda x: x[1], reverse=True)]    
    #sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    k = int(rho_max * bp_base.size)
    selected_sensors = set(sorted_nodes[:k])
    return selected_sensors


def closeness_selection(bp_base, rho_max, m=None, G=None, **kwargs):
    if G is None:
        raise ValueError("Graph G must be provided for closeness selection.")
    centrality = nx.closeness_centrality(G)

    items = list(centrality.items())
    np.random.shuffle(items)  # FIX: break symmetry

    sorted_nodes = [n for n, _ in sorted(items, key=lambda x: x[1], reverse=True)]    
    #sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
    k = int(rho_max * bp_base.size)
    selected_sensors = set(sorted_nodes[:k])
    return selected_sensors