
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def filter_df(df, method=None, metric=None, delta=None, lam=None, sim=None):
    dff = df.copy()
    # if method is not None:
    #     dff = dff[dff["method"] == method]
    # if metric is not None:
    #     dff = dff[dff["metric"] == metric]
    # if delta is not None:
    #     dff = dff[dff["delta"] == delta]
    # if lam is not None:
    #     dff = dff[dff["lambda"] == lam]
    # if sim is not None:
    #     dff = dff[dff["sim"] == sim]
    
    dff["method_metric"] = np.where(
        dff["method"] == "random",
        "rnd",
        dff["method"].astype(str) + "_" + dff["metric"].astype(str)
    )
    return dff #dff.sort_values("rho")


def plot_pinf_vs_rho(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="p_inf_selected", 
                 hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("p_inf (selected)")
    ax.set_title("Selected node infection probability vs rho")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot


def plot_rank_vs_rho(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]

    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="rank_in_candidates", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title("Rank of selected node vs rho")
    plt.gca().invert_yaxis()  # optional: better rank = higher visually
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot

def plot_true_infected_vs_rho(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="true_state_t0", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("Fraction truly infected (t=0)")
    ax.set_title("True infection rate of selected nodes vs rho")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot


def plot_neighbor_pinf_vs_rho(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="neigh_mean_p_inf", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("Neighbor mean p_inf")
    ax.set_title("Neighbors' infection probability vs rho")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot


def plot_entropy_vs_rho(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="entropy_selected", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("Entropy (selected)")
    ax.set_title("Entropy of selected nodes vs rho")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot