
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


# plot entropy vs mean entropy of candidates for given method
def plot_entropy_comparison_vs_rho(df, method=None, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    if method is not None:
        dff = dff[dff["method_metric"] == method]

    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    sns.lineplot(data=dff, x="rho", y="entropy_selected", palette="Set2", ax=ax, color="red", label="Selected")
    sns.lineplot(data=dff, x="rho", y="entropy_cand_mean", palette="Set2", ax=ax, color="blue", label="Candidates")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Entropy")
    ax.set_title("Entropy of selected nodes vs candidates' mean entropy")
    plt.tight_layout()
    plt.grid()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot



def plot_pinf_selected_vs_candidates(df, method_metric=None, delta=None, window=10, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    if method_metric is not None:
        dff = dff[dff["method_metric"] == method_metric]

    dff = dff.sort_values("rho")
    dff["p_inf_selected_smooth"] = dff["p_inf_selected"].rolling(window, center=True, min_periods=1).mean()
    dff["p_inf_cand_mean_smooth"] = dff["cand_mean_p_inf"].rolling(window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    # raw values faded in background
    ax.plot(dff["rho"], dff["p_inf_selected"], color="red", alpha=0.15)
    ax.plot(dff["rho"], dff["cand_mean_p_inf"], color="blue", alpha=0.15)
    # smoothed on top
    ax.plot(dff["rho"], dff["p_inf_selected_smooth"], color="red", label="Selected")
    ax.plot(dff["rho"], dff["p_inf_cand_mean_smooth"], color="blue", label="Candidates (mean)")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("p_inf")
    title = f"p_inf: selected vs candidates"
    if method_metric:
        title += f" ({method_metric})"
    if delta:
        title += f" [delta={delta}]"
    ax.set_title(title)
    ax.legend()
    plt.grid()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_entropy_selected_vs_candidates(df, method_metric=None, delta=None, window=10, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    if method_metric is not None:
        dff = dff[dff["method_metric"] == method_metric]

    dff = dff.sort_values("rho")
    dff["entropy_selected_smooth"] = dff["entropy_selected"].rolling(window, center=True, min_periods=1).mean()
    dff["entropy_cand_mean_smooth"] = dff["entropy_cand_mean"].rolling(window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dff["rho"], dff["entropy_selected"], color="red", alpha=0.15)
    ax.plot(dff["rho"], dff["entropy_cand_mean"], color="blue", alpha=0.15)
    ax.plot(dff["rho"], dff["entropy_selected_smooth"], color="red", label="Selected")
    ax.plot(dff["rho"], dff["entropy_cand_mean_smooth"], color="blue", label="Candidates (mean)")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Entropy")
    title = "Entropy: selected vs candidates"
    if method_metric:
        title += f" ({method_metric})"
    if delta:
        title += f" [delta={delta}]"
    ax.set_title(title)
    plt.grid()
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_entropy0_selected_vs_candidates(df, method_metric=None, delta=None, window=10, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    if method_metric is not None:
        dff = dff[dff["method_metric"] == method_metric]

    dff = dff.sort_values("rho")
    dff["p_inf_selected_smooth"] = dff["p_inf_selected"].rolling(window, center=True, min_periods=1).mean()
    dff["p_inf_cand_mean_smooth"] = dff["cand_mean_p_inf"].rolling(window, center=True, min_periods=1).mean()

    dff["entropy0_selected_smooth"] = - dff["p_inf_selected_smooth"] * np.log(dff["p_inf_selected_smooth"] + 1e-10) - (1 - dff["p_inf_selected_smooth"]) * np.log(1 - dff["p_inf_selected_smooth"] + 1e-10)
    dff["entropy0_cand_mean_smooth"] = - dff["p_inf_cand_mean_smooth"] * np.log(dff["p_inf_cand_mean_smooth"] + 1e-10) - (1 - dff["p_inf_cand_mean_smooth"]) * np.log(1 - dff["p_inf_cand_mean_smooth"] + 1e-10)

    fig, ax = plt.subplots(figsize=(8, 5))
    # raw values faded in background
    ax.plot(dff["rho"], dff["entropy0_selected_smooth"], color="red", alpha=0.15)
    ax.plot(dff["rho"], dff["entropy0_cand_mean_smooth"], color="blue", alpha=0.15)
    # smoothed on top
    ax.plot(dff["rho"], dff["entropy0_selected_smooth"], color="red", label="Selected")
    ax.plot(dff["rho"], dff["entropy0_cand_mean_smooth"], color="blue", label="Candidates (mean)")

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Entropy at t=0")
    title = f"Entropy at t=0: selected vs candidates"
    if method_metric:
        title += f" ({method_metric})"
    if delta:
        title += f" [delta={delta}]"
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.grid()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)



def plot_true_infected_vs_rho_smooth(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]
    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    dff['true_state_t0_smooth'] = dff['true_state_t0'].rolling(window=10, center=True, min_periods=1).mean()
    sns.lineplot(data=dff, x="rho", y="true_state_t0_smooth", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Fraction truly infected (t=0)")
    ax.set_title("True infection rate of selected nodes vs rho")
    plt.tight_layout()
    plt.grid()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot



def plot_rank_vs_rho_smooth(df, delta=None, save=False, save_path=None, **filters):
    dff = filter_df(df, **filters)
    if delta is not None:
        dff = dff[dff["delta"] == delta]

    fig, ax = plt.subplots(figsize=(8, 5))  # always fresh figure
    dff['rank_in_candidates_smooth'] = dff['rank_in_candidates'].rolling(window=10, center=True, min_periods=1).mean()
    sns.lineplot(data=dff, x="rho", y="rank_in_candidates_smooth", hue="method_metric", palette="Set2", ax=ax)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("Average rank (1 = best)")
    ax.set_title("Rank of selected node vs rho")
    plt.gca().invert_yaxis()  # optional: better rank = higher visually
    plt.tight_layout()
    plt.grid()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # prevent bleed into next plot