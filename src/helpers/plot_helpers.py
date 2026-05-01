import seaborn as sns
import matplotlib.pyplot as plt
import os

#results_df = pd.DataFrame(columns=["sensor_type", "rho", "delta", "lambda", "O", "O_tilde", "rank", "precision", "recall", "f1"])


# Plot various performance metrics for both random and selected sensors, side by side for comparison
# plots metric vs rho for different values of delta

def plot_side_by_side(random_results, selected_results, metric="O", save=False, title=None, save_path=None):
    """ 
    Plots the given metric for both random and selected sensor results side by side for comparison.
    Metrics: "O" (overlap), "O_tilde" (rescaled overlap), "rank", "precision", "recall", "f1"
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    sns.lineplot(data=random_results, x="rho", y=metric, hue="delta", palette="Set2", ax=axes[0])
    axes[0].set_title("Random Sensors")
    axes[0].set_xlabel("$\\rho$") 
    axes[0].set_ylabel("{}".format(metric))
    axes[0].legend(title="$\\delta$", loc="upper left")

    sns.lineplot(data=selected_results, x="rho", y=metric, hue="delta", palette="Set2", ax=axes[1])
    axes[1].set_title("Selected Sensors")
    axes[1].set_xlabel("$\\rho$") 
    axes[1].set_ylabel("{}".format(metric))
    axes[1].legend(title="$\\delta$", loc="upper left")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_comparison(results, eval_metric="O", delta=0.1, save=False, title=None, save_path=None):
    """ 
    Plots the given metric for both random and selected sensor results on the same plot for comparison.
    Metrics: "O" (overlap), "O_tilde" (rescaled overlap), "rank", "precision", "recall", "f1"
    """
    results = results.copy()
    results["method_metric"] = results["method"].astype(str) + "_" + results["metric"].astype(str)
    # set nan to rnd for method_metric where method is rnd 
    results.loc[(results["method"] == "random") & (results["method_metric"].isna()), "method_metric"] = "rnd"
    print("List of method_metric combinations: ", results["method_metric"].unique())
    plt.figure(figsize=(8, 6))
    # hue on method and metric (selection method)
    sns.lineplot(data=results[results["delta"] == delta], x="rho", y=eval_metric, hue="method_metric", palette="Set2")
    plt.title("Comparison of {} for Random vs Selected Sensors (delta={})".format(eval_metric, delta))
    plt.xlabel("$\\rho$") 
    plt.ylabel("{}".format(eval_metric))
    plt.legend(title="Sensor Type")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def plot_comparison_old(results, metric="O", delta=0.1, save=False, title=None, save_path=None):
    """ 
    Plots the given metric for both random and selected sensor results on the same plot for comparison.
    Metrics: "O" (overlap), "O_tilde" (rescaled overlap), "rank", "precision", "recall", "f1"
    """
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=results[results["delta"] == delta], x="rho", y=metric, hue="method", palette="Set2")
    plt.title("Comparison of {} for Random vs Selected Sensors (delta={})".format(metric, delta))
    plt.xlabel("$\\rho$") 
    plt.ylabel("{}".format(metric))
    plt.legend(title="Sensor Type")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


## Plot precision-recall curve for both random and selected sensor results, side by side for comparison
# for a given value of delta and rho
def plot_precision_recall(random_results, selected_results):
    """ 
    Plots the precision-recall curve for both random and selected sensor results side by side for comparison. 
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    sns.lineplot(data=random_results, x="precision", y="recall", hue="delta", palette="Set2", ax=axes[0])
    axes[0].set_title("Random Sensors")
    axes[0].set_xlabel("Precision") 
    axes[0].set_ylabel("Recall")
    axes[0].legend(title="$\\delta$", loc="upper left")

    sns.lineplot(data=selected_results, x="precision", y="recall", hue="delta", palette="Set2", ax=axes[1])
    axes[1].set_title("Selected Sensors")
    axes[1].set_xlabel("Precision") 
    axes[1].set_ylabel("Recall")
    axes[1].legend(title="$\\delta$", loc="upper left")
    plt.tight_layout()
    plt.show()



# For sensors dfs:
# sensors_df = pd.DataFrame(columns=["sim", "method", "graph_kind", "N", "d", "delta", "lam", "rho", "k", "subset_size", "mean_pairwise_distance", "boundary_size", "density", "mean_degree", "degree_bias"])
# plot node properties, with hue on method, sns

def plot_sensor_properties(sensors_df, property_name="boundary_size", save=False, title=None):
    """ 
    Plots the given sensor property for both random and selected sensor results side by side for comparison. 
    """
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=sensors_df, x="method", y=property_name, hue="delta", palette="Set2")
    plt.title("Comparison of Sensor {} for Random vs Selected Sensors".format(property_name))
    plt.xlabel("Sensor Selection Method") 
    plt.ylabel(property_name.replace("_", " ").title())
    plt.legend(title="$\\delta$", loc="upper left")
    plt.tight_layout()
    plt.show()
    save_path = f"./results/plots/{title}.png" if save and title else None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
# now plot sensor properties as a function of rho, with hue on method, sns.lineplot

def plot_sensor_properties_vs_rho(sensors_df, property_name="boundary_size", save=False, title=None):
    """ 
    Plots the given sensor property as a function of rho for both random and selected sensor results side by side for comparison. 
    """
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=sensors_df, x="rho", y=property_name, hue="method", palette="Set2")
    plt.title("Comparison of Sensor {} vs $\\rho$ for Random vs Selected Sensors".format(property_name))
    plt.xlabel("$\\rho$") 
    plt.ylabel(property_name.replace("_", " ").title())
    plt.legend(title="Sensor Selection Method", loc="upper left")
    plt.tight_layout()
    plt.show()
    save_path = f"./results/plots/{title}.png" if save and title else None
    # make directory plots if it doesn't exist already
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
