import seaborn as sns
import matplotlib.pyplot as plt

#results_df = pd.DataFrame(columns=["sensor_type", "rho", "delta", "lambda", "O", "O_tilde", "rank", "precision", "recall", "f1"])


# Plot various performance metrics for both random and selected sensors, side by side for comparison
# plots metric vs rho for different values of delta

def plot_side_by_side(random_results, selected_results, metric="O", save=False, title=None):
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
    if save and title:
        plt.savefig(f"./results/static/{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_comparison(results, metric="O", delta=0.1):
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
    plt.show()


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