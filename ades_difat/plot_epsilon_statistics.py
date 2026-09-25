import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import math


def plot_epsilon_statistics(eps_stats, lambda_mean, save_path):
    """
    Plot epsilon statistics over training epochs.

    Parameters
    ----------
    eps_stats : dict
        Dictionary containing:
            - 'mean'
            - 'std'
            - 'min'
            - 'max'
            - 'median'
        Each value should be a list of length = number of epochs.
    """

    mean = np.array(eps_stats["mean"])
    std = np.array(eps_stats["std"])
    min_eps = np.array(eps_stats["min"])
    max_eps = np.array(eps_stats["max"])
    median = np.array(eps_stats["median"])

    epochs = np.arange(1, len(mean) + 1)

    # Linear target: 0 -> target_max
    target = np.linspace(0, 8/255, len(epochs))

    plt.figure(figsize=(8, 5))

    # Mean
    plt.plot(epochs, mean, label="Mean", linewidth=2)

    # Mean ± std
    plt.fill_between(
        epochs,
        mean - std,
        mean + std,
        alpha=0.25,
        label="Mean ± Std"
    )

    # Median
    plt.plot(
        epochs,
        median,
        linestyle="--",
        linewidth=2,
        label="Median"
    )

    # Min / Max
    plt.plot(
        epochs,
        min_eps,
        linestyle=":",
        linewidth=1.5,
        label="Min"
    )
    plt.plot(
        epochs,
        max_eps,
        linestyle=":",
        linewidth=1.5,
        label="Max"
    )

    # Target behaviour
    #plt.plot(
    #    epochs,
    #    target,
    #    linestyle="--",
    #    linewidth=2.5,
    #    label=r"Target $\epsilon$ (0 $\rightarrow$ 8/255)"
    #)

    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.title(f"Adaptive Epsilon Statistics During Training with lamba mean = {lambda_mean}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_epsilon_convergence(eps_stats, lambda_mean, save_path):
    mean = np.array(eps_stats["mean"])
    std = np.array(eps_stats["std"])
    epochs = np.arange(1, len(mean) + 1)

    eps_min = 2 / 255

    plt.figure(figsize=(8, 4))

    plt.plot(epochs, mean, marker="o", linewidth=2, label="Mean ε")
    plt.fill_between(epochs, mean - std, mean + std, alpha=0.25)

    plt.axhline(
        eps_min,
        linestyle="--",
        linewidth=2,
        label="2/255"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.title(f"Convergence of Adaptive Epsilon with lambda_mean = {lambda_mean}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def get_epsilon(epoch, num_epochs_rampup, eps_start, eps_end, sched_type):
    t = min(epoch / num_epochs_rampup, 1.0)  # clamp to [0,1]
    
    # Linear: slow and steady
    linear = eps_start + (eps_end - eps_start) * t

    # Cosine: gentler start, faster finish
    cosine = eps_start + (eps_end - eps_start) * (1 - math.cos(math.pi * t)) / 2

    # Exponential: very gentle start
    exponential = eps_start + (eps_end - eps_start) * (t ** 2)
    if sched_type == 'linear':
        return linear
    elif sched_type == 'cosine':
        return cosine


def plot_scheduler(values, save_path, title, xlabel="Epochs", ylabel="Epsilon values"):
    """
    Plot the values of a numerical list.

    Args:
        values (list): List of numerical values to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

#with open("history/history_pgdat_ades_difat/history_pgdat_baseline__linear_eps_sched_numeprampup12_lr_0.001_seed_42_epochs_25_freeze.json", "r") as f:
#    stats = json.load(f)
#
#loss_label = "MAXLOSS"
#lambda_mean = stats["lambda_mean"]

#save_path_1 = f"plots/eps_stats_{loss_label}_lambda_mean_{lambda_mean}_numepochs_25.png"
#save_path_2 = f"plots/eps_mean_{loss_label}_lambda_mean_{lambda_mean}_num_epochs_25.png"

save_path_1 = f"plots/eps_stats_baseline_numepochs_25.png"
save_path_2 = f"plots/eps_mean_baseline_num_epochs_25.png"

epsilon_list = []

num_epochs = 25
for epoch in range(num_epochs):
    eps = get_epsilon(epoch, num_epochs, 0/255, 8/255, "cosine")
    epsilon_list.append(eps)

plot_scheduler(epsilon_list, save_path="plots/generic_cosine_eps_scheduler.png", title="Cosine Epsilon Scheduler Plot")
#plot_epsilon_statistics(stats["eps_stats"], lambda_mean, save_path_1)
#plot_epsilon_convergence(stats["eps_stats"], lambda_mean, save_path_2)