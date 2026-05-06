import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("src/GNN/results/gnn_vs_random_N50_att_500_var_cmov_mlp.csv")

# ── config ────────────────────────────────────────────────────────────────────
metric   = "O"        # change to "MO", "O_tilde", "f1", "precision", etc.
methods  = {"gnn": ("tab:blue", "-"), "random": ("tab:orange", "--")}

# ── aggregate over sims ───────────────────────────────────────────────────────
agg = df.groupby(["method", "rho"])[metric].agg(["mean", "std"]).reset_index()

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

for method, (color, ls) in methods.items():
    sub = agg[agg["method"] == method]
    ax.plot(sub["rho"], sub["mean"], label=method, color=color, ls=ls, lw=2)
    ax.fill_between(sub["rho"],
                    sub["mean"] - sub["std"],
                    sub["mean"] + sub["std"],
                    alpha=0.15, color=color)

ax.set_xlabel("ρ (sensor density)")
ax.set_ylabel(metric)
ax.set_title(f"{metric} vs sensor density")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"src/GNN/results/overlap_vs_rho_{metric}_N50_att_500_var_cmov_mlp.png", dpi=150)
plt.show()
print(f"Saved to src/GNN/results/overlap_vs_rho_{metric}_N50_att_500_var_cmov_mlp.png")