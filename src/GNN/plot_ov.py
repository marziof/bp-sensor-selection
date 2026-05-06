import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt

# load all overlap logs
all_overlaps = []
for f in glob.glob("src/GNN/data2/overlap_log_*.pt"):
    all_overlaps.extend(torch.load(f, weights_only=False))

df = pd.DataFrame(all_overlaps)
print(f"Loaded {len(df)} overlap entries")
print(df.head())

agg = df.groupby("rho")["overlap"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(agg["rho"], agg["mean"], lw=2, label="sequential oracle")
ax.fill_between(agg["rho"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"], alpha=0.2)
ax.axvline(x=0.3, color='r', ls='--', lw=1, label="rho=delta")
ax.set_xlabel("rho")
ax.set_ylabel("overlap")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("src/GNN/results/overlap_vs_rho_oracle.png", dpi=150)
plt.show()