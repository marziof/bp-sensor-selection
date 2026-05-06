import os
from src.GNN.data_collection import GNNTrainingCollector
from src.GNN.generate_gnn_data import generate_gnn_dataset

# -------------------
# CONFIGS
# -------------------
CONFIG = {
    "N": 50,
    "d": 3,
    "T_max": 10,
    "Nsim": 2000,
    "graph_type": "rrg",
    "rho_max": 1, # Warning: 1.0 means picking every node; 0.2-0.5 is usually sufficient for gain
    "lambdas": [0.3],
    "deltas": [0.3]#[0.1, 0.2, 0.3, 0.4, 0.5]
}

# -------------------
# RESULTS STORAGE
# -------------------
# Change extension to .pt to support binary tensor serialization
collector = GNNTrainingCollector(save_dir="src/GNN/data_var_ov")
# -------------------
# RUN EXPERIMENTS
# -------------------
if __name__ == "__main__":
    # Ensure the 'data' directory exists
    os.makedirs("src/GNN/data_var_ov", exist_ok=True)
    
    print(f"Starting generation: {len(CONFIG['deltas']) * len(CONFIG['lambdas']) * CONFIG['Nsim']} simulation loops.")
    
    generate_gnn_dataset(
        deltas=CONFIG["deltas"],
        lambdas=CONFIG["lambdas"],
        rho_max=CONFIG["rho_max"],
        Nsim=CONFIG["Nsim"],
        N=CONFIG["N"],
        T_max=CONFIG["T_max"],
        d=CONFIG["d"],
        graph_type=CONFIG["graph_type"],
        collector=collector
    )