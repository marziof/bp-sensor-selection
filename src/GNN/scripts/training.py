
import os
import torch
import torch.nn.functional as F
from src.GNN.model import SensorSelectorGNN, SensorSelectorMLP
from src.GNN.dataloader import get_dataloader, get_full_dataloader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys

print("Starting training script...")
# Setup
T = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SensorSelectorMLP(in_channels=T+3, hidden=64).to(device)

#model = SensorSelectorGNN(in_channels=T+3, hidden_channels=64).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=50,      # total epochs
#     eta_min=1e-4    # minimum lr
# )
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-4
)


# Get the loader
loader = get_full_dataloader(save_dir="src/GNN/data_var", batch_size=64)
loss_history = []
epochs = []
temperature = 0.2

model.train()
for epoch in tqdm(range(500), desc="Training Epochs:"):
    total_loss = 0

    for batch in loader:                          # ← batch loop
        batch = batch.to(device)
        optimizer.zero_grad() 

        out = model(batch.x, batch.edge_index)

        graph_losses = []
        for graph_id in batch.batch.unique():     # ← graph loop inside batch
            mask = (batch.batch == graph_id) & batch.mask
            if mask.sum() < 2:
                continue
            scores = out[mask]
            gains = batch.y[mask]
            loss = F.kl_div(
                F.log_softmax(scores, dim=0),
                F.softmax(gains / temperature, dim=0),
                reduction='sum'
            )
            graph_losses.append(loss)

        if len(graph_losses) == 0:
            continue

        batch_loss = torch.stack(graph_losses).mean()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    avg_loss = total_loss / len(loader)
    scheduler.step(avg_loss)
    tqdm.write(f"Epoch {epoch} | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    sys.stdout.flush()
    epochs.append(epoch)
    loss_history.append(avg_loss)

# plot loss curve vs epoch
plt.figure(figsize=(6,4))
plt.plot(np.array(epochs), np.array(loss_history), marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# save in src/GNN/plots
os.makedirs("src/GNN/plots", exist_ok=True)
plt.savefig(f"src/GNN/plots/loss_curve_epoch_{epoch}_att_cmov_mlp.png")
plt.show()

    # for batch in loader:
    #     batch = batch.to(device)

    #     # --- ABLATION TEST: Shuffle node features ---
    #     # This keeps the graph structure (edge_index) intact,
    #     # but assigns random feature vectors to the nodes.
    #     # If the loss is still near zero, the model is NOT using the features.
    #     #batch.x = batch.x[torch.randperm(batch.x.size(0))]
        
    #     optimizer.zero_grad()
    #     out = model(batch.x, batch.edge_index)
        
    #     loss = F.mse_loss(out, batch.y)
    #     loss.backward()
    #     optimizer.step()
        
    #     total_loss += loss.item()
    
    # print(f"Epoch {epoch} | Average Loss: {total_loss / len(loader):.4f}")

# Save the final trained model
os.makedirs("src/GNN/models", exist_ok=True)
torch.save(model.state_dict(), "src/GNN/models/sensor_gnn_final_att_500_var_cmov_mlp.pth")
print("Final model saved.")