import torch
import torch_geometric.data as pyg_data
import os

class GNNTrainingCollector:
    def __init__(self, save_dir="data/gnn_training"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.samples = []          # Removed the duplicate initialization
        self.candidate_buffer = []
        self.overlap_log = []      # New log for overlaps

    def reset_buffer(self):
        self.candidate_buffer = []

    def log_candidate(self, candidate, gain, overlap):
        self.candidate_buffer.append({'candidate': candidate, 'gain': gain, 'overlap': overlap})

    def commit_step(self, bp_marginals, G, delta, lam, sim_id, k, sensor_set):
        N = bp_marginals.shape[0]

        # node features: normalized marginals + observed mask
        x = torch.tensor(bp_marginals, dtype=torch.float)
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)

        observed = torch.zeros(N, dtype=torch.float)
        for node in sensor_set:
            observed[node] = 1.0
        x = torch.cat([x, observed.unsqueeze(1)], dim=1)  # (N, T+3)

        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

        y = torch.zeros(N, dtype=torch.float)
        mask = torch.zeros(N, dtype=torch.bool)

        gains = torch.tensor([e['gain'] for e in self.candidate_buffer])
        if gains.std() < 1e-6:
            self.reset_buffer()
            return  # skip flat steps

        # log overlap of best candidate
        best_idx = gains.argmax().item()
        best_overlap = self.candidate_buffer[best_idx]['overlap']
        self.overlap_log.append({'rho': k/N, 'overlap': best_overlap, 'sim_id': sim_id})

        # rank-based labels
        ranks = gains.argsort().argsort().float()
        ranks /= (len(ranks) - 1)
        for i, entry in enumerate(self.candidate_buffer):
            y[entry['candidate']] = ranks[i]
            mask[entry['candidate']] = True

        # consistent permutation across all tensors
        perm = torch.randperm(N)
        x = x[perm]
        y = y[perm]
        mask = mask[perm]

        mapping = torch.zeros(N, dtype=torch.long)
        mapping[perm] = torch.arange(N)
        edge_index = mapping[edge_index]

        data = pyg_data.Data(x=x, edge_index=edge_index, y=y, mask=mask)
        data.delta, data.lam, data.sim_id = delta, lam, sim_id

        assert isinstance(data, pyg_data.Data), f"Expected Data, got {type(data)}"
        self.samples.append(data)
        self.reset_buffer()

    # def commit_step(self, bp_marginals, G, delta, lam, sim_id):
    #     x = torch.tensor(bp_marginals, dtype=torch.float)
    #     x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6) 

    #     edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    #     y = torch.zeros(bp_marginals.shape[0], dtype=torch.float)
    #     mask = torch.zeros(bp_marginals.shape[0], dtype=torch.bool)  # ADD THIS
    #     for entry in self.candidate_buffer:
    #         y[entry['candidate']] = entry['gain']
    #         mask[entry['candidate']] = True                           # AND THIS

    #     num_nodes = x.size(0)
    #     perm = torch.randperm(num_nodes)
        
    #     x = x[perm]
    #     y = y[perm]
    #     mask = mask[perm]                                             # SHUFFLE WITH PERM
        
    #     mapping = torch.zeros(num_nodes, dtype=torch.long)
    #     mapping[perm] = torch.arange(num_nodes)
    #     edge_index = mapping[edge_index]
            
    #     data = pyg_data.Data(x=x, edge_index=edge_index, y=y)
    #     data.mask = mask                                              # STORE ON DATA
    #     data.delta, data.lam, data.sim_id = delta, lam, sim_id
        
    #     self.samples.append(data)
    #     self.reset_buffer()

    def save_checkpoint(self, checkpoint_id):
        # Prevent saving if buffer is empty
        if not self.samples:
            return

        filename = f"dataset_checkpoint_{checkpoint_id}.pt"
        path = os.path.join(self.save_dir, filename)
        
        torch.save(self.samples, path)
        
        print(f"\n[Checkpoint] Saved {len(self.samples)} samples to {filename}")
        
        # CRITICAL: Clear samples to prevent duplicates
        self.samples = []

        # ADD THESE TWO LINES
        torch.save(self.overlap_log, os.path.join(self.save_dir, f"overlap_log_{checkpoint_id}.pt"))
        self.overlap_log = []  # clear to avoid duplicates across checkpoints
        print(f"[Checkpoint] Saved overlap log with {len(self.overlap_log)} entries to overlap_log_{checkpoint_id}.pt")