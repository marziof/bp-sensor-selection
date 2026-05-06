import torch
import glob
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def get_dataloader(data_dir, batch_size=32):
    # Load all .pt files from the folder
    all_data = []
    for f in glob.glob(os.path.join(data_dir, "*.pt")):
        all_data.extend(torch.load(f))
    
    print(f"Loaded {len(all_data)} total graphs.")
    return DataLoader(all_data, batch_size=batch_size, shuffle=True)


def get_full_dataloader(save_dir="src/GNN/data", batch_size=32):
    print(f"Looking for .pt files in: {os.path.abspath(save_dir)}")
    file_list = glob.glob(os.path.join(save_dir, "*.pt"))
    # exclude overlap logs
    file_list = [f for f in file_list if "overlap_log" not in f]
    print(f"Found {len(file_list)} checkpoint files.")
    
    all_samples = []
    for f in file_list:
        data_list = torch.load(f, weights_only=False)
        for d in data_list:
            if not isinstance(d, Data):
                continue
            clean = Data(x=d.x, edge_index=d.edge_index, y=d.y, mask=d.mask)
            all_samples.append(clean)
        
    print(f"Total graphs loaded: {len(all_samples)}")
    return DataLoader(all_samples, batch_size=batch_size, shuffle=True)

# def get_full_dataloader(save_dir="src/GNN/data", batch_size=32):
#     # Find all .pt files in your directory
#     # print full path for debugging
#     print(f"Looking for .pt files in: {os.path.abspath(save_dir)}")
#     file_list = glob.glob(os.path.join(save_dir, "*.pt"))
#     print(f"Found {len(file_list)} checkpoint files.")
    
#     all_samples = []
#     for f in file_list:
#         # Load the list of Data objects from each file
#         data_list = torch.load(f, weights_only=False)
#         all_samples.extend(data_list)
        
#     print(f"Total graphs loaded: {len(all_samples)}")
    
#     # Create the DataLoader for batching
#     return DataLoader(all_samples, batch_size=batch_size, shuffle=True)