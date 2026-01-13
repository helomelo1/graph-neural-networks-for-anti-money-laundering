import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        return focal_loss.mean()


def load_data(path='dataset/ibm_processed.pt', device='cpu'):
    if not os.path.exists(path):
        raise FileNotFoundError("Come On! You can do better than this.")
    
    torch.serialization.add_safe_globals([Data])
    data = torch.load(path, map_location=device)
    print(f"# Nodes: {data.num_nodes}")
    print(f"# Edges: {data.num_edges}")
    return data.to(device)


def get_class_weights(y, device):
    class_counts = torch.bincount(y)
    total_samples = len(y)

    weights = total_samples / (2.0 * class_counts.float())

    return weights.to(device)


def get_train_val_test_masks(data, train_ratio=0.7, val_ratio=0.15):
    num_nodes = data.num_nodes
    indices = range(num_nodes)
    y_numpy = data.y.cpu().numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=y_numpy,
        random_state=42
    )

    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=0.5, 
        stratify=y_numpy[temp_idx], 
        random_state=42
    )

    train_mask = torch.zeros(num_nodes,  dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True 
    val_mask[val_idx] = True 
    test_mask[test_idx] = True 

    return train_mask, val_mask, test_mask


def save_model(model, path='models/best_model.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

    print(f"Model saved to {path}")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")