import torch
import os
from sklearn.model_selection import train_test_split


def load_data(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError("Come On! You can do better than this.")
    
    data = torch.load(path)
    return data.to(device)


def get_class_weights(y, device):
    