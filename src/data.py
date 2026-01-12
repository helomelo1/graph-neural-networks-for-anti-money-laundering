import pandas as pd
import torch
import os
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler


def process_data():
    csv_path = "data/HI-Small_Trans.csv"
    output_path = "data/ibm_processed.pt"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Come On! You can do better than this.")
        return
    
    df['src_str'] = 