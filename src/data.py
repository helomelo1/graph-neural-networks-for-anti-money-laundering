import pandas as pd
import torch
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
    
    df['src_str'] = df['From Bank'].astype(str) + "_" + df["Account"].astype(str)
    df['dest_str'] = df['To Bank'].astype(str) + "_" + df["Account.1"].astype(str)

    all_accounts = pd.concat([df["src_str"], df["dest_str"]]).unique()
    print(f"No. of Accounts: {len(all_accounts)}")

    account_map = {acc: i for i, acc in enumerate(all_accounts)}

    df['src_id'] = df['src_str'].map(account_map)
    df['dest_id'] = df['dest_str'].map(account_map)

    src_tensor = torch.tensor(df['src_id'].values, dtype=torch.long)
    dst_tensor = torch.tensor(df['dst_id'].values, dtype=torch.long)
    edge_index = torch.stack([src_tensor, dst_tensor], dim=0)

    print("Preparing Features (X)..")

    num_nodes = len(all_accounts)
    node_features = torch.zeros((num_nodes, 2), dtype=torch.float)

    sent_stats = df.groupby('src_id')['Amount Recieved'].sum()
    recv_stats = df.groupby('dst_id')['Amount Recieved'].sum()

    node_features[sent_stats.index, 0] = torch.tensor(sent_stats.values, dtype=torch.float)
    node_features[recv_stats.index, 1] = torch.tensor(recv_stats.values, dtype=torch.float)

    node_features = torch.log1p(node_features)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(node_features.numpy())
    x = torch.tensor(scaled_features, dtype=torch.float)

    print("Preparing Labels (y)")

    y = torch.zeros(num_nodes, dtype=torch.long)

    bad_trans = df[df['Is Laundering'] == 1]
    bad_accts = pd.concat([bad_trans['src_id'], bad_trans['dest_id']]).unique()

    y[bad_accts] = 1

    print(f"#Illicit Accounts: {len(bad_accts)}")
    print(f"#Licit Accounts = {num_nodes - len(bad_accts)}")

    data = Data(x=node_features, edge_index=edge_index, y=y)
    torch.save(data, output_path)
    print(f"Data Successfully Created and Saved To {output_path}")


if __name__ == "__main__":
    process_data()