# GNN-Anti-Money-Laundering

A Graph Neural Network (GNN) implementation for detecting illicit transactions in financial networks using the Elliptic Bitcoin dataset.

## Project Goal
To improve the detection of "money laundering" patterns (layering/smurfing) by analyzing transaction relationships rather than just individual transaction features. This project uses Graph Attention Networks (GAT) to aggregate risk signals from neighboring nodes.



## Technical Architecture
* **Model:** Graph Attention Network (GATv2) with multi-head attention.
* **Graph Type:** Directed Graph (Nodes = Transactions, Edges = Flow of funds).
* **Data:** Elliptic Bitcoin Dataset (203k nodes, 234k edges).
* **Explainability:** GNNExplainer is used to identify the sub-graph most responsible for a "high-risk" classification.

## Implementation Details

### 1. Data Processing
* Converts tabular transaction data into a `torch_geometric.data.Data` object.
* Normalizes 166-dimensional feature vectors.
* Handles temporal splits (training on early timestamps, testing on later ones) to prevent data leakage.

### 2. Model Structure
* **Input Layer:** Linear projection of 166 features to 128 hidden dimensions.
* **GAT Layers:** 2 layers of Graph Attention with 8 heads each.
* **Activation:** ELU (Exponential Linear Unit) for gradient stability.
* **Loss Function:** Weighted Cross-Entropy to handle extreme class imbalance (~10% illicit labels).

### 3. Deployment (FastAPI)
* The model is served via a REST API.
* The API accepts a Node ID and retrieves its $k$-hop neighborhood for real-time inference.

## Performance
*Currently benchmarking. The table below will be updated with final test set results.*

| Metric | GAT Model |
| :--- | :--- |
| **Precision** | *TBD* |
| **Recall** | *TBD* |
| **F1-Score** | *TBD* |

## 📁 Directory Structure
```text
├── data/           # Raw and processed dataset files
├── models/         # Saved .pth model weights
├── src/
│   ├── model.py    # PyTorch Geometric GAT implementation
│   ├── train.py    # Training and evaluation pipeline
│   ├── data.py 
│   └── utils.py    # Data loaders and helper functions
├── app.py          # FastAPI entry point
├── Dockerfile      # Containerization configuration
└── requirements.txt
