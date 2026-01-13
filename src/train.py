import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
import os

from model import GraphNeuralNet
from utils import load_data, get_class_weights, get_train_val_test_masks, get_device, FocalLoss


HIDDEN_CHANNELS = 128
LR = 0.01
EPOCHS = 100
MODEL_PATH = "models/best_model.pth"

def train():
    device = get_device()
    print(f"Training on {device}")

    data = load_data("dataset/ibm_processed.pt", device)
    print(f"DEBUG: Node features (x) shape: {data.x.shape}")
    print(f"DEBUG: Edge index shape: {data.edge_index.shape}")
    print(f"DEBUG: Unique labels in Y: {torch.unique(data.y)}")
    
    if data.x.shape[0] == 0:
        print("❌ ERROR: Your feature matrix is empty. Re-run src/process_data.py")
        return

    train_mask, val_mask, test_mask = get_train_val_test_masks(data)

    weights = get_class_weights(data.y[train_mask], device)
    print(f"Loss Weights: Licit={weights[0]:.2f}, Illicit={weights[1]:.2f}")

    model = GraphNeuralNet(
        num_node_features=data.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        num_classes=2   
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_v_f1 = 0
    print("Starting Training..")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            val_f1, val_prec, val_rec = evaluate(model, data, val_mask)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

            if val_f1 > best_v_f1:
                best_v_f1 = val_f1
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)

    print("Training Finished.")
    model.load_state_dict(torch.load(MODEL_PATH))

    t_f1, t_prec, t_rec = evaluate(model, data, test_mask)

    print("Final Test Metrics:")
    print(f"Precision: {t_prec:.4f}")
    print(f"Recall: {t_rec:.4f}")
    print(f"F1-Score: {t_f1:.4f}")


def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return f1, prec, rec


if __name__ == "__main__":
    train()