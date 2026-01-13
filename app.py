import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

from src.model import GraphNeuralNet
from src.utils import load_data, get_device


device = get_device()
ml_assets = {}
model = None
data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, data
    print("Initializing API...")

    try:
        data = load_data("dataset/ibm_processed.pt")
    
        model = GraphNeuralNet(
            num_node_features=data.num_features,
            hidden_channels=128,
            num_classes=2
        ).to(device)

        model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
        model.eval()

        ml_assets["model"] = model
        ml_assets["data"] = data
        print("Model and Graph Data Loaded.")

    except Exception as e:
        print(f"Come on you can do better than this. {e}")

    yield
    
    ml_assets.clear()

app = FastAPI(
    title="AML Graph Detective API",
    description="GNN-based Anti-Money Laundering Detection for Bank Accounts",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    node_id: int


class PredictionResponse(BaseModel):
    node_id: int
    risk_score: float
    prediction: str
    recommendation: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if "model" not in ml_assets:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    data = ml_assets["data"]
    model = ml_assets["model"]

    if request.node_id < 0 or request.node_id >= data.num_nodes:
        raise HTTPException(status_code=404, detail="Account ID not found")
    
    with torch.no_grad():
        # Inference
        out = model(data.x, data.edge_index)
        probs = torch.exp(out[request.node_id]) # Convert log_softmax to prob
        
        risk_score = probs[1].item()
        prediction = out[request.node_id].argmax().item()

    return PredictionResponse(
        node_id=request.node_id,
        risk_score=round(risk_score, 4),
        prediction="Illicit" if prediction == 1 else "Licit",
        recommendation="Manual Review" if prediction == 1 else "Auto-Pass"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)