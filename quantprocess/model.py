# Script to train model
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import random

# ======= 1. Load JSONL files =======
data_dir = Path(__file__).resolve().parent.parent / "ProcessedData"
files = [
    "AICrouch.jsonl",
    "FacingForwardCrouch.jsonl",
    "HorridCrouch.jsonl",
    "NathanCrouching.jsonl",
    "SyntheticData.jsonl",
]

X_list = []
y_list = []

for file in files:
    path = data_dir / file
    if not path.exists():
        continue
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            X_list.append(data["input"])
            y_list.append([data["output"]])

# Convert to tensors
X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)

print(f"Total dataset: X={X.shape}, y={y.shape}")

# ======= 2. Shuffle + train/test split =======
indices = torch.randperm(len(X))

X = X[indices]
y = y[indices]

split_ratio = 0.8
split_idx = int(len(X) * split_ratio)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# ======= 3. Define the model =======
model = nn.Sequential(
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# ======= 4. Loss and optimizer =======
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======= 5. Training loop =======
epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

# ======= 6. Evaluation =======
with torch.no_grad():
    train_preds = model(X_train)
    test_preds = model(X_test)

    train_error = (train_preds - y_train).abs().mean()
    test_error = (test_preds - y_test).abs().mean()

    print("\n=== Results ===")
    print(f"Train MAE: {train_error:.4f}")
    print(f"Test  MAE: {test_error:.4f}")

    print("\nSample test predictions:")
    print(test_preds[:10].T)
    print("Ground truth:")
    print(y_test[:10].T)
torch.save(model.state_dict(), Path(__file__).resolve().parent / "crouch_model.pth")