import json
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob

# ======= 1. Load JSONL files =======
files = ["AICrouch.jsonl", "FacingForwardCrouch.jsonl", "HorridCrouch.jsonl", "NathanCrouching.jsonl"]

X_list = []
y_list = []

for file in files:
    with open("./lms/ProcessedData/" + file, 'r') as f:
        for line in f:
            data = json.loads(line)
            X_list.append(data["input"])
            y_list.append([data["output"]])  # make it a 1-element list for BCELoss

X = torch.tensor(X_list, dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# ======= 2. Define the model =======
model = nn.Sequential(
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
    nn.Sigmoid()  # because output is 0 or 1
)

# ======= 3. Loss and optimizer =======
criterion = nn.BCELoss()  # binary output
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ======= 4. Training loop =======
epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ======= 5. Test predictions =======
with torch.no_grad():
    preds = model(X).round()
    print("Predictions:")
    print(preds.T)
    acc = (preds == y).float().mean()
    print(f"Training Accuracy: {acc:.4f}")
