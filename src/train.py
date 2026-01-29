import torch
import torch.nn as nn
import torch.optim as optim
import json
import yaml
from model import LogisticRegression

with open("params.yaml") as f:
    params = yaml.safe_load(f)

images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")

images = images.view(images.size(0), -1)

model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["train"]["lr"])

for epoch in range(params["train"]["epochs"]):
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

_, preds = torch.max(outputs, 1)
accuracy = (preds == labels).float().mean().item()

torch.save(model.state_dict(), "model.pt")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "loss": loss.item()}, f)
