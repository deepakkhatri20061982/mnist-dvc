import torch
import torch.nn as nn
import torch.optim as optim
import json
import yaml
from model import LogisticRegression
import mlflow
import mlflow.pytorch
from torch.utils.data import TensorDataset, DataLoader
import joblib

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]

mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

lr = params["train"]["lr"]
epochs = params["train"]["epochs"]
images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")
images = images.view(images.size(0), -1)

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

with mlflow.start_run(run_name="training"):
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.4f}")

    # Accuracy Calculation
    model.eval()

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == labels).float().mean().item()

    print("Accuracy:", accuracy)

    torch.save(model.state_dict(), "model.pt")
    mlflow.log_metric("loss", loss.item())
    mlflow.log_metric("accuracy", accuracy)

    joblib.dump(model.state_dict(), "model.pkl")
    mlflow.log_artifact("../model.pkl", artifact_path="model")

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "loss": loss.item()}, f)
