import torch
import torch.nn as nn
import torch.optim as optim
import json
import yaml
from model import LogisticRegression
import mlflow
import mlflow.pytorch

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

model = LogisticRegression()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

with mlflow.start_run(run_name="training"):
    mlflow.log_param("lr", lr)
    mlflow.log_param("epochs", epochs)
    for epoch in range(epochs):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, preds = torch.max(outputs, 1)
    accuracy = (preds == labels).float().mean().item()

    torch.save(model.state_dict(), "model.pt")
    mlflow.log_metric("loss", loss.item())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.pytorch.log_model(model, artifact_path="model")

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": accuracy, "loss": loss.item()}, f)
