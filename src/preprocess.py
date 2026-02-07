import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
import mlflow
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

with open("experiments/experiment.yaml") as f:
    exp_cfg = yaml.safe_load(f)
experiment_name = exp_cfg["experiment_name"]
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment(experiment_name)

with open("params.yaml") as f:
    params = yaml.safe_load(f)

transform = [transforms.ToTensor()]
normalize = params["preprocess"]["normalize"]
if normalize:
    transform.append(transforms.Normalize((0.1307,), (0.3081,)))

transform = transforms.Compose(transform)

with mlflow.start_run(run_name="preprocess"):
    mlflow.log_param("normalize", normalize)
    dataset = datasets.MNIST(
        root="data/raw",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=1000)

    os.makedirs("data/processed", exist_ok=True)

    images, labels = [], []

    for x, y in loader:
        images.append(x)
        labels.append(y)

    images = torch.cat(images)
    labels = torch.cat(labels)

    os.makedirs("data/processed", exist_ok=True)
    torch.save(images, "data/processed/images.pt")
    torch.save(labels, "data/processed/labels.pt")
    mlflow.log_metric("num_samples", len(images))
    mlflow.log_artifact("data/processed/images.pt")

