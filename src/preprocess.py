import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml

with open("params.yaml") as f:
    params = yaml.safe_load(f)

transform = [transforms.ToTensor()]
if params["preprocess"]["normalize"]:
    transform.append(transforms.Normalize((0.1307,), (0.3081,)))

transform = transforms.Compose(transform)

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

torch.save(images, "data/processed/images.pt")
torch.save(labels, "data/processed/labels.pt")
