import torch
import random
from model import LogisticRegression
import torch.nn.functional as F

# Load trained model
model = LogisticRegression()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

# Load data
images = torch.load("data/processed/images.pt")
labels = torch.load("data/processed/labels.pt")

# Randomly select 10 unique indices
indices = random.sample(range(images.size(0)), 10)

# Prepare batch
batch_images = images[indices].view(10, -1)
true_labels = labels[indices]

# Inference
with torch.no_grad():
    outputs = model(batch_images)
    predictions = torch.argmax(outputs, dim=1)
    probabilities = F.softmax(outputs, dim=1)

# Print results
print("Index | True | Predicted | Confidence")
print("-" * 40)

for i, idx in enumerate(indices):
    pred = predictions[i].item()
    true = true_labels[i].item()
    conf = probabilities[i][pred].item()

    print(f"{idx:5d} | {true:4d} | {pred:9d} | {conf:.4f}")
