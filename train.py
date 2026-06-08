import torch
import pickle
import os

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from models.custom_network import FashionNet
from utils import plot_metrics

# -----------------------------------
# Load Dataset
# -----------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# -----------------------------------
# Data Loaders
# -----------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# -----------------------------------
# Model
# -----------------------------------

model = FashionNet()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------------------------
# Store Metrics
# -----------------------------------

train_losses = []
val_losses = []

train_accs = []
val_accs = []

epochs = 20

# -----------------------------------
# Training Loop
# -----------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Training
    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_acc = 100 * correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validation
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

    val_loss = val_loss / len(test_loader)

    val_acc = 100 * correct / total

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # Print Results
    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

# -----------------------------------
# Save Model
# -----------------------------------

with open(
    "saved_models/model.pkl",
    "wb"
) as f:

    pickle.dump(
        model.state_dict(),
        f
    )

print("\nModel saved successfully!")

# -----------------------------------
# Plot Metrics
# -----------------------------------
print("Current Directory:", os.getcwd())
print("Before the Plot_metrics")

plot_metrics(
    train_losses,
    val_losses,
    train_accs,
    val_accs
)
print("after the plot_metrics")

print("Training Complete!")
# ------------------------------------
# Generate submission.csv
# ------------------------------------
import pandas as pd

predictions = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        predictions.extend(
            preds.cpu().numpy()
        )

print("Number of predictions:", len(predictions))

if len(predictions) > 0:
    print("First 10 predictions:", predictions[:10])

submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv(
    "outputs/submission.csv",
    index=False
)

print("submission.csv created successfully!")