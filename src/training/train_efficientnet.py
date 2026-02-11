import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from src.data.dataloader import get_dataloaders
from src.models.efficientnet_model import EfficientNetB0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = PROJECT_ROOT / "train"
TEST_DIR = PROJECT_ROOT / "test"

EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001


def train():
    train_loader, test_loader, _ = get_dataloaders(
        TRAIN_DIR, TEST_DIR, batch_size=BATCH_SIZE
    )

    model = EfficientNetB0(num_classes=4).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.classifier[1].parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "efficientnet_baseline.pth")
    print("✅ EfficientNet model saved")


if __name__ == "__main__":
    train()
