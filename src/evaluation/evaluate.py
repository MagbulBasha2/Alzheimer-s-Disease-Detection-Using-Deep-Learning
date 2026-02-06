import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from src.data.dataloader import get_dataloaders
from src.models.model import DenseNet121

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "train"
TEST_DIR = PROJECT_ROOT / "test"

def evaluate():
    _, test_loader, _ = get_dataloaders(TRAIN_DIR, TEST_DIR, batch_size=32)

    model = DenseNet121(num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load("densenet121_baseline.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
