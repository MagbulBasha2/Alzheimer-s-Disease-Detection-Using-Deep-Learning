import torch
import torch.nn as nn
from torchvision import models

class DenseNet121(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNet121, self).__init__()

        self.model = models.densenet121(pretrained=True)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace classifier
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
