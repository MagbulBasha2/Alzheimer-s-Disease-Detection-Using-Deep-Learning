import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super(EfficientNetB0, self).__init__()

        # Load pretrained EfficientNet
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Freeze backbone layers
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Replace classifier for 4 Alzheimer classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
