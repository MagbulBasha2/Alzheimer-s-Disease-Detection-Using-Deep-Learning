import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Image size used across the project
IMAGE_SIZE = (224, 224)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

def get_dataloaders(train_dir, test_dir, batch_size=32):
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=True)
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_dataset
