import os


def list_classes(path):
    """Return sorted class folder names for ImageFolder datasets."""
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
