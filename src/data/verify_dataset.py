from dataloader import get_dataloaders

TRAIN_DIR = "../../train"
TEST_DIR = "../../test"

train_loader, test_loader, train_dataset = get_dataloaders(
    TRAIN_DIR,
    TEST_DIR,
    batch_size=32
)

print("✅ Total training images:", len(train_dataset))
print("✅ Class names:", train_dataset.classes)
print("✅ Class index mapping:", train_dataset.class_to_idx)

# Check one batch
images, labels = next(iter(train_loader))
print("✅ Batch image shape:", images.shape)
print("✅ Batch labels shape:", labels.shape)
