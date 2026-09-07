import os
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models

import matplotlib.pyplot as plt

train_dir = r"split_data/train"
val_dir   = r"split_data/val"
test_dir  = r"split_data/test"

IMG_SIZE = 224          # ResNet expects 224x224
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def build_dataframe(root_dir):
    records = []
    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_folder = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_folder):
            continue
        for fname in os.listdir(cls_folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                records.append({
                    "filepath": os.path.join(cls_folder, fname),
                    "label_name": cls,
                    "label": class_to_idx[cls]
                })

    df = pd.DataFrame(records)
    return df, class_to_idx

train_df, class_to_idx = build_dataframe(train_dir)
val_df, _ = build_dataframe(val_dir)
test_df, _ = build_dataframe(test_dir)
NUM_CLASSES = len(class_to_idx)

idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Classes:", class_to_idx)
print("Train samples:\n", train_df["label_name"].value_counts())

class LogoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        label = row["label"]

        if self.transform:
            img = self.transform(img)

        return img, label

# Stronger augmentation for better real-world generalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(25),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=10
    ),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = LogoDataset(train_df, transform=train_transform)
val_dataset   = LogoDataset(val_df, transform=val_test_transform)
test_dataset  = LogoDataset(test_df, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def create_model(num_classes):
    """Create a pretrained ResNet18 and replace the final layer."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers (optional but helps with small datasets)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


model = create_model(NUM_CLASSES).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

best_val_acc = 0.0
patience = 7
patience_counter = 0

for epoch in range(EPOCHS):
    start = time.time()

    train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': NUM_CLASSES
        }, "best_laliga_model.pth")
        print(f"  → Saved new best model (val_acc={val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model and evaluate on test set
checkpoint = torch.load("best_laliga_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None)
print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index_label="epoch")
print(history_df.tail())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_df["train_acc"], label="Train Acc")
plt.plot(history_df["val_acc"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_df["train_loss"], label="Train Loss")
plt.plot(history_df["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# Also save a final version
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': class_to_idx,
    'idx_to_class': idx_to_class,
    'num_classes': NUM_CLASSES
}, "laliga_logo_cnn_final.pth")
print("Final model saved as laliga_logo_cnn_final.pth")
