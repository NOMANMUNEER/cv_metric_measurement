"""
U-Net Segmentation Training Script for Diary Object Measurement
Architecture: U-Net with ResNet34 backbone (segmentation-models-pytorch)
Strictly adheres to assessment rule: Non-YOLO, Non-Roboflow.
Trains on train split, validates on val split, never touches test split.
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DiaryDataset(Dataset):
    def __init__(self, manifest_file, images_dir, masks_dir, img_size=(512, 384)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size  # (W, H)
        
        with open(manifest_file, "r") as f:
            self.filenames = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.masks_dir, f"{base}_mask.png")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        # PyTorch format: (C, H, W)
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def train_model(dataset_dir="dataset", output_dir="models", epochs=15, batch_size=4, lr=1e-4, seed=42):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")
    train_manifest = os.path.join(dataset_dir, "train_split.txt")
    val_manifest = os.path.join(dataset_dir, "val_split.txt")

    train_ds = DiaryDataset(train_manifest, images_dir, masks_dir)
    val_ds = DiaryDataset(val_manifest, images_dir, masks_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded {len(train_ds)} training images, {len(val_ds)} validation images.")

    # U-Net with ResNet34 backbone
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    criterion = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_iou = 0.0
    best_weights_path = os.path.join(output_dir, "unet_resnet34_diary.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks) + bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_iou_list = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                intersection = (preds * masks).sum(dim=(2, 3))
                union = (preds + masks).clamp(0, 1).sum(dim=(2, 3))
                iou = (intersection + 1e-6) / (union + 1e-6)
                val_iou_list.extend(iou.cpu().numpy().flatten())

        val_iou = np.mean(val_iou_list)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path} (Val IoU: {best_val_iou:.4f})")

    print(f"\nTraining complete. Best Validation IoU: {best_val_iou:.4f}")
    return best_weights_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_model(args.dataset_dir, args.output_dir, args.epochs, args.batch_size, args.lr)
