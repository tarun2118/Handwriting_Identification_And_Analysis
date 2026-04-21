import os
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ============================================================
# DATASET CLASS
# ============================================================
class LineWriterDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = row["label"]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


# ============================================================
# VALIDATION
# ============================================================
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# ============================================================
# MAIN
# ============================================================
def main(args):
    df = pd.read_csv(args.writer_csv)

    # Label mapping
    writers = sorted(df["writer_id"].unique())
    label2id = {w: i for i, w in enumerate(writers)}
    id2label = {i: w for w, i in label2id.items()}

    df["label"] = df["writer_id"].map(label2id)

    # Shuffle + split
    df = df.sample(frac=1).reset_index(drop=True)
    n = len(df)
    t1 = int(n * 0.8)
    t2 = int(n * 0.9)

    df_train = df[:t1]
    df_val = df[t1:t2]
    df_test = df[t2:]

    print("Total:", n)
    print("Train:", len(df_train), "Val:", len(df_val), "Test:", len(df_test))

    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(5),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Datasets
    train_ds = LineWriterDataset(df_train, train_tf)
    val_ds   = LineWriterDataset(df_val, val_tf)
    test_ds  = LineWriterDataset(df_test, val_tf)

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(writers))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    best_val_acc = 0
    hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        tr_l, tr_a = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_l, vl_a = eval_epoch(model, val_loader, criterion, device)

        hist["train_loss"].append(tr_l)
        hist["val_loss"].append(vl_l)
        hist["train_acc"].append(tr_a)
        hist["val_acc"].append(vl_a)

        print(f"Train Loss={tr_l:.4f}, Acc={tr_a:.4f}")
        print(f"Val   Loss={vl_l:.4f}, Acc={vl_a:.4f}")

        if vl_a > best_val_acc:
            best_val_acc = vl_a
            torch.save({
                "model_state": model.state_dict(),
                "label2id": label2id,
                "id2label": id2label
            }, args.output_model)
            print(f"Saved best model → {args.output_model}")

    # Test
    ckpt = torch.load(args.output_model, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    tl, ta = eval_epoch(model, test_loader, criterion, device)
    print(f"\nFINAL TEST ACCURACY = {ta*100:.2f}%")

    # Plot
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(hist["train_loss"], label="Train")
    plt.plot(hist["val_loss"], label="Val")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(hist["train_acc"], label="Train")
    plt.plot(hist["val_acc"], label="Val")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig("writer_id_training_curves.png")
    print("Saved: writer_id_training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--writer_csv", type=str, default="writer_id.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output_model", type=str, default="writer_id_best.pt")
    args = parser.parse_args()

    main(args)