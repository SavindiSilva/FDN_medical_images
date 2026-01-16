import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms, models

sys.path.append(os.getcwd())
from src.dataset import HAM10000Dataset
from src.utils import seed_everything

# --- CONFIG FOR CURRICULUM ---
SWITCH_EPOCH = 5  # When to switch from Easy -> Hard
# "Dampened" Weights for Stage 2 (Calculated to be safe, not explosive)
# Order: [nv, mel, bkl, bcc, akiec, vasc, df]
# We give rare classes ~3x-5x boost, not 50x.
STAGE_2_WEIGHTS = [0.5, 2.0, 1.5, 2.5, 4.0, 4.0, 4.0]

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_idx, phase_name):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, leave=False, desc=f"Ep {epoch_idx+1} [{phase_name}]")
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_description(f"Ep {epoch_idx+1} [{phase_name}] Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_mcc, epoch_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_warmup", type=str, required=True, help="Full dataset for stability")
    parser.add_argument("--csv_finetune", type=str, required=True, help="Balanced dataset for precision")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/Phase7_Curriculum")
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Curriculum Training on {device}...")

    # --- TRANSFORMS (Added Light ColorJitter for Stage 2 Robustness) ---
    # We will use one transform for simplicity, but you could swap these too.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)), # Mild rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),       # Mild color noise
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- DATASETS ---
    print(" Loading Datasets...")
    ds_warmup = HAM10000Dataset(args.csv_warmup, args.image_dir, transform=train_transform)
    ds_finetune = HAM10000Dataset(args.csv_finetune, args.image_dir, transform=train_transform)
    ds_val = HAM10000Dataset("data/splits/val_fold_0.csv", args.image_dir, transform=val_transform)

    loader_warmup = DataLoader(ds_warmup, batch_size=32, shuffle=True, num_workers=2)
    loader_finetune = DataLoader(ds_finetune, batch_size=32, shuffle=True, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2)

    # --- MODEL ---
    print(" Loading ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Keep Low LR for safety

    # --- LOSS FUNCTIONS ---
    criterion_flat = nn.CrossEntropyLoss() # For Warmup
    
    # Weighted Loss for Finetune (Moves focus to rare classes)
    weights_tensor = torch.tensor(STAGE_2_WEIGHTS).float().to(device)
    criterion_weighted = nn.CrossEntropyLoss(weight=weights_tensor)

    best_mcc = -1.0

    # --- TRAINING LOOP ---
    for epoch in range(20):
        print(f"\n--- Epoch {epoch+1}/20 ---")
        
        # LOGIC: SWITCH CURRICULUM
        if epoch < SWITCH_EPOCH:
            phase = "WARMUP (Full Data)"
            loader = loader_warmup
            criterion = criterion_flat
        else:
            phase = "FINETUNE (Sieve + Weighted)"
            loader = loader_finetune
            criterion = criterion_weighted
            
            if epoch == SWITCH_EPOCH:
                print("🔀 SWITCHING GEARS: Activating Sieve Data & Class Weights!")

        train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device, epoch, phase)
        val_loss, val_acc, val_mcc, val_f1 = validate(model, loader_val, criterion_flat, device) # Always validate with flat loss

        print(f"[{phase}] Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f}")

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_curriculum.pth"))
            print(f"🌟 New Best Model Saved! (MCC: {best_mcc:.4f})")

if __name__ == "__main__":
    main()