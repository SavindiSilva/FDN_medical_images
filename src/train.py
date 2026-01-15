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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    loop = tqdm(loader, leave=False)
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        loop.set_description(f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_mcc, epoch_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--csv_file", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # Load Config (for seed only)
    config = load_config()
    seed_everything(config['seed'])
    
    train_csv = args.csv_file
    val_csv = os.path.join("data", "splits", "val_fold_0.csv")
    image_dir = args.image_dir
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training {args.model_name} (Pretrained) on {device}...")

    # --- DATA & SAMPLER ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = HAM10000Dataset(train_csv, image_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_csv, image_dir, transform=val_transform)

    # Calculate Weights for Sampler
    print("Building Weighted Random Sampler...")
    df = pd.read_csv(train_csv)
    if 'label' not in df.columns:
        lesion_type_dict = {'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6}
        df['label'] = df['dx'].map(lesion_type_dict)
    
    y_train = df['label'].values
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --- MODEL (FORCE PRETRAINED) ---
    print("Loading Pretrained ResNet50 Weights...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Replace Head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # Safe LR for pretrained

    # --- LOOP ---
    best_mcc = -1.0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_mcc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f}")
        
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"🌟 New Best Model Saved! (MCC: {val_mcc:.4f})")
            
        # CSV Log
        log_df = pd.DataFrame([{
            'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_mcc': val_mcc
        }])
        log_df.to_csv(os.path.join(output_dir, "log.csv"), mode='a', header=not os.path.exists(os.path.join(output_dir, "log.csv")), index=False)

    print("\nTraining Complete")

if __name__ == "__main__":
    main()