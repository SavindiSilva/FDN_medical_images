import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())
# Assuming these imports exist in your repo structure
from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset
from src.models import get_model
from src.sieve import Sieve

def calculate_losses(model, loader, criterion, device):
    """Scan dataset and return loss per sample"""
    model.eval()
    losses = []
    # Use reduction='none' to get individual losses
    criterion_none = nn.CrossEntropyLoss(weight=criterion.weight, reduction='none')
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="🔍 Scanning Dataset", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_losses = criterion_none(outputs, labels)
            losses.extend(batch_losses.cpu().numpy())
    return losses

def train_warmup(model, loader, criterion, optimizer, device):
    """Standard training loop (no sieve filtering)"""
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, leave=False)
    
    for images, labels, _ in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")
        
    return running_loss / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="experiments/diagnostic_prototype")
    parser.add_argument("--epochs", type=int, default=8, help="Run just enough to see the separation")
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # simple transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dataset
    train_dataset = HAM10000Dataset(args.csv_file, args.image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    # note: create a secondary loader with shuffle=False for the Sieve scan
    scan_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=2)

    # model Setup
    model = get_model("resnet50", num_classes=7).to(device)
    config = load_config()
    weights = torch.tensor(config['training']['class_weights']).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    sieve = Sieve(output_dir=args.output_dir)

    print(f"starting diagnostic run ({args.epochs} epochs)")
    print("goal: Visualize loss separation (No filtering applied)")

    for epoch in range(1, args.epochs + 1):
        # train normally (Warmup)
        loss = train_warmup(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {loss:.4f}")
        
        # at specific checkpoints, run the scanner AND SAVE THE MODEL
        if epoch in [1, 4, 8]:
            print(f"   taking Sieve Snapshot...")
            losses = calculate_losses(model, scan_loader, criterion, device)
            sieve.analyze(losses, epoch)
            
            # --- CRITICAL ADDITION: SAVE THE MODEL ---
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   model saved: {save_path}")

    print("\n diagnostic complete!")
    print(f"check {args.output_dir} for the histograms and checkpoints")

if __name__ == "__main__":
    main()