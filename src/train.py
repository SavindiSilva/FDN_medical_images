# import sys
# import os
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms

# sys.path.append(os.getcwd())

# from src.utils import load_config, seed_everything
# from src.dataset import HAM10000Dataset
# from src.models import get_model

# def main():
#     print("initializing pipeline")
#     config = load_config()
#     seed_everything(config['seed'])
    

#     csv_path = os.path.join("data", "raw", "HAM10000_metadata.csv")
#     img_dir = os.path.join("data", "raw", "images")
    
#     #check if files exist before crashing
#     if not os.path.exists(csv_path):
#         print(f"CSV not found at {csv_path}")
#         print("put 'HAM10000_metadata.csv' in 'data/raw/'")
#         return

#     # 
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])

#     #initialize dataset
#     print("\nloading dataset")
#     dataset = HAM10000Dataset(csv_path, img_dir, transform=transform)
#     print(f"dataset found: {len(dataset)} images")

#     #dataloader
#     loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
#     #grab one batch to see if it works
#     images, labels, indices = next(iter(loader))
    
#     print("\ndata batch test")
#     print(f"Batch Image Shape: {images.shape}") #should be [4, 3, 224, 224]
#     print(f"Batch Labels: {labels}")
#     print(f"Sample Indices: {indices}")
    
#     print("\ninitializing model")
#     model = get_model(config['training']['model_name'])
    
#     #pass dummy data through model to check dimensions
#     output = model(images)
#     print(f"Model Output Shape: {output.shape}") 
    
#     print("\ndata pipeline & model are connected!")

# if __name__ == "__main__":
#     main()

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
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset
from src.models import get_model

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
        
        # Store for metrics
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
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_mcc = matthews_corrcoef(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_mcc, epoch_f1

def main():
    # 1. Parse Args (Allows overriding config from Colab)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--csv_file", type=str, default=None) # Train CSV
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # 2. Load Config
    config = load_config()
    seed_everything(config['seed'])
    
    # Overrides
    model_name = args.model_name if args.model_name else config['training']['model_name']
    epochs = args.epochs if args.epochs else config['training']['epochs']
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    
    # Paths
    train_csv = args.csv_file if args.csv_file else config['noise']['input_csv']
    val_csv = os.path.join("data", "splits", "val_fold_0.csv") # Always use Fold 0 Val for screening
    
    image_dir = args.image_dir if args.image_dir else "data/raw/images" # Default local
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training {model_name} on {device} for {epochs} epochs...")
    print(f"   Train Data: {train_csv}")
    print(f"   Val Data:   {val_csv}")

    # 3. Data Setup
    # Train Augmentation (Light)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Val Transform (No Augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = HAM10000Dataset(train_csv, image_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_csv, image_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Model & Loss
    model = get_model(model_name, num_classes=7).to(device)
    
    # Class Weights from Config
    weights = torch.tensor(config['training']['class_weights']).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # 5. Training Loop
    best_mcc = -1.0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_mcc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f}")
        
        # Save Best Model
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New Best Model Saved! (MCC: {val_mcc:.4f})")
            
        # Log to CSV
        log_path = os.path.join(output_dir, "log.csv")
        log_df = pd.DataFrame([{
            'epoch': epoch+1, 
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_mcc': val_mcc, 'val_f1': val_f1
        }])
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    print("\ntraining Complete")

if __name__ == "__main__":
    main()