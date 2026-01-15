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
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset
from src.models import get_model

def get_class_weights(csv_path, device):
    """Calculates "Safe" weights: Boosts rare classes without killing the majority class."""
    print(f"Computing class weights from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Handle label mapping
    if 'label' not in df.columns:
        lesion_type_dict = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
        }
        df['label'] = df['dx'].map(lesion_type_dict)
        
    y = df['label'].values
    classes = np.unique(y)
    
    # 1. Compute Sklearn Balanced Weights
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    # 2. DAMPENING & NORMALIZATION (The Fix)
    # We take the square root to make the penalty less aggressive
    weights = np.sqrt(weights)
    
    # We normalize so the SMALLEST weight is 1.0 (Training Baseline)
    # This ensures Class 0 (Nevi) still generates strong enough gradients to learn features
    weights = weights / np.min(weights)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"Safe Class Weights: {weights_tensor.cpu().numpy()}")
    return weights_tensor

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
    # 1. Parse Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--csv_file", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # 2. Load Config
    config = load_config()
    seed_everything(config['seed'])
    
    model_name = args.model_name if args.model_name else config['training']['model_name']
    epochs = args.epochs if args.epochs else config['training']['epochs']
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    
    train_csv = args.csv_file if args.csv_file else config['noise']['input_csv']
    val_csv = os.path.join("data", "splits", "val_fold_0.csv")
    
    image_dir = args.image_dir if args.image_dir else "data/raw/images"
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training {model_name} on {device} for {epochs} epochs...")

    # 3. Data Setup & SAMPLER
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
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

    # --- NEW: Create Weighted Sampler ---
    print("Building Weighted Random Sampler to fix imbalance...")
    
    # 1. Load DF to get labels
    df = pd.read_csv(train_csv)
    # Ensure map
    if 'label' not in df.columns:
        lesion_type_dict = {'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6}
        df['label'] = df['dx'].map(lesion_type_dict)
    
    y_train = df['label'].values
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    
    # Weight = 1 / count (Rare classes get huge weights)
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    
    # Note: shuffle=False is REQUIRED when using a sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Model & Loss
    model = get_model(model_name, num_classes=7).to(device)
    
    # --- CRITICAL: Remove Class Weights ---
    # The Sampler handles the balance now. Double-weighting would be bad.
    criterion = nn.CrossEntropyLoss() 
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # 5. Training Loop
    best_mcc = -1.0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_mcc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Acc: {val_acc:.4f} | MCC: {val_mcc:.4f} | F1: {val_f1:.4f}")
        
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New Best Model Saved! (MCC: {val_mcc:.4f})")
            
        # Log
        log_path = os.path.join(output_dir, "log.csv")
        log_df = pd.DataFrame([{
            'epoch': epoch+1, 
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_mcc': val_mcc, 'val_f1': val_f1
        }])
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    print("\nTraining Complete")

if __name__ == "__main__":
    main()