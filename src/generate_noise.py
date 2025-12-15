import sys
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset
from src.models import get_model

def train_proxy(model, loader, device, epochs=5):
    """Trains a quick proxy model to learn feature confusion."""
    print(f"   -> Training Proxy Model for {epochs} epochs...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for imgs, labels, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model

def get_confusion_probabilities(model, loader, device):
    """Passes data through model to get probability predictions."""
    print("   -> Calculating Feature Confusion...")
    model.eval()
    all_probs = []
    all_indices = []
    
    with torch.no_grad():
        for imgs, _, indices in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_indices.append(indices.numpy())
            
    return np.concatenate(all_probs), np.concatenate(all_indices)

def inject_feature_dependent_noise(df, probs, noise_rate, seed):
    """Flips labels based on model confusion (Feature Dependent)."""
    np.random.seed(seed)
    n_samples = len(df)
    n_noisy = int(n_samples * noise_rate)
    
    # 1. Calculate 'Difficulty' (Entropy)
    # Higher entropy = Model is more confused = More likely to be mislabeled by human
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    
    # 2. Select samples to flip (Top N most confusing images)
    # We sort by entropy and pick the hardest ones
    sorted_indices = np.argsort(entropy)[::-1] # Descending order
    noisy_indices = sorted_indices[:n_noisy]
    
    new_df = df.copy()
    original_labels = new_df['dx'].values
    
    # Map class names to indices for logic
    class_map = {name: i for i, name in enumerate(sorted(df['dx'].unique()))}
    idx_to_class = {i: name for name, i in class_map.items()}
    
    print(f"   -> Injecting {int(noise_rate*100)}% noise ({n_noisy} samples)...")
    
    for idx in noisy_indices:
        # Flip to the class the model was *most* confused with (2nd highest prob)
        sample_probs = probs[idx]
        current_label_idx = class_map[original_labels[idx]]
        
        # Zero out true class to find the next best guess
        sample_probs[current_label_idx] = -1 
        new_label_idx = np.argmax(sample_probs)
        
        # Assign new label
        new_df.at[idx, 'dx'] = idx_to_class[new_label_idx]
        
    return new_df

def main():
    print("--- 📉 STARTING FEATURE-DEPENDENT NOISE INJECTION ---")
    config = load_config()
    seed_everything(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data
    train_csv = os.path.join("data", "splits", "train_clean.csv")
    img_dir = os.path.join("data", "raw", "images")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(config['training']['normalize_mean'], config['training']['normalize_std'])
    ])
    
    dataset = HAM10000Dataset(train_csv, img_dir, transform=transform)
    # Use smaller batch size for safety on laptop
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 2. Train Proxy Model (ResNet18 is faster than ResNet50)
    print("1. Training Proxy Model (ResNet18) on CLEAN data...")
    # Using ResNet18 because it's faster and sufficient for finding 'hard' samples
    model = get_model('resnet18', num_classes=7).to(device)
    model = train_proxy(model, loader, device, epochs=5) # 5 Epochs is enough
    
    # 3. Get Confusion Matrix (Probabilities)
    probs, indices = get_confusion_probabilities(model, loader, device)
    
    # Re-order probabilities to match DataFrame order (loader might shuffle if not careful)
    # But we set shuffle=False, so it matches exactly.
    
    # 4. Generate & Save Noisy Datasets
    print("\n2. Generating Noisy CSVs...")
    df = pd.read_csv(train_csv)
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    noise_levels = [0.2, 0.4, 0.6] # 20%, 40%, 60%
    
    for rate in noise_levels:
        noisy_df = inject_feature_dependent_noise(df, probs, rate, config['seed'])
        
        save_path = os.path.join(output_dir, f"train_noise_{int(rate*100)}.csv")
        noisy_df.to_csv(save_path, index=False)
        print(f"   ✅ Saved: {save_path}")

    print("\n🎉 DONE! You now have Feature-Dependent Noisy Data.")

if __name__ == "__main__":
    main()