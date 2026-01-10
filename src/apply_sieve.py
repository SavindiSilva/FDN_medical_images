import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add src to path to import your modules
sys.path.append('src')
from dataset import SkinDataset, get_transforms  
from models import build_resnet50

# ================= CONFIGURATION =================
# INPUTS
# (We use the local path because you just finished training)
MODEL_PATH = "experiments/diagnostic_prototype/checkpoint_epoch_8.pth" 
INPUT_CSV = "/content/drive/My Drive/backbone_selection/train_idn_20.csv" 
IMAGE_DIR = "/content/data/images"

# OUTPUT (Save to Drive so it is safe!)
OUTPUT_CSV = "/content/drive/My Drive/backbone_selection/train_sieved_20.csv"
PLOT_PATH = "/content/drive/My Drive/backbone_selection/sieve_plot.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def apply_sieve():
    print(f" loading original data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Setup Dataset (No Augmentation, just Resize/Normalize for evaluation)
    dataset = SkinDataset(df, IMAGE_DIR, transform=get_transforms(phase='val'))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # Load Model
    print(f" loading Phase 4 Model from {MODEL_PATH}...")
    model = build_resnet50(num_classes=7).to(DEVICE)
    
    # Load weights
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model not found at {MODEL_PATH}. Did Phase 4 finish?")
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none') # Per-sample loss

    # 1. Calculate Losses
    print(" scanning dataset for noisy samples (Inference)...")
    all_losses = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.extend(loss.cpu().numpy())

    all_losses = np.array(all_losses)
    
    # 2. Fit GMM (The Sieve)
    print(" fitting GMM to separate Clean vs. Suspect...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    loss_reshape = all_losses.reshape(-1, 1)
    gmm.fit(loss_reshape)
    
    # The component with the LOWER mean is the "Clean" cluster
    if gmm.means_[0] < gmm.means_[1]:
        clean_idx = 0
    else:
        clean_idx = 1
        
    probs = gmm.predict_proba(loss_reshape)
    clean_probs = probs[:, clean_idx]
    
    # 3. Filter Data
    # We keep samples that are >50% likely to be in the Clean cluster
    is_clean = clean_probs > 0.5
    
    df_clean = df[is_clean].copy()
    df_suspect = df[~is_clean].copy()
    
    # 4. Save Results
    print(f"\n SIEVE COMPLETE")
    print(f"   Original Size: {len(df)}")
    print(f"   Clean Samples Kept: {len(df_clean)} ({len(df_clean)/len(df):.1%})")
    print(f"   Suspects Removed:   {len(df_suspect)} ({len(df_suspect)/len(df):.1%})")
    
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"   clean dataset saved to: {OUTPUT_CSV}")

    # 5. Generate Proof Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(all_losses[is_clean], color='green', label='Kept (Clean)', kde=True, bins=50)
    sns.histplot(all_losses[~is_clean], color='red', label='Removed (Suspect)', kde=True, bins=50)
    plt.title(f"Sieve Separation (Kept: {len(df_clean)} | Removed: {len(df_suspect)})")
    plt.xlabel("Loss Value")
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"   graph saved to: {PLOT_PATH}")

if __name__ == "__main__":
    apply_sieve()