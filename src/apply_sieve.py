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

# Add src to path
sys.path.append('src')

# Imports
from dataset import HAM10000Dataset, get_transforms
from models import get_model

# ================= CONFIGURATION =================
# INPUTS
# Pointing specifically to the folder you just saved
MODEL_PATH = "/content/drive/My Drive/experiments/diagnostic_prototype_SAVED/checkpoint_epoch_8.pth"
INPUT_CSV = "/content/drive/My Drive/backbone_selection/train_idn_20.csv"
IMAGE_DIR = "/content/data/images"

# OUTPUTS (Saving the new Clean CSV to Drive)
OUTPUT_CSV = "/content/drive/My Drive/backbone_selection/train_sieved_20.csv"
PLOT_PATH = "/content/drive/My Drive/backbone_selection/sieve_plot.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def apply_sieve():
    print(f"Loading original data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Setup Dataset
    dataset = HAM10000Dataset(df, IMAGE_DIR, transform=get_transforms(phase='val'))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"loading Phase 4 Model from {MODEL_PATH}...")

    # Initialize Model
    model = get_model(model_name='resnet50', num_classes=7).to(DEVICE)

    # Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f" CRITICAL ERROR: File not found at {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint)
        print("model weights loaded successfully.")
    except Exception as e:
        print(f"error loading weights: {e}")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 1. Calculate Losses
    print("scanning dataset for noisy samples (Inference)...")
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(loader):
            # Handle unpack based on what dataset returns (2 or 3 items)
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.extend(loss.cpu().numpy())

    all_losses = np.array(all_losses)

    # 2. Fit GMM
    print("fitting GMM to separate Clean vs. Suspect...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    loss_reshape = all_losses.reshape(-1, 1)
    gmm.fit(loss_reshape)

    # Identify Clean Cluster (The one with LOWER mean loss)
    if gmm.means_[0] < gmm.means_[1]:
        clean_idx = 0
    else:
        clean_idx = 1

    probs = gmm.predict_proba(loss_reshape)
    clean_probs = probs[:, clean_idx]

    # 3. Filter Data (>50% probability of being clean)
    is_clean = clean_probs > 0.5

    df_clean = df[is_clean].copy()
    df_suspect = df[~is_clean].copy()

    # 4. Save
    print(f"\n SIEVE COMPLETE")
    print(f"   Original Size: {len(df)}")
    print(f"   Clean Samples Kept: {len(df_clean)} ({len(df_clean)/len(df):.1%})")
    print(f"   Suspects Removed:   {len(df_suspect)} ({len(df_suspect)/len(df):.1%})")

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"   leacn dataset saved to: {OUTPUT_CSV}")

    # 5. Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(all_losses[is_clean], color='green', label='Kept (Clean)', kde=True, bins=50)
    sns.histplot(all_losses[~is_clean], color='red', label='Removed (Suspect)', kde=True, bins=50)
    plt.title(f"Sieve Separation (Kept: {len(df_clean)})")
    plt.xlabel("Loss Value")
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"   graph saved to: {PLOT_PATH}")

if __name__ == "__main__":
    apply_sieve()