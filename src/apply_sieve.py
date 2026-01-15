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
MODEL_PATH = "/content/drive/My Drive/experiments/diagnostic_prototype_SAVED/checkpoint_epoch_8.pth"
INPUT_CSV = "/content/drive/My Drive/backbone_selection/train_idn_20.csv"
IMAGE_DIR = "/content/data/images"

# OUTPUTS
OUTPUT_CSV = "/content/drive/My Drive/backbone_selection/train_sieved_20.csv"
PLOT_PATH = "/content/drive/My Drive/backbone_selection/sieve_plot.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- NEW SETTINGS FOR STABILITY ---
MAX_REMOVE_RATIO = 0.35  # Safety Cap: Never remove more than 35% of data

def apply_sieve():
    print(f"Loading original data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Setup Dataset
    dataset = HAM10000Dataset(df, IMAGE_DIR, transform=get_transforms(phase='val'))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"Loading Phase 4 Model from {MODEL_PATH}...")

    # Initialize Model
    model = get_model(model_name='resnet50', num_classes=7).to(DEVICE)

    # Load Weights
    if not os.path.exists(MODEL_PATH):
        print(f" CRITICAL ERROR: File not found at {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    # 1. Calculate Losses
    print("Scanning dataset for noisy samples (Inference)...")
    all_losses = []

    with torch.no_grad():
        for batch in tqdm(loader):
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
    print("Fitting GMM to separate Clean vs. Suspect...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    loss_reshape = all_losses.reshape(-1, 1)
    gmm.fit(loss_reshape)

    # Identify which cluster is "Clean" (Lower Mean Loss)
    if gmm.means_[0] < gmm.means_[1]:
        clean_idx = 0
        noisy_idx = 1
    else:
        clean_idx = 1
        noisy_idx = 0

    # Get probability of being CLEAN
    probs = gmm.predict_proba(loss_reshape)
    clean_probs = probs[:, clean_idx]
    
    # Get probability of being NOISY (for sorting)
    noisy_probs = probs[:, noisy_idx]

    # ---APPLY FILTER WITH SAFETY CAP (The Fix) ---
    
    # Initial decision: Standard GMM threshold
    is_clean_initial = clean_probs > 0.5
    num_suspects_initial = np.sum(~is_clean_initial)
    total_samples = len(df)
    
    max_allowed_removal = int(total_samples * MAX_REMOVE_RATIO)
    
    print(f"\nSieve Analysis...")
    print(f"Total Samples: {total_samples}")
    print(f"GMM initially flagged {num_suspects_initial} samples as suspect.")
    print(f"Safety Cap allows removing max: {max_allowed_removal} ({MAX_REMOVE_RATIO:.0%})")
    
    if num_suspects_initial > max_allowed_removal:
        print(f"CAP TRIGGERED: Reducing suspects from {num_suspects_initial} to {max_allowed_removal}")
        
        # Sort ALL samples by "Noisiness" (Highest noisy_prob first)
        # We only remove the 'worst of the worst' up to the limit
        sorted_indices = np.argsort(noisy_probs)[::-1] # Descending order
        
        # Create a boolean mask of False (Keep everything)
        to_remove_mask = np.zeros(total_samples, dtype=bool)
        
        # Mark only the top X 'most noisy' samples for removal
        # But ONLY if they were actually flagged by GMM (probability > 0.5)
        count_removed = 0
        for idx in sorted_indices:
            if count_removed >= max_allowed_removal:
                break
            
            if noisy_probs[idx] > 0.5: # It is statistically noisy
                to_remove_mask[idx] = True
                count_removed += 1
        
        # Final Clean Mask (Inverse of remove mask)
        is_clean_final = ~to_remove_mask
        
    else:
        print(" GMM is within safety limits. Using standard filtering.")
        is_clean_final = is_clean_initial

    # Apply split
    df_clean = df[is_clean_final].copy()
    df_suspect = df[~is_clean_final].copy()

    # 4. Save
    print(f"\nSIEVE COMPLETE")
    print(f"Original Size: {len(df)}")
    print(f"Clean Samples Kept: {len(df_clean)} ({len(df_clean)/len(df):.1%})")
    print(f"Suspects Removed:   {len(df_suspect)} ({len(df_suspect)/len(df):.1%})")

    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Clean dataset saved to: {OUTPUT_CSV}")

    # 5. Plot
    plt.figure(figsize=(10, 6))
    # We plot using the FINAL decision mask
    sns.histplot(all_losses[is_clean_final], color='green', label='Kept (Clean)', kde=True, bins=50)
    sns.histplot(all_losses[~is_clean_final], color='red', label='Removed (Suspect)', kde=True, bins=50)
    plt.title(f"Sieve Separation (Kept: {len(df_clean)})")
    plt.xlabel("Loss Value")
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"Graph saved to: {PLOT_PATH}")

if __name__ == "__main__":
    apply_sieve()