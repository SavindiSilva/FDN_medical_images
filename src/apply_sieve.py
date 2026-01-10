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

# --- IMPORTS FROM YOUR REPO ---
# (Adjust these imports to match your actual file structure)
from dataset import SkinDataset, get_transforms  
from models import build_resnet50

# ================= CONFIGURATION =================
# Path to your Phase 4 Model (Epoch 8)
MODEL_PATH = "experiments/diagnostic_prototype/checkpoint_epoch_8.pth" 
# Original Noisy Data
INPUT_CSV = "/content/drive/My Drive/backbone_selection/train_idn_20.csv" 
IMAGE_DIR = "/content/data/images"
# Where to save the new clean list
OUTPUT_CSV = "/content/drive/My Drive/backbone_selection/train_sieved_20.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =================================================

def apply_sieve():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Setup Dataset (No Augmentation for filtering, just Resize/Normalize)
    dataset = SkinDataset(df, IMAGE_DIR, transform=get_transforms(phase='val'))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # Load Model
    print(f"Loading Phase 4 model from {MODEL_PATH}...")
    model = build_resnet50(num_classes=7).to(DEVICE)
    
    # Load weights (Handle 'state_dict' key if present)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none') # We need per-sample loss

    # 1. Calculate Losses for ALL samples
    print("Calculating losses for all training samples...")
    all_losses = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            all_losses.extend(loss.cpu().numpy())

    all_losses = np.array(all_losses)
    
    # 2. Fit GMM (The Sieve)
    print("Fitting GMM to isolate Clean vs. Suspect...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    loss_reshape = all_losses.reshape(-1, 1)
    gmm.fit(loss_reshape)
    
    # Identify which cluster is "Clean" (Low Loss)
    if gmm.means_[0] < gmm.means_[1]:
        clean_idx = 0
    else:
        clean_idx = 1
        
    probs = gmm.predict_proba(loss_reshape)
    clean_probs = probs[:, clean_idx]
    
    # 3. Create the Split
    # If probability of being clean is > 0.5, we keep it.
    is_clean = clean_probs > 0.5
    
    df_clean = df[is_clean].copy()
    df_suspect = df[~is_clean].copy()
    
    # 4. Save and Report
    print(f"\n--- SIEVE REPORT ---")
    print(f"Total Samples: {len(df)}")
    print(f"Kept (Clean):  {len(df_clean)} ({len(df_clean)/len(df):.1%})")
    print(f"Removed (Suspect): {len(df_suspect)} ({len(df_suspect)/len(df):.1%})")
    
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"saved clean dataset to: {OUTPUT_CSV}")

    # 5. Visualize (Evidence for Supervisor)
    plt.figure(figsize=(10, 6))
    sns.histplot(all_losses[is_clean], color='green', label='Kept (Clean)', kde=True, bins=50)
    sns.histplot(all_losses[~is_clean], color='red', label='Removed (Suspect)', kde=True, bins=50)
    plt.title("Phase 5 Data Sieve: Separation of Clean vs Noisy")
    plt.xlabel("Loss Value")
    plt.legend()
    plt.savefig("experiments/sieve_separation_proof.png")
    print("Graph saved to experiments/sieve_separation_proof.png")

if __name__ == "__main__":
    apply_sieve()