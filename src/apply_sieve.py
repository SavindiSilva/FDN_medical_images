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

sys.path.append('src')

from dataset import HAM10000Dataset, get_transforms
from models import get_model

#configuration
#inputs
MODEL_PATH = "/content/drive/My Drive/experiments/diagnostic_prototype_SAVED/checkpoint_epoch_8.pth"
INPUT_CSV = "/content/drive/My Drive/backbone_selection/train_idn_20.csv"
IMAGE_DIR = "/content/data/images"

#outputs
OUTPUT_CSV = "/content/drive/My Drive/backbone_selection/train_sieved_20.csv"
PLOT_PATH = "/content/drive/My Drive/backbone_selection/sieve_plot.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#no more than 35% of images in a single class can be removed
MAX_REMOVE_RATIO = 0.35 

def apply_sieve():
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    #ensure numeric labels exist
    if 'label' not in df.columns:
        #map if necessary 
        lesion_type_dict = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6
        }
        df['label'] = df['dx'].map(lesion_type_dict)

    #setup Dataset
    dataset = HAM10000Dataset(df, IMAGE_DIR, transform=get_transforms(phase='val'))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"Loading Model from {MODEL_PATH}...")
    model = get_model(model_name='resnet50', num_classes=7).to(DEVICE)
    
    #load Weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')

    #calculate Losses
    print("Calculating losses...")
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

    df['loss'] = np.array(all_losses)
    
    #fit GMM 
    print("Fitting Global GMM...")
    gmm = GaussianMixture(n_components=2, random_state=42)
    loss_reshape = df['loss'].values.reshape(-1, 1)
    gmm.fit(loss_reshape)
    
    #identify Noisy Component (Higher Mean)
    noisy_component = np.argmax(gmm.means_)
    probs = gmm.predict_proba(loss_reshape)
    df['noisy_prob'] = probs[:, noisy_component]

    #STRATIFIED FILTERING 
    print("\n Applying Stratified Sieve (Per Class)")
    keep_mask = np.zeros(len(df), dtype=bool)
    
    classes = df['label'].unique()
    classes.sort()
    
    for cls in classes:
        #get data for this class
        cls_indices = df[df['label'] == cls].index.to_numpy()
        cls_probs = df.loc[cls_indices, 'noisy_prob'].values
        
        n_total = len(cls_indices)
        n_allowed_remove = int(n_total * MAX_REMOVE_RATIO)
        
        #sort this class by "noisiness"
        sorted_local_indices = np.argsort(cls_probs)[::-1] #highest prob first
        
        #determine how many are actually "suspect" (prob > 0.5)
        n_suspects = np.sum(cls_probs > 0.5)
        
        #remove the minimum of (Actual Suspects, Allowed Cap)
        n_remove = min(n_suspects, n_allowed_remove)
        
        #identify indices to remove for this class
        remove_local_indices = sorted_local_indices[:n_remove]
        
        # Mark these in the global mask
        # Default is False (Remove), so we set Keepers to True
        # First, mark EVERYONE in this class as Keep
        keep_mask[cls_indices] = True
        
        # Then uncheck the ones to remove
        indices_to_drop = cls_indices[remove_local_indices]
        keep_mask[indices_to_drop] = False
        
        print(f"Class {cls}: Total {n_total} | GMM Suspects {n_suspects} | Removed {n_remove} (Cap: {n_allowed_remove})")

    df_clean = df[keep_mask].copy()
    df_suspect = df[~keep_mask].copy()

    #Save
    print(f"\nSIEVE COMPLETE")
    print(f"Original: {len(df)}")
    print(f"Clean:    {len(df_clean)} ({len(df_clean)/len(df):.1%})")
    print(f"Removed:  {len(df_suspect)} ({len(df_suspect)/len(df):.1%})")

    df_clean.drop(columns=['loss', 'noisy_prob'], inplace=True, errors='ignore')
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")

    #plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df.loc[keep_mask, 'loss'], color='green', label='Kept', kde=True, bins=50)
    sns.histplot(df.loc[~keep_mask, 'loss'], color='red', label='Removed', kde=True, bins=50)
    plt.legend()
    plt.title("Stratified Sieve Separation")
    plt.savefig(PLOT_PATH)

if __name__ == "__main__":
    apply_sieve()