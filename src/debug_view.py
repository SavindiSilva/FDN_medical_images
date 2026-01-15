import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

# Setup paths
sys.path.append(os.getcwd())
from src.dataset import HAM10000Dataset

def denormalize(tensor):
    """Reverses the ImageNet normalization for visualization"""
    means = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    stds = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    img = tensor.numpy()
    img = img * stds + means # Un-normalize
    img = np.clip(img, 0, 1) # Clip to valid range
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    return img

def debug():
    # 1. Config
    csv_file = "data/splits/val_fold_0.csv"  # Use the file you just tested
    image_dir = "/content/data/images"       # Your image path
    
    print(f"Checking data from: {csv_file}")
    
    # 2. Transform (Exact same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load Dataset
    dataset = HAM10000Dataset(csv_file, image_dir, transform=transform)
    
    # 4. Check Sampler Logic (Replicating train.py logic)
    df = pd.read_csv(csv_file)
    if 'label' not in df.columns:
         lesion_type_dict = {'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6}
         df['label'] = df['dx'].map(lesion_type_dict)
    
    y_train = df['label'].values
    class_counts = np.bincount(y_train, minlength=7)
    
    # Check for empty classes
    if np.any(class_counts == 0):
        print(" WARNING: Some classes have 0 samples! This will break the Sampler.")
        print(f"Counts: {class_counts}")
        
    class_weights = 1. / np.maximum(class_counts, 1) # Safety for 0
    sample_weights = torch.from_numpy(class_weights[y_train])
    sampler = WeightedRandomSampler(sample_weights, 16) # Draw 16 samples

    loader = DataLoader(dataset, batch_size=16, sampler=sampler)

    # 5. Fetch One Batch
    print("Attempting to fetch one batch...")
    try:
        images, labels, _ = next(iter(loader))
        print(f" Batch loaded. Image Shape: {images.shape}, Labels: {labels}")
    except Exception as e:
        print(f" DATALOADER CRASHED: {e}")
        return

    # 6. Visualize
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    label_map = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}
    
    for i in range(16):
        img = denormalize(images[i])
        ax = axes[i]
        ax.imshow(img)
        label_id = labels[i].item()
        ax.set_title(f"Label: {label_id} ({label_map.get(label_id, '?')})")
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("debug_batch_view.png")
    print("\n Saved 'debug_batch_view.png'. Please inspect it.")
    print("If images are BLACK or NOISE, your dataset loading is broken.")
    print("If labels are wrong, your CSV mapping is broken.")

if __name__ == "__main__":
    debug()