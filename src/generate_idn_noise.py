import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

#add project root to path
sys.path.append(os.getcwd())

from src.dataset import HAM10000Dataset
from src.models import get_model
from src.utils import load_config, seed_everything

def train_proxy(model, loader, device, epochs=5):
    """
    task 2.1
    trains a lightweight proxy model (ResNet18) quickly
    don't need a perfect model; we need one that learns 'easy' features
    so it gets confused by 'hard' features
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    print(f"\ntraining weak proxy (ResNet18) for {epochs} epochs")
    
    for epoch in range(epochs):
        loop = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for images, labels, _ in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
    print("training complete")
    return model

def calculate_entropy(model, loader, device):
    """
    task 2.2 2.3
    calculates entropy (uncertainty) for every image
    high entropy = model is confused = hard sample (candidate for noise)
    """
    model.eval()
    entropies = []
    indices = []
    targets = []
    
    print("\ncalculating entropy (hardness) for each image")
    
    with torch.no_grad():
        for images, labels, idx in tqdm(loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Entropy formula: - sum(p * log(p))
            e = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
            
            entropies.extend(e.cpu().numpy())
            indices.extend(idx.cpu().numpy()) 
            targets.extend(labels.numpy())
            
    return np.array(indices), np.array(entropies), np.array(targets)

def generate_noise(df, indices, entropies, noise_rate, output_path):
    """
    task 2.5 2.7
    selects top X% hardest images and flips their labels
    """
    print(f"\n[noise] generating {int(noise_rate*100)}% IDN Noise")
    
    #sort by Entropy (High -> Low)
    sorted_idxs = np.argsort(entropies)[::-1]
    
    #select the top N samples
    num_noise = int(len(df) * noise_rate)
    target_indices = indices[sorted_idxs[:num_noise]] #these are the 'Hard' images
    
    #label Mapping (Int -> Str)
    #need to flip the string label 'dx', not just the integer
    idx_to_class = {0: 'nv', 1: 'mel', 2: 'bkl', 3: 'bcc', 4: 'akiec', 5: 'vasc', 6: 'df'}
    classes = list(idx_to_class.values())
    
    df_noisy = df.copy()
    flip_count = 0
    
    #flip Loop
    for idx in tqdm(target_indices, desc="Flipping Labels"):
        current_label = df_noisy.loc[idx, 'dx']
        
        #pick a random NEW label (Simulating confusion)
        new_label = np.random.choice(classes)
        while new_label == current_label:
            new_label = np.random.choice(classes)
        
        df_noisy.loc[idx, 'dx'] = new_label
        flip_count += 1

    #save
    #add a 'clean_dx' column so we can check our work later
    df_noisy['clean_dx'] = df.loc[df_noisy.index, 'dx']
    df_noisy.to_csv(output_path, index=False)
    print(f"saved {output_path} (Flipped {flip_count} labels)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to images")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CLEAN train csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    config = load_config()
    seed_everything(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running IDN Generator on: {device}")
    
    #setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    #important: shuffle=False so the indices match the CSV rows exactly
    dataset = HAM10000Dataset(args.csv_file, args.image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    #train Proxy (Task 2.1)
    model = get_model('resnet18', num_classes=7).to(device)
    model = train_proxy(model, loader, device, epochs=5)
    
    #calculate Entropy (Task 2.3)
    indices, entropies, _ = calculate_entropy(model, loader, device)
    
    #generate Datasets (Task 2.6 - 2.8)
    df = pd.read_csv(args.csv_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    #generate 20% IDN
    generate_noise(df, indices, entropies, 0.20, os.path.join(args.output_dir, "train_idn_20.csv"))
    
    #generate 40% IDN
    generate_noise(df, indices, entropies, 0.40, os.path.join(args.output_dir, "train_idn_40.csv"))

if __name__ == "__main__":
    main()