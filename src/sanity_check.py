import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import HAM10000Dataset
import numpy as np
import sys
import os

# --- CRITICAL CONFIG CHANGE ---
# We are changing this to point to the folder you just created
IMG_DIR = "/content/final_images" 
CSV_FILE = "data/splits/train_phase6_final.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f" RUNNING SANITY CHECK...")
    print(f"   Looking for images in: {IMG_DIR}")
    
    # Check if folder exists and is not empty
    if not os.path.exists(IMG_DIR):
        print(" ERROR: The folder /content/final_images does not exist!")
        return
    
    num_files = len(os.listdir(IMG_DIR))
    print(f"   Found {num_files} files in directory.")
    if num_files == 0:
        print(" ERROR: The folder is empty!")
        return

    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = HAM10000Dataset(CSV_FILE, IMG_DIR, transform=transform)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        # 2. Grab ONE batch
        images, labels, _ = next(iter(loader))
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    print(f"\n📊 Batch Stats:")
    print(f"   Min Pixel Value: {images.min().item():.3f} (Should be ~ -2.0)")
    print(f"   Max Pixel Value: {images.max().item():.3f} (Should be > 2.0)")
    
    # If Max is negative, we are still loading black squares
    if images.max().item() < 0:
        print(" CRITICAL: Max pixel value is negative. Images are still loading as black squares!")
        return

    # 3. Setup Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. The Loop
    print("\n Attempting to overfit one batch...")
    model.train()
    
    for i in range(51):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(output, 1)
        acc = (preds == labels).float().mean()
        
        if i % 10 == 0:
            print(f"   Step {i}: Loss = {loss.item():.4f} | Acc = {acc.item():.4f}")

    if acc.item() > 0.9:
        print("\n SUCCESS: Model learned the batch! You are ready to train.")
    else:
        print("\n FAILURE: Something is still wrong.")

if __name__ == "__main__":
    main()