import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import HAM10000Dataset
import matplotlib.pyplot as plt
import numpy as np

# CONFIG
CSV_FILE = "data/splits/train_phase6_final.csv"
IMG_DIR = "/content/train_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def denormalize(tensor):
    """Reverses the ImageNet normalization for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    print(" RUNNING SANITY CHECK...")
    
    # 1. Setup Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = HAM10000Dataset(CSV_FILE, IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Grab ONE batch
    images, labels, _ = next(iter(loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    
    print(f" Batch Stats:")
    print(f"   Shape: {images.shape}")
    print(f"   Min Pixel Value: {images.min().item():.3f}")
    print(f"   Max Pixel Value: {images.max().item():.3f}")
    print(f"   Labels: {labels.cpu().numpy()}")
    
    if torch.isnan(images).any():
        print(" CRITICAL ERROR: Input images contain NaNs!")
        return

    # 3. Setup Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Standard LR
    criterion = nn.CrossEntropyLoss()
    
    # 4. The Loop (Train on SAME batch 50 times)
    print("\n Attempting to overfit one batch...")
    model.train()
    
    for i in range(51):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate Acc
        _, preds = torch.max(output, 1)
        acc = (preds == labels).float().mean()
        
        if i % 10 == 0:
            print(f"   Step {i}: Loss = {loss.item():.4f} | Acc = {acc.item():.4f}")

    if acc.item() > 0.9:
        print("\n SUCCESS: Model can learn! The issue is likely 'Hard Data' or 'Hyperparams'.")
    else:
        print("\n FAILURE: Model cannot even memorize 8 images. The CODE/DATA is broken.")

if __name__ == "__main__":
    main()