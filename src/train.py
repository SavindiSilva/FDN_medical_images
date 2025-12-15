import sys
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())

from src.utils import load_config, seed_everything
from src.dataset import HAM10000Dataset
from src.models import get_model

def main():
    print("INITIALIZING PIPELINE")
    config = load_config()
    seed_everything(config['seed'])
    

    csv_path = os.path.join("data", "raw", "HAM10000_metadata.csv")
    img_dir = os.path.join("data", "raw", "images")
    
    # Check if files exist before crashing
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found at {csv_path}")
        print("Please put 'HAM10000_metadata.csv' in 'data/raw/'")
        return

    # 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    #initialize dataset
    print("\nLOADING DATASET")
    dataset = HAM10000Dataset(csv_path, img_dir, transform=transform)
    print(f"Dataset found: {len(dataset)} images")

    #dataloader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab one batch to see if it works
    images, labels, indices = next(iter(loader))
    
    print("\nDATA BATCH INSPECTION")
    print(f"Batch Image Shape: {images.shape}") # Should be [4, 3, 224, 224]
    print(f"Batch Labels: {labels}")
    print(f"Sample Indices: {indices}")
    
    print("\nINITIALIZING MODEL")
    model = get_model(config['training']['model_name'])
    
    # Pass dummy data through model to check dimensions
    output = model(images)
    print(f"Model Output Shape: {output.shape}") 
    
    print("\nSUCCESS: Data Pipeline & Model are connected!")

if __name__ == "__main__":
    main()