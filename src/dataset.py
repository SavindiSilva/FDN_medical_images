import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image ID and Label
        img_id = self.df.iloc[idx]['image_id']
        label = int(self.df.iloc[idx]['label'])
        
        # 2. Construct Path (Handle missing .jpg extension if needed)
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        
        # 3. Load Image (Use PIL for safety)
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            # Fallback for debugging - create a black square if file missing
            # But print a warning so we know!
            print(f" WARNING: Could not find image {img_path}")
            image = Image.new('RGB', (224, 224), color='black')

        # 4. Apply Transforms
        if self.transform:
            image = self.transform(image)
        
        # 5. Return (Image, Label, ID)
        # We return ID too, just in case we need to debug which image failed
        return image, label, img_id