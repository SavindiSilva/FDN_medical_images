# import torch
# from torch.utils.data import Dataset

# class HAM10000Dataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         # We will implement the full CSV reading logic later
#         # For now, this is a placeholder to prove the module works
#         self.csv_file = csv_file
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return 10  # Dummy length for testing

#     def __getitem__(self, idx):
#         # Create a fake image tensor (3 channels, 224x224)
#         image = torch.randn(3, 224, 224)
#         label = 0 # Fake label
#         return image, label, idx

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        #HAM10000 mapping
        self.label_map = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        #get image Path
        img_id = self.df.iloc[idx]['image_id']

        if not img_id.endswith('.jpg'):
            img_id = img_id + ".jpg"
            
        img_path = os.path.join(self.img_dir, img_id)
        
        #load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"image not found at {img_path}")
            # Return a black image so training doesn't crash (Robustness)
            image = Image.new('RGB', (224, 224))
        
        #get Label
        label_str = self.df.iloc[idx]['dx']
        label = self.label_map.get(label_str, 0) # Default to 0 if error

        #apply Transforms
        if self.transform:
            image = self.transform(image)
            
        #return Index (Critical for Sieve!)
        return image, label, idx