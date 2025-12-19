import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):

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
            #return a black image so training doesn't crash (robustness)
            image = Image.new('RGB', (224, 224))
        
        #get Label
        label_str = self.df.iloc[idx]['dx']
        label = self.label_map.get(label_str, 0) # Default to 0 if error

        #apply Transforms
        if self.transform:
            image = self.transform(image)
            
        #return Index (Critical for Sieve!)
        return image, label, idx