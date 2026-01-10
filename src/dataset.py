import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms(phase='train'):
    """Standard ImageNet transforms for ResNet50"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else: # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

class HAM10000Dataset(Dataset):
    def __init__(self, csv_input, img_dir, transform=None):
        """
        Args:
            csv_input (string or pd.DataFrame): Path to the csv file OR a DataFrame.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        # --- Handle both String Path AND DataFrame Object ---
        if isinstance(csv_input, str):
            self.df = pd.read_csv(csv_input)
        else:
            self.df = csv_input
            
        self.img_dir = img_dir
        self.transform = transform 
        
        # HAM10000 mapping
        self.label_map = {
            'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 
            'akiec': 4, 'vasc': 5, 'df': 6
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get Image Path
        if 'image_id' in self.df.columns:
            img_id = self.df.iloc[idx]['image_id']
        else:
            img_id = self.df.iloc[idx, 0] # Fallback

        if not str(img_id).endswith('.jpg'):
            img_id = str(img_id) + ".jpg"
            
        img_path = os.path.join(self.img_dir, img_id)
        
        # 2. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (224, 224))
        
        # 3. Get Label
        if 'dx' in self.df.columns:
            label_str = self.df.iloc[idx]['dx']
            label = self.label_map.get(label_str, 0)
        else:
            try:
                label = int(self.df.iloc[idx]['label']) 
            except:
                label = 0 

        # 4. Apply Transforms
        if self.transform:
            image = self.transform(image)
            
        # Return 3 items (Image, Label, Index)
        return image, label, idx