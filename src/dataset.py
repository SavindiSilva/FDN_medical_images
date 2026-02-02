import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # map HAM10000 dx to numeric labels
        self.label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image id
        img_id = row["image_id"]  # your CSV has this

        # label: use numeric label if exists, else map from dx
        if "label" in self.df.columns:
            label = int(row["label"])
        elif "dx" in self.df.columns:
            label = int(self.label_map[row["dx"]])
        else:
            raise KeyError("CSV must contain 'label' or 'dx' column")

        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")

        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            print(f"WARNING: Could not find image {img_path}")
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            image = self.transform(image)

        return image, label, img_id