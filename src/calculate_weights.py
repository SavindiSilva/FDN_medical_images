import sys
import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path
sys.path.append(os.getcwd())

from src.utils import load_config

def main():
    print("calculating class weights")
    
    #load train split
    train_csv_path = os.path.join("data", "splits", "train_clean.csv")
    
    if not os.path.exists(train_csv_path):
        print(f"error: {train_csv_path} not found")
        return

    df = pd.read_csv(train_csv_path)
    
    # get all labels
    y = df['dx'].values
    classes = np.unique(y)
    
    # compute weights
    # formula: n_samples / (n_classes * n_samples_j)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    
    print("\nsuccess Copy this list into config.yaml under 'class_weights':")
    print("-" * 50)
    #print formatted list for YAML
    print(f"[{', '.join([f'{w:.4f}' for w in weights])}]")
    print("-" * 50)
    
    print("\nreference mapping (for sanity check):")
    for cls, w in zip(classes, weights):
        print(f"  {cls}: {w:.4f}")

if __name__ == "__main__":
    main()