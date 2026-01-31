import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from utils import load_config, seed_everything

def main():
    print("starting 5-fold lesion-wise splitting")
    
    #load cleaned data
    input_path = os.path.join("data", "processed", "metadata_cleaned.csv")
    if not os.path.exists(input_path):
        print("cleaned metadata not found!")
        return

    df = pd.read_csv(input_path)
    
    #5-fold splitter
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    output_dir = os.path.join("data", "splits")
    os.makedirs(output_dir, exist_ok=True)

    #loop through all 5 folds
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X=df, y=df['dx'], groups=df['lesion_id'])):
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        #leakage check
        overlap = set(train_df['lesion_id']).intersection(set(val_df['lesion_id']))
        if overlap:
            print(f"leekage in fold {fold}!")
            return

        #save
        train_df.to_csv(os.path.join(output_dir, f"train_fold_{fold}.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, f"val_fold_{fold}.csv"), index=False)
        print(f"saved Fold {fold} (Train: {len(train_df)}, Val: {len(val_df)})")

    print(f"all splits saved to {output_dir}")

if __name__ == "__main__":
    main()