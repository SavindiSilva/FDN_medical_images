import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from src.utils import load_config, seed_everything

def main():
    print("strating the splitting") #lesion wise split
    # config = load_config()
    # seed_everything(config['seed'])
    
    #load Config & Seed
    try:
        config = load_config()
        seed_everything(config['seed'])
    except Exception as e:
        print("error")
        return
    #load raw metadata
    raw_path = os.path.join("data", "raw", "HAM10000_metadata.csv")
    
    if not os.path.exists(raw_path):
        print(f"metadata not found at {raw_path}")
        return
        
    df = pd.read_csv(raw_path)
    print(f"loaded {len(df)} of total records")

    #stratified group split
    #n_splits=5 to get a 20% Test set 
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['seed'])
    
    #this generates indices based on the groups
    train_idx, test_idx = next(splitter.split(X=df, y=df['dx'], groups=df['lesion_id']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    #verify no leakage
    train_lesions = set(train_df['lesion_id'].unique())
    test_lesions = set(test_df['lesion_id'].unique())
    overlap = train_lesions.intersection(test_lesions)
    
    if len(overlap) == 0:
        print("no lesions overlap between train and test")
    else:
        print(f"leakage detected: {len(overlap)} lesions share images!")
        return

    print(f"training set: {len(train_df)} images")
    print(f"testing set:  {len(test_df)} images")

    #save to splits folder
    output_dir = os.path.join("data", "splits")
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_clean.csv"), index=False)
    
    print(f"splits saved to {output_dir}")

if __name__ == "__main__":
    main()