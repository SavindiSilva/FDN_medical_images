import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())
from src.utils import load_config, seed_everything

def main():
    print("generating synthetic noise (20%)")
    
    #setup
    config = load_config()
    seed_everything(config['seed']) #CRITICAL: ensures the noise is exactly the same every time
    
    #load the clean training fold 0
    #we use fold 0 for the backbone screening
    input_path = os.path.join("data", "splits", "train_fold_0.csv")
    output_path = os.path.join("data", "processed", "train_noise_20.csv")
    
    if not os.path.exists(input_path):
        print(f"error: could not find {input_path}")
        return
        
    df = pd.read_csv(input_path)
    print(f"   Loaded {len(df)} clean labels.")
    
    #inject Noise
    noise_rate = 0.20 # 20%
    n_noisy = int(len(df) * noise_rate)
    
    #get all possible classes
    classes = df['dx'].unique()
    
    #randomly choose indices to corrupt
    noisy_indices = np.random.choice(df.index, size=n_noisy, replace=False)
    
    #create a copy to modify
    df_noisy = df.copy()
    
    print(f"flipping {n_noisy} labels...")
    
    for idx in noisy_indices:
        true_label = df.loc[idx, 'dx']
        
        #pick a random NEW label (cannot be the true label)
        possible_noise = [c for c in classes if c != true_label]
        new_label = np.random.choice(possible_noise)
        
        df_noisy.loc[idx, 'dx'] = new_label
        
    #save
    #we add a 'clean_label' column so we can cheat/check later if we want
    df_noisy['clean_dx'] = df.loc[df_noisy.index, 'dx']
    
    df_noisy.to_csv(output_path, index=False)
    print(f"saved noisy dataset to: {output_path}")
    print("(added column 'clean_dx' for reference)")

if __name__ == "__main__":
    main()