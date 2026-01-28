import os
import pandas as pd
import numpy as np

def main():
    print("cleaning metadata")
    raw_path = os.path.join("data", "raw", "HAM10000_metadata.csv")
    out_path = os.path.join("data", "processed", "metadata_cleaned.csv")
    
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(raw_path)

    #fill missing age with median
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)
    print(f"   Filled missing ages with median: {median_age}")

    #basic standardization 
    #ensure lesion_id and image_id have no whitespace
    df['lesion_id'] = df['lesion_id'].str.strip()
    df['image_id'] = df['image_id'].str.strip()

    df.to_csv(out_path, index=False)
    print(f"cleaned metadata saved to: {out_path}")

if __name__ == "__main__":
    main()