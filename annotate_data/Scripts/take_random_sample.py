import pandas as pd
import os

from pathlib import Path

# Directory where this script lives
BASE_DIR = Path(__file__).resolve().parent

def take_subset_sample(input_file, output_file, sample_size=10000):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Make sure the first script finished!")
        return

    print(f"Reading {input_file}...")
    # Read the data
    df = pd.read_csv(input_file)
    
    total_rows = len(df)
    print(f"Total available rows: {total_rows}")

    if total_rows == 0:
        print("The file is empty.")
        return

    # Determine how many to take (can't take more than exists)
    actual_sample_n = min(sample_size, total_rows)
    
    print(f"Sampling {actual_sample_n} random rows...")
    
    # Randomly sample
    # random_state is set to 42 so the 'randomness' is repeatable if you run it again
    sampled_df = df.sample(n=actual_sample_n, random_state=42)

    # Save to new file
    sampled_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Done! Saved to {output_file}")

# --- Configuration ---
# Match these filenames to what you actually have on your disk
input_name = BASE_DIR / "../Dataset/Unlabled_Data/sampled_selftext_usable.csv" 
output_name = BASE_DIR / "../Dataset/Unlabled_Data/final_10k_sample.csv"

if __name__ == "__main__":
    take_subset_sample(input_name, output_name)