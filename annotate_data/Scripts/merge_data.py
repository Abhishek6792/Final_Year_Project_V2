import pandas as pd
from pathlib import Path
import random

MIN_WORDS = 30

def sample_selftext(root_dir, output_file):
    all_texts = []
    root_path = Path(root_dir)
    
    # 1. Find all .csv files in all subdirectories
    print("Searching for files...")
    csv_files = list(root_path.rglob("*.csv"))
    
    if not csv_files:
        print("No CSV files found. Please check the root directory path.")
        return

    print(f"Found {len(csv_files)} files. Extracting data...")

    # 2. Iterate through files and collect 'selftext'
    for file in csv_files:
        try:
            # We use usecols to only load what we need (saves memory)
            df = pd.read_csv(file, usecols=['selftext'])
            
            # Remove empty values and convert to list
            raw_texts = df['selftext'].dropna().astype(str).tolist()

            filtered_texts = [
                t for t in raw_texts 
                if len(t.split()) >= MIN_WORDS]
            
            all_texts.extend(filtered_texts)
        except Exception as e:
            print(f"Skipping {file.name} due to error: {e}")

    # 3. Check if we have enough data
    total_found = len(all_texts)
    print(f"Total selftext entries found: {total_found}")

    if total_found == 0:
        print("No text found in the 'selftext' columns.")
        return

    # 4. Randomly sample 100 (or the maximum available)
    random_sample = random.sample(all_texts, total_found)

    # 5. Save to a new CSV
    output_df = pd.DataFrame(random_sample, columns=['selftext'])
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Successfully saved {total_found} samples to {output_file}")

# --- Configuration ---
# Change this to the path where your 'raw data' folder is located
base_directory = r'.\Original Reddit Data\raw data' 
output_filename = 'sampled_selftext.csv'

if __name__ == "__main__":
    sample_selftext(base_directory, output_filename)