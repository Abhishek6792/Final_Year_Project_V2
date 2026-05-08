import pandas as pd

from pathlib import Path

# Directory where this script lives
BASE_DIR = Path(__file__).resolve().parent

# Load the CSV
df = pd.read_csv(BASE_DIR / "../Dataset/Unlabled_Data/final_10k_sample.csv")

# Split into parts
part1 = df.iloc[:5000]
part2 = df.iloc[5000:8000]
part3 = df.iloc[8000:10000]

# Save to new CSV files
part1.to_csv("Avi_5000.csv", index=False)
part2.to_csv("Ekam_3000.csv", index=False)
part3.to_csv("Ayush_2000.csv", index=False)

print("CSV split completed.")