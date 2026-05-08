import pandas as pd
from pathlib import Path

# Directory where this script lives
BASE_DIR = Path(__file__).resolve().parent

# Load CSV
file = pd.read_csv(BASE_DIR / "../Dataset/Unlabled_Data/sampled_selftext_usable.csv")

# Select text column
df = file['selftext']

# Count words in each row
word_counts = df.apply(lambda x: len(str(x).split(" ")))

# Distribution of word counts
distribution = word_counts.value_counts().sort_index()

#print(distribution)
print(word_counts.describe())