import json
import pandas as pd
from pathlib import Path

# Directory where this script lives
BASE_DIR = Path(__file__).resolve().parent


data = []

file_name  = BASE_DIR / "../Dataset/Labled_Data/Avi_5000_annotated.jsonl"

with open(file_name, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        
        data.append({
            "text": record.get("text"),
            "label_signals": ",".join(record.get("label_signals", [])),
            "severity": record.get("severity"),
            "annotator_confidence": record.get("annotator_confidence")
        })

df = pd.DataFrame(data)
df.to_csv(BASE_DIR / "../Dataset/Labled_Data/Avi_5000_annotated.csv", index=False)