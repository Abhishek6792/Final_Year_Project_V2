import argparse
import glob
import json
import os
import re
import sys
import time

import pandas as pd
import requests

# Constants
VALID_SIGNALS = {
    "attention_dysregulation",
    "anxious_affect",
    "autistic_trait_discussion",
    "emotional_instability",
    "depressive_affect",
    "trauma_stress",
    "crisis_self_harm",
    "no_clear_signal",
}

GUIDELINES = """
You are an expert psychological language annotator.

You must select ONLY labels from this exact taxonomy:
- attention_dysregulation
- anxious_affect
- autistic_trait_discussion
- emotional_instability
- depressive_affect
- trauma_stress
- crisis_self_harm
- no_clear_signal

Rules:
- Use no_clear_signal ONLY when no other label applies
- Multiple labels are allowed

Severity scale:
0 = none
1 = mild
2 = moderate
3 = severe/crisis
"""

JSON_INSTRUCTION = """
Respond ONLY with a raw JSON object containing:
{
  "label_signals": ["label1", "label2"],
  "severity": 0,
  "annotator_confidence": 0.0
}
"""

OLLAMA_URL = "http://localhost:11434/api/generate"

# Shared requests session
session = requests.Session()


def generate_with_ollama(prompt, model, retries=3):
    """Generate annotation using Ollama with retry support."""

    payload = {
        "model": model,
        "prompt": prompt + "\n\n" + JSON_INSTRUCTION,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }

    for attempt in range(retries):
        try:
            response = session.post(
                OLLAMA_URL,
                json=payload,
                timeout=120,
            )

            response.raise_for_status()
            return response.json().get("response", "{}")

        except Exception as e:
            if attempt == retries - 1:
                raise e

            print(f"Retry {attempt + 1}/{retries} after error: {e}")
            time.sleep(2)


def extract_json(text):
    """Extract JSON object from noisy model output."""

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model response")

    return json.loads(match.group())


def get_last_processed_index(output_jsonl):
    """Resume safely using source_index from existing JSONL."""

    if not os.path.exists(output_jsonl):
        return -1

    last_index = -1

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                last_index = row.get("source_index", last_index)
            except Exception:
                continue

    return last_index


def validate_result(result_dict):
    """Validate and sanitize model output."""

    labels = [
        signal
        for signal in result_dict.get("label_signals", [])
        if signal in VALID_SIGNALS
    ]

    if not labels:
        labels = ["no_clear_signal"]

    try:
        severity = int(result_dict.get("severity", 0))
    except Exception:
        severity = 0

    severity = max(0, min(3, severity))

    try:
        confidence = float(result_dict.get("annotator_confidence", 0.0))
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "label_signals": labels,
        "severity": severity,
        "annotator_confidence": confidence,
    }


def annotate_text(text, model):
    """Run annotation pipeline for a single text."""

    prompt = f"{GUIDELINES}\n\nAnnotate:\n{text}"

    response_text = generate_with_ollama(prompt, model)

    parsed = extract_json(response_text)

    return validate_result(parsed)


def process_file(input_csv, output_jsonl, model, limit=None):
    """Process one CSV file into JSONL annotations."""

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found")
        sys.exit(1)

    # Resume support
    last_processed_index = get_last_processed_index(output_jsonl)

    if last_processed_index >= 0:
        print(
            f"--- Resuming from source_index {last_processed_index + 1} ---"
        )

    df = pd.read_csv(input_csv)

    remaining_df = df[df.index > last_processed_index]

    if remaining_df.empty:
        print("All rows already processed.")
        return

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True) \
        if os.path.dirname(output_jsonl) else None

    success_count = 0
    fail_count = 0
    processed_this_run = 0

    buffer = []

    print(f"Processing {len(remaining_df)} remaining rows...")

    with open(output_jsonl, "a", encoding="utf-8") as out_file:

        for idx, row in remaining_df.iterrows():

            if limit and processed_this_run >= limit:
                print("Reached session limit.")
                break

            text = str(row.get("selftext", "")).strip()

            print(f"[{idx + 1}/{len(df)}] Annotating...")

            # Handle empty rows consistently
            if not text or text.lower() == "nan":

                final_row = {
                    "source_index": int(idx),
                    "text": "",
                    "label_signals": ["no_clear_signal"],
                    "severity": 0,
                    "annotator_confidence": 1.0,
                }

                buffer.append(json.dumps(final_row) + "\n")

                success_count += 1
                processed_this_run += 1

                continue

            try:
                annotation = annotate_text(text, model)

                final_row = {
                    "source_index": int(idx),
                    "text": text,
                    **annotation,
                }

                success_count += 1

            except Exception as e:
                print(f"  -> Error at row {idx}: {e}")

                final_row = {
                    "source_index": int(idx),
                    "text": text,
                    "label_signals": ["no_clear_signal"],
                    "severity": 0,
                    "annotator_confidence": 0.0,
                    "error": str(e),
                }

                fail_count += 1

            buffer.append(json.dumps(final_row, ensure_ascii=False) + "\n")

            processed_this_run += 1

            # Checkpoint every 10 rows
            if len(buffer) >= 10:
                out_file.writelines(buffer)
                out_file.flush()
                os.fsync(out_file.fileno())

                print(
                    f"--- Checkpoint: Saved {len(buffer)} rows to {output_jsonl} ---"
                )

                buffer = []

        # Final flush
        if buffer:
            out_file.writelines(buffer)
            out_file.flush()
            os.fsync(out_file.fileno())

    print("\nSession complete!")
    print(f"Success: {success_count}")
    print(f"Failed : {fail_count}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file or directory",
    )

    parser.add_argument(
        "--model",
        default="dcarrascosa/medgemma-1.5-4b-it:Q4_K_M",
        help="Ollama model name",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum rows to process this session",
    )

    args = parser.parse_args()

    if os.path.isdir(args.input):

        out_dir = os.path.join(args.input, "annotated")
        os.makedirs(out_dir, exist_ok=True)

        csv_files = glob.glob(os.path.join(args.input, "*.csv"))

        if not csv_files:
            print("No CSV files found.")
            return

        for csv_file in csv_files:

            base_name = os.path.splitext(
                os.path.basename(csv_file)
            )[0]

            out_path = os.path.join(
                out_dir,
                f"{base_name}_annotated.jsonl",
            )

            print(f"\n=== Processing {csv_file} ===")

            process_file(
                input_csv=csv_file,
                output_jsonl=out_path,
                model=args.model,
                limit=args.limit,
            )

    else:

        out_path = (
            os.path.splitext(args.input)[0]
            + "_annotated.jsonl"
        )

        process_file(
            input_csv=args.input,
            output_jsonl=out_path,
            model=args.model,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()