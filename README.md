# Mental Health LLM Annotator

## 🎯 Purpose
This toolkit (`mental-health-llm-annotator`) is a specialized data-annotation pipeline for the **Detection of Mental Disorders v2** project. It uses a local LLM (MedGemma 1.5 via Ollama) to annotate Reddit posts with mental health signal labels.

The dataset (`reddit_test.csv`) is **included in the repo** — no scraping or browser extension needed.

## 📦 Architecture

### Dataset (`reddit_test.csv`)
A pre-collected CSV file containing ~10,000 Reddit posts. Each row has three columns:
- An index/ID column
- `post` — the full text of the Reddit post
- `label` — an existing numeric label (used for reference, not consumed by the annotator)

### Annotator (`llm_annotator.py`)
A Python script that:
1. Reads the CSV and detects the text column automatically (`post`, `text`, `body`, etc.)
2. Filters out posts shorter than 15 words or 45 characters
3. Sends each post to a local Ollama model for annotation
4. Writes structured JSONL output with multi-label signals, severity (0-3), and confidence (0.0-1.0)

**Model:** `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` (MedGemma 1.5, medical-domain optimized)

**Fallback:** If the LLM fails on a row, it writes `no_clear_signal`, severity `0`, and confidence `0.0`.

## 🏷️ Signal Taxonomy

| Signal | Description |
|---|---|
| `attention_dysregulation` | ADHD symptoms, focus/impulsivity issues |
| `anxious_affect` | Anxiety, panic, excessive worry |
| `autistic_trait_discussion` | Sensory issues, social difficulties, stimming |
| `emotional_instability` | Mood swings, splitting, intense anger (BPD-adjacent) |
| `depressive_affect` | Low mood, hopelessness, anhedonia |
| `trauma_stress` | PTSD, flashbacks, abuse references |
| `crisis_self_harm` | Explicit self-harm or suicidal language |
| `no_clear_signal` | No mental health signal detected |

## 🚀 Quick Start

### For Windows Users
See [WINDOWS_SETUP.md](./WINDOWS_SETUP.md) for a complete step-by-step guide.

### For macOS / Linux Users
```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Make sure Ollama is running
ollama serve  # in a separate terminal

# 3. Test on a small sample first
python llm_annotator.py --input reddit_test.csv --limit 5

# 4. Full run
python llm_annotator.py --input reddit_test.csv
```

### CLI Options
| Flag | Description | Default |
|---|---|---|
| `--input` | Path to a CSV file or directory of CSVs | *(required)* |
| `--model` | Ollama model name | `dcarrascosa/medgemma-1.5-4b-it:Q4_K_M` |
| `--limit` | Max rows to process (for testing) | *(all rows)* |

## 📄 Output Format
The annotator produces a `.jsonl` file where each line is:
```json
{
  "text": "The original Reddit post text...",
  "source_platform": "reddit",
  "thread_id": "",
  "post_id": "40508",
  "label_signals": ["trauma_stress", "anxious_affect"],
  "severity": 2,
  "annotator_confidence": 0.92,
  "split": "unassigned"
}
```

## 📋 Handoff
Once annotation is complete, zip the `_annotated.jsonl` file and send it to the project maintainers for integration into the main dataset.

**Note to future developers:** This is purely a data processing pipeline. Do not add UI features, backend routing, or browser extensions here.
