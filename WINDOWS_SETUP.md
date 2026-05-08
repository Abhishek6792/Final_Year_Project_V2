# Setup & Usage Guide (Windows)

Welcome! This toolkit lets you annotate mental health-related Reddit posts using a completely free, completely local AI model running on your own computer via Ollama.

The dataset (`reddit_test.csv`) is already included — no scraping required.

## Prerequisites

1. **Install Python**: Download and install [Python for Windows](https://www.python.org/downloads/windows/). **Important:** During setup, check the box that says `Add python.exe to PATH`.
2. **Install Ollama**: Download and install [Ollama for Windows](https://ollama.com/download).
3. **Pull the Model**: Open Command Prompt (`cmd`) or PowerShell, and run:
   ```
   ollama pull dcarrascosa/medgemma-1.5-4b-it:Q4_K_M
   ```
   *This downloads MedGemma 1.5, a medical-domain AI model (~3.3GB).*

## Step 1: Setup the Python Environment

Open Command Prompt or PowerShell **in the folder where you unzipped this toolkit**.
*(Tip: In File Explorer, click the address bar, type `cmd` or `powershell`, and press Enter.)*

**Command Prompt (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **PowerShell Note:** If you get an error about "execution policy", run this command first:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

## Step 2: Make Sure Ollama is Running

Open a **separate** terminal window and run:
```
ollama serve
```
*(If Ollama is already running in your system tray, you can skip this step.)*

## Step 3: Run the Annotator

Back in your first terminal (with the virtual environment activated), run:

**Test run (5 rows) — recommended first:**
```cmd
python llm_annotator.py --input reddit_test.csv --limit 5
```

**Full run (all ~10,000 rows):**
```cmd
python llm_annotator.py --input reddit_test.csv
```

The annotator will process each row through MedGemma 1.5 and save the results to `reddit_test_annotated.jsonl`.

## Step 4: Send Back Results

When the annotator finishes, zip up the generated `reddit_test_annotated.jsonl` file and send it back!

## What the Annotator Does

For each Reddit post in `reddit_test.csv`, the AI labels it with:

| Signal Label | Description |
|---|---|
| `attention_dysregulation` | ADHD-related symptoms, focus issues |
| `anxious_affect` | Anxiety, panic, excessive worry |
| `autistic_trait_discussion` | Sensory issues, social difficulties, stimming |
| `emotional_instability` | Mood swings, splitting, intense anger |
| `depressive_affect` | Low mood, hopelessness, fatigue |
| `trauma_stress` | PTSD, flashbacks, abuse references |
| `crisis_self_harm` | Self-harm or suicidal language |
| `no_clear_signal` | No mental health signal detected |

Each annotation also includes a **severity score** (0-3) and a **confidence score** (0.0-1.0).

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure you activated the virtual environment and ran `pip install -r requirements.txt`. |
| `Failed to connect to Ollama` | Make sure `ollama serve` is running in a separate terminal window. |
| `ollama pull` is very slow | The model is ~3.3GB. Make sure you have a stable internet connection. |
| PowerShell "scripts disabled" error | Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` first. |
| Script seems stuck on a row | Some long posts take longer. Wait up to 2 minutes before worrying. |
