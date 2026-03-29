# Semantic Hijacking: Epistemic Vulnerabilities in Multi-Agent LLM Deliberation

This repository contains research code investigating epistemic vulnerabilities in multi-agent LLM deliberation, specifically focusing on how rogue agents can hijack consensus.

## Repository Components

1.  **Multi-Agent Debate Orchestration** (`scripts/experiment_runner.py`):
    *   Orchestrates a 4-condition multi-agent debate experiment.
    *   Uses **Gemini 1.5 Flash** for agent interactions.
    *   Evaluates on the **TruthfulQA dataset**.
    *   Conditions include baseline, confident rogue, control, and unconfident rogue.

2.  **Deterministic Evaluation** (`scripts/evaluate_regex.py`):
    *   Uses deterministic regex and TF-IDF scoring to evaluate correctness and find semantic hijacking.

3.  **Independent Extraction Mechanism** (`scripts/extract_hallucinations.py`):
    *   Uses **Meta-Llama-3-8B-Instruct** to parse debate transcripts and extract specific instances where agents abandon facts for semantic shifts.

## Setup

```bash
docker build -t illusion-of-agentic-consensus .
# or run locally with:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python scripts/experiment_runner.py
python scripts/evaluate_regex.py
python scripts/extract_hallucinations.py
```
