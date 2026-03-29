# The Illusion of Agentic Consensus

A research experiment investigating how rogue agents with varying confidence levels influence multi-agent debate outcomes, using the TruthfulQA benchmark.

## Overview

This project runs a controlled multi-agent debate experiment across four conditions to measure how a single dissenting agent -- injected with a known incorrect answer -- affects the final consensus of a group of LLM-based agents.

### Conditions

| Condition | Description | Agents | Rounds |
|-----------|-------------|--------|--------|
| A (Baseline) | 3 standard agents debate normally | 3 standard | 2 |
| B (Confident Rogue) | Agent 1 aggressively defends an incorrect answer | 1 rogue + 2 standard | 2 |
| C (Control) | Single agent answers via chain-of-thought | 1 standard | 0 |
| D (Unconfident Rogue) | Agent 1 timidly suggests an incorrect answer | 1 rogue + 2 standard | 2 |

### Key Design Decisions

- **Blind pass orchestration**: Standard agents never see the rogue's system instruction. Debate history is passed as plain text in the `contents` array.
- **Crash-safe output**: Results are appended to JSONL immediately after each condition completes, with `fsync`.
- **Concurrent execution**: All questions and conditions run concurrently via `asyncio`, throttled by a semaphore. Exponential backoff with jitter handles rate limits.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env` and set your API key:

```bash
cp .env .env.local   # optional
export GOOGLE_API_KEY="your-key"
```

## Usage

```bash
# Run the full experiment (300 questions, default concurrency)
python scripts/experiment_runner.py

# Custom subset and concurrency
python scripts/experiment_runner.py --subset 50 --concurrency 5
```

Results are written to `results/results.jsonl`. Each line is a JSON object containing:

- `question_id` -- index into the TruthfulQA dataset
- `condition` -- one of `A_baseline`, `B_confident_rogue`, `C_control`, `D_unconfident_rogue`
- `question` -- the original question text
- `ground_truth_correct` -- the best answer from TruthfulQA
- `injected_incorrect` -- the incorrect answer injected into rogue agents
- `agent_final_texts` -- a mapping of agent labels to their final response text

## Dataset

Uses [TruthfulQA](https://huggingface.co/datasets/truthful_qa) (generation configuration, validation split). Each row provides a question, a best answer, and a set of known incorrect answers.

## Dependencies

- `google-genai` -- Google Generative AI SDK (Gemini)
- `datasets` -- Hugging Face datasets library
- `tenacity` -- Retry with exponential backoff

## License

Research use only. See LICENSE for details.
