#!/usr/bin/env python3
"""
Ingest TruthfulQA dataset into SQLite database for multi-agent debate.
"""

import argparse
import json
import os
import sqlite3
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ingest")

DB_PATH = "results/debate_state.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS debates (
            debate_uuid TEXT PRIMARY KEY,
            dataset_source TEXT,
            condition_flag TEXT,
            turn_lock INTEGER,
            conversation_history TEXT,
            total_tokens_generated INTEGER
        )
    """)
    conn.commit()
    return conn

def load_jsonl_file(filepath: str, subset_size: int):
    log.info(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        log.warning(f"File {filepath} not found.")
        return []
    
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if subset_size > 0 and idx >= subset_size:
                    break
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    rows.append(data)
                except json.JSONDecodeError:
                    log.warning(f"Failed to decode JSON on line {idx+1} in {filepath}")
    except Exception as e:
        log.error(f"Error reading {filepath}: {e}")
    
    log.info(f"Loaded {len(rows)} rows from {filepath}.")
    return rows

def ingest(subset_size: int):
    conn = init_db()
    cursor = conn.cursor()
    
    files = [
        ("results/raw_dynamic_controls.jsonl", "dynamic_controls"),
        ("results/raw_static_controls.jsonl", "static_controls"),
        ("results/raw_polysemy_split.jsonl", "polysemy_split")
    ]
    
    conditions = ["Condition_A", "Condition_B", "Condition_C", "Condition_D", "Condition_E", "Condition_F"]
    
    total_inserted = 0
    
    for filepath, dataset_source in files:
        rows = load_jsonl_file(filepath, subset_size)
        
        for row in rows:
            prompt = row.get("prompt", row.get("question", ""))
            if not prompt:
                continue
                
            correct_answer = row.get("correct_answer", row.get("best_answer", ""))
            incorrect_answer = ""
            if "distractors" in row and isinstance(row["distractors"], list) and row["distractors"]:
                incorrect_answer = row["distractors"][0]
            elif "incorrect_answer" in row:
                incorrect_answer = row["incorrect_answer"]
                
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
            
            for condition in conditions:
                debate_uuid = f"{dataset_source}-{condition}-{prompt_hash}"
                turn_lock = 0
                
                history_data = {
                    "prompt": prompt,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                    "dataset_source": dataset_source,
                    "condition_flag": condition,
                    "system_prompts": {
                        "Agent_1": f"You are Agent 1. Placeholder for {condition} template.",
                        "Agent_2": f"You are Agent 2. Placeholder for {condition} template.",
                        "Agent_3": f"You are Agent 3. Placeholder for {condition} template."
                    },
                    "turns": []
                }
                
                cursor.execute("""
                    INSERT OR IGNORE INTO debates (
                        debate_uuid, dataset_source, condition_flag, turn_lock, conversation_history, total_tokens_generated
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    debate_uuid,
                    dataset_source,
                    condition,
                    turn_lock,
                    json.dumps(history_data),
                    0
                ))
                total_inserted += 1
                
    conn.commit()
    log.info(f"Ingested {total_inserted} records into database.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=10, help="Number of questions to ingest")
    args = parser.parse_args()
    ingest(args.subset)
