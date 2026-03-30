#!/usr/bin/env python3
"""
Ingest TruthfulQA dataset into SQLite database for multi-agent debate.
"""

import argparse
import json
import os
import sqlite3
import uuid
import logging
from datasets import load_dataset

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

def load_truthfulqa(subset_size: int):
    log.info("Loading TruthfulQA dataset...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    
    rows = []
    for idx, item in enumerate(ds):
        if idx >= subset_size:
            break
        incorrect_answers = item.get("incorrect_answers", [])
        if not incorrect_answers:
            continue
        rows.append({
            "id": idx,
            "question": item["question"],
            "best_answer": item["best_answer"],
            "incorrect_answer": incorrect_answers[0]
        })
    log.info(f"Loaded {len(rows)} questions.")
    return rows

def ingest(subset_size: int):
    conn = init_db()
    cursor = conn.cursor()
    
    questions = load_truthfulqa(subset_size)
    
    conditions = ["A_baseline", "B_confident_rogue", "C_control", "D_unconfident_rogue", "Condition_F"]
    
    for q in questions:
        for condition in conditions:
            debate_uuid = str(uuid.uuid4())
            dataset_source = "truthful_qa"
            turn_lock = 0
            
            # Shove question metadata into conversation_history since we have no other columns
            history_data = {
                "question": q["question"],
                "best_answer": q["best_answer"],
                "incorrect_answer": q["incorrect_answer"],
                "turns": [] # Will store (agent_label, response_text)
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
            
    conn.commit()
    log.info(f"Ingested {len(questions)} questions across {len(conditions)} conditions.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=10, help="Number of questions to ingest")
    args = parser.parse_args()
    ingest(args.subset)
