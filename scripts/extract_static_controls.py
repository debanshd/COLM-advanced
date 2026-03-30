# 26f5d162-1d80-41ed-a396-1c3905b1c7da
import json
import os
from datasets import load_dataset

def get_dataset(path, name=None, split="train", n_rows=250, seed=42):
    print(f"Loading {path} ({name if name else split})...")
    try:
        ds = load_dataset(path, name, split=split)
    except Exception as e:
        print(f"Failed to load split '{split}' for {path}, trying without specifying split... Error: {e}")
        try:
            ds_dict = load_dataset(path, name)
            splits = list(ds_dict.keys())
            if not splits:
                raise ValueError(f"No splits found for {path}")
            target_split = splits[0]
            for s in ["test", "validation", "train"]:
                if s in splits:
                    target_split = s
                    break
            print(f"Found splits: {splits}. Using '{target_split}'.")
            ds = ds_dict[target_split]
        except Exception as e2:
            print(f"Failed to load dataset {path} entirely: {e2}")
            return []
            
    shuffled = ds.shuffle(seed=seed)
    N = min(n_rows, len(shuffled))
    selected = shuffled.select(range(N))
    return selected

def normalize_gpqa(row, idx):
    q_id = row.get("Record ID") or f"gpqa_{idx}"
    question = row.get("Question", "")
    correct = row.get("Correct Answer", "")
    distractors = []
    if "Incorrect Answer 1" in row: distractors.append(row["Incorrect Answer 1"])
    if "Incorrect Answer 2" in row: distractors.append(row["Incorrect Answer 2"])
    if "Incorrect Answer 3" in row: distractors.append(row["Incorrect Answer 3"])
    
    return {
        "question_id": str(q_id),
        "dataset_source": "gpqa_extended",
        "prompt": question,
        "correct_answer": correct,
        "distractors": distractors
    }

def normalize_mmlu_pro(row, idx):
    q_id = row.get("question_id", f"mmlu_{idx}")
    question = row.get("question", "")
    options = row.get("options", [])
    answer_idx = row.get("answer_index")
    
    correct = ""
    distractors = []
    if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
        correct = options[answer_idx]
        distractors = [opt for j, opt in enumerate(options) if j != answer_idx]
    else:
        correct = row.get("answer", "")
        distractors = [opt for opt in options if opt != correct]
        
    return {
        "question_id": str(q_id),
        "dataset_source": "mmlu_pro",
        "prompt": question,
        "correct_answer": correct,
        "distractors": distractors
    }

def main():
    gpqa_ds = get_dataset("Idavidrein/gpqa", "gpqa_extended", split="train", n_rows=250)
    gpqa_normalized = [normalize_gpqa(row, i) for i, row in enumerate(gpqa_ds)]
    
    mmlu_ds = get_dataset("TIGER-Lab/MMLU-Pro", split="test", n_rows=250)
    mmlu_normalized = [normalize_mmlu_pro(row, i) for i, row in enumerate(mmlu_ds)]
    
    combined = gpqa_normalized + mmlu_normalized
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_file = os.path.join(base_dir, "results", "raw_static_controls.jsonl")
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    print(f"Saving {len(combined)} rows to {out_file}...")
    with open(out_file, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
