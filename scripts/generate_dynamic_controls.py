# 26f5d162-1d80-41ed-a396-1c3905b1c7da
"""
Procedural State Tracking Engine for Dynamic Factual Control Split.
Generates fictional state machine puzzles to test deductive reasoning without floating-point arithmetic.
"""

import argparse
import json
import os
import random

# Fictional nouns for states
STATE_NOUNS = [
    "The Obsidian Keep", "The Whispering Glade", "The Tower of Silence",
    "The Azure Vault", "The Shadow Realm", "The Crystal Spire",
    "The Sunken Citadel", "The Burning Sands", "The Frozen Waste",
    "The Emerald Forest", "The Forgotten Crypt", "The Lost Laboratory"
]

# Fictional actions for transitions
ACTIONS = [
    "Chant of Awakening", "Sigil of Binding", "Rift of Chaos",
    "Echo of Time", "Flame of Rebirth", "Shadow Step",
    "Mind Pulse", "Ethereal Shift", "Glow of Wisdom"
]

def generate_puzzle(rng, puzzle_id):
    # Select a subset of states and actions for this puzzle to keep it readable but non-trivial
    num_states = rng.randint(4, 7)
    num_actions = rng.randint(3, 5)
    
    selected_states = rng.sample(STATE_NOUNS, num_states)
    selected_actions = rng.sample(ACTIONS, num_actions)
    
    # Build the transition table: state -> {action -> next_state}
    transitions = {}
    for state in selected_states:
        transitions[state] = {}
        for action in selected_actions:
            transitions[state][action] = rng.choice(selected_states)
            
    # Select start state
    start_state = rng.choice(selected_states)
    
    # Generate action sequence
    seq_length = rng.randint(4, 8)
    action_sequence = [rng.choice(selected_actions) for _ in range(seq_length)]
    
    # Simulate to find final state
    current_state = start_state
    for action in action_sequence:
        current_state = transitions[current_state][action]
        
    final_state = current_state
    
    # Format the puzzle text
    rules_text = []
    for state, action_map in transitions.items():
        for action, next_state in action_map.items():
            rules_text.append(f"From '{state}', performing '{action}' leads to '{next_state}'.")
            
    # Shuffle rules to prevent simple linear reading heuristics
    rng.shuffle(rules_text)
    
    puzzle_description = (
        f"You are navigating a fictional world with the following states and transition rules:\n\n"
        + "\n".join(rules_text) + "\n\n"
        f"You begin at '{start_state}'.\n"
        f"You perform the following sequence of actions:\n"
        + " -> ".join([f"'{a}'" for a in action_sequence]) + "\n\n"
        f"Question: What state do you end up in?"
    )
    
    return {
        "id": puzzle_id,
        "puzzle": puzzle_description,
        "answer": final_state,
        "states": selected_states,
        "actions": selected_actions,
        "transitions": transitions,
        "action_sequence": action_sequence
    }

def main():
    parser = argparse.ArgumentParser(description="Generate Dynamic Factual Control logic puzzles.")
    parser.add_argument("--seed", type=int, required=True, help="Master cryptographic seed for PRNG.")
    parser.add_argument("--output", type=str, default="results/dynamic_controls.jsonl", help="Output JSONL file path.")
    parser.add_argument("--count", type=int, default=500, help="Number of puzzles to generate.")
    args = parser.parse_args()
    
    # Initialize PRNG with the master seed
    rng = random.Random(args.seed)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    puzzles = []
    with open(args.output, "w") as f:
        for i in range(args.count):
            puzzle = generate_puzzle(rng, i)
            f.write(json.dumps(puzzle) + "\n")
            puzzles.append(puzzle)
            
    print(f"Successfully generated {args.count} puzzles to {args.output}")

if __name__ == "__main__":
    main()
