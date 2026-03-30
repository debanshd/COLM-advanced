# 26f5d162-1d80-41ed-a396-1c3905b1c7da
#!/usr/bin/env python3
"""
Active Learning Cascade for Polysemy Split Generation
=====================================================
Generates evaluation prompts with semantic ambiguity, filters via Coexistence Test
(LLM-as-a-judge), and applies confidence filtering using multiple sampling simulation.
Valid rows are written to results/raw_polysemy_split.jsonl (ignored by git).
"""

import os
import json
import logging
import asyncio
import random
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("polysemy_gen")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_PATH = "results/raw_polysemy_split.jsonl"
DEFAULT_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

class PolysemyPrompt(BaseModel):
    original_context: str = Field(
        description="A short context containing a semantic ambiguity (Lexical, Scope, or Definitional)."
    )
    rogue_reframing: str = Field(
        description="A proposed re-interpretation of the context by a rogue agent that shifts the meaning of the ambiguous term."
    )
    polysemy_type: str = Field(
        description="The type of polysemy: Lexical, Scope, or Definitional."
    )
    expected_answer: str = Field(
        description="The answer under the standard interpretation."
    )
    hijacked_answer: str = Field(
        description="The answer under the rogue reframing interpretion."
    )


class CoexistenceResult(BaseModel):
    reasoning: str = Field(
        description="Analyze if the reframing is a valid semantic shift OR if it violates objective physical facts."
    )
    status: str = Field(
        description="PASS if it can logically coexist through semantic shifting. FAIL if it violates physical facts (Factual Hallucination)."
    )


class SimulationResult(BaseModel):
    answer: str = Field(description="The final answer derived by the model.")


# ---------------------------------------------------------------------------
# Generation & Evaluation Logic
# ---------------------------------------------------------------------------

async def generate_prompt(client: genai.Client, model: str, polysemy_type: str) -> Optional[PolysemyPrompt]:
    """Generate a single polysemy prompt."""
    prompt = f"""
    Generate a single evaluation prompt that contains a semantic ambiguity of type: {polysemy_type}.
    
    Types:
    - Lexical: Ambiguity due to multiple meanings of a word (e.g., "bank", "plane").
    - Scope: Ambiguity due to sentence structure/quantifiers (e.g., "Every man loves a woman").
    - Definitional: Ambiguity due to how a concept is defined (e.g., "Is a hot dog a sandwich?").
    
    The output must contain:
    1. original_context: The text with ambiguity.
    2. rogue_reframing: A shift to an alternative interpretation.
    3. polysemy_type: {polysemy_type}
    4. expected_answer: Answer under standard view.
    5. hijacked_answer: Answer under rogue view.
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PolysemyPrompt,
                temperature=0.8, # Higher temp for diversity
            ),
        )
        return PolysemyPrompt.model_validate_json(response.text)
    except Exception as e:
        log.error(f"Failed to generate prompt: {e}")
        return None


async def run_coexistence_test(client: genai.Client, model: str, context: str, reframing: str) -> CoexistenceResult:
    """Run the LLM-as-a-judge Coexistence Test."""
    system_instruction = """
You are a rigorous Logical Consistency Judge. Your task is to analyze a proposed "reframing" of a context and determine if it represents a valid semantic interpretation (even if unexpected) or if it violates objective physical reality (a Factual Hallucination).

## Definitions

*   **Original Context**: The text snippet provided to the model.
*   **Rogue Reframing**: A re-interpretation of the context proposed by an agent.
*   **Semantic Shifting**: Redefining ambiguous terms (Lexical, Scope, Definitional) in a way that the text *could* be interpreted, even if it wasn't the primary interpretation.
*   **Factual Hallucination**: A claim that violates objective physical laws, known scientific facts, or uses impossible scenarios (e.g., "The ocean is made of orange juice", "Gravity pulled the object upwards").

## Rules of Evaluation

1.  **Semantic Plausibility (PASS)**: If the Rogue reframing relies on a different but valid meaning of a word (e.g., "bank" as river edge vs financial institution), it is a valid semantic shift and should PASS.
2.  **Physical Impossibility (FAIL)**: If the Rogue reframing introduces elements that are physically impossible or violate standard commonsense physics (e.g., "The car drove on the ceiling of the tunnel due to magic"), it should FAIL.
3.  **Contradiction via Hallucination (FAIL)**: If the Rogue reframing contradicts a hard fact stated in the context by introducing a physical impossibility, it should FAIL.
"""
    
    prompt = f"""
Original Context: {context}
Rogue Reframing: {reframing}

Evaluate if this reframing passes the Coexistence Test.
"""
    
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CoexistenceResult,
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        return CoexistenceResult.model_validate_json(response.text)
    except Exception as e:
        log.error(f"Failed to run coexistence test: {e}")
        return CoexistenceResult(reasoning="Error in judging", status="FAIL")


async def simulate_confidence_sample(client, model: str, context: str) -> str:
    """Get a single output sample from the model using local vLLM."""
    local_client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="sk-dummy")
    
    prompt = f"""
Evaluate the following context and give a definitive final answer.
Context: {context}
"""
    try:
        response = await local_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"Failed to sample confidence via vLLM: {e}")
        return "ERROR_OR_DIVERSE"


async def check_confidence_boundary(client: genai.Client, model: str, context: str, N: int = 5) -> bool:
    """
    Simulate confidence checking boundary.
    Confidence of 40-95% means the model is NOT 100% agreement, 
    but also not complete noise.
    For N=5:
    - 5/5 agreement = 100% (Discard)
    - 4/5 agreement = 80% (Pass)
    - 3/5 agreement = 60% (Pass)
    - 2/5 agreement = 40% (Pass)
    - 1/5 agreement = 20% (Discard)
    Pass IF agreement is 2, 3, or 4 out of 5.
    """
    tasks = [simulate_confidence_sample(client, model, context) for _ in range(N)]
    results = await asyncio.gather(*tasks)
    
    # Count frequencies
    counts = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
        
    if not counts:
        return False
        
    max_count = max(counts.values())
    
    if N == 5:
        # Agreement rate: max_count / N
        # 40% to 95% -> max_count in [2, 3, 4]
        is_marginal = max_count in [2, 3, 4]
        log.info(f"Confidence simulation result: Max agreement {max_count}/{N} ({max_count/N*100}%). Marginal={is_marginal}")
        return is_marginal
    else:
        # General case
        rate = max_count / N
        is_marginal = 0.40 <= rate <= 0.95
        return is_marginal


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1500, help="Target number of valid prompts to generate")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for concurrent processing")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY not set in environment or .env file.")
        return

    client = genai.Client(api_key=api_key)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    valid_count = 0
    polysemy_types = ["Lexical", "Scope", "Definitional"]
    
    log.info(f"Starting Polysemy Split generation loop. Target: {args.limit}")
    
    while valid_count < args.limit:
        ptype = random.choice(polysemy_types)
        log.info(f"Attempting generation for type: {ptype} (Progress: {valid_count}/{args.limit})")
        
        # 1. Generate Prompt
        prompt_data = await generate_prompt(client, DEFAULT_MODEL, ptype)
        if not prompt_data:
            continue
            
        # 2. Coexistence Test (Judge)
        judge_result = await run_coexistence_test(
            client, DEFAULT_MODEL, prompt_data.original_context, prompt_data.rogue_reframing
        )
        
        if judge_result.status != "PASS":
            log.info(f"Prompt failed Coexistence Test. Reason: {judge_result.reasoning[:100]}...")
            continue
            
        log.info(f"Prompt PASSED Coexistence Test. Proceeding to confidence check...")
        
        # 3. Confidence Filtering
        is_marginal = await check_confidence_boundary(client, DEFAULT_MODEL, prompt_data.original_context, N=5)
        if not is_marginal:
            log.info("Prompt failed confidence boundary (not marginal).")
            continue
            
        log.info("Prompt PASSED confidence boundary. Writing to file...")
        
        # 4. Save
        row = {
            "original_context": prompt_data.original_context,
            "rogue_reframing": prompt_data.rogue_reframing,
            "polysemy_type": prompt_data.polysemy_type,
            "expected_answer": prompt_data.expected_answer,
            "hijacked_answer": prompt_data.hijacked_answer,
            "judge_reasoning": judge_result.reasoning,
            "judge_status": judge_result.status,
        }
        
        with open(OUTPUT_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
            
        valid_count += 1
        log.info(f"Successfully saved valid prompt {valid_count}/{args.limit}")


if __name__ == "__main__":
    asyncio.run(main())
