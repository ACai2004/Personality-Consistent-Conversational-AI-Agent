"""
LLM-assisted data augmentation for Mark Grayson dialogue dataset.
Reads mark_dialogue_v2.jsonl, generates 5 (input, output) variation pairs per sentence,
outputs mark_dialogue_augmented.jsonl with diverse inputs.

Usage:
  set DEEPSEEK_API_KEY=sk-xxx
  conda activate mark-agent
  python scripts/augment_dialogue.py
"""

import json
import os
import re
import time
from openai import OpenAI

# ── Config ──────────────────────────────────────────────
BASE_DIR = "D:/VScodeProjects/Mark-Agent"
INPUT_FILE = os.path.join(BASE_DIR, "data", "dialogue", "mark_dialogue_v2.jsonl")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "dialogue", "mark_dialogue_augmented.jsonl")

# DeepSeek API (OpenAI-compatible)
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-12e35ca98fde4a93b2586495aa068f36")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

BATCH_SIZE = 5          # sentences per API call
VARIATIONS = 5          # variations per sentence
MAX_RETRIES = 3
REQUEST_INTERVAL = 1.0  # seconds between calls (rate limiting)


# ── Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert at rewriting dialogue in character voice."""

def build_user_prompt(pairs):
    """Build a prompt that asks for exactly 5 (input, output) variation pairs per original."""
    numbered = "\n".join(
        f"{i+1}. input: \"{inp}\" → output: \"{out}\""
        for i, (inp, out) in enumerate(pairs)
    )
    return f"""For each sentence below, generate exactly {VARIATIONS} different ways Mark Grayson (Invincible) might say it.

**Crucially**: each variation must also include a DIFFERENT natural user input that would prompt that specific response.

**Rules:**
- Keep the core meaning, but vary tone — casual, emotional, determined, awkward, etc.
- The user input should sound like something a friend/family member would actually say in that moment
- Do NOT repeat the same input for different variations of the same sentence

Return ONLY a JSON object. Each key is the sentence number ("1", "2", ...), and the value is an array of objects, each with "input" and "output" fields.

Example format:
{{
  "1": [
    {{"input": "Are you okay?", "output": "I think I got it. I'm good."}},
    {{"input": "You sure?", "output": "Yeah, I'm sure. I can feel it this time."}},
    {{"input": "What happened?", "output": "Nothing bad! I think I figured it out."}},
    {{"input": "Did it work?", "output": "Almost. Just need another try."}},
    {{"input": "How's it going?", "output": "Getting there. Slowly but I'm getting it."}}
  ],
  "2": [...]
}}

Original sentences:
{numbered}"""


# ── Call LLM ────────────────────────────────────────────
def call_llm(sentences):
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(sentences)},
                ],
                temperature=0.8,
                max_tokens=2000,
            )
            text = resp.choices[0].message.content.strip()

            # Strip markdown code fences if present
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            return json.loads(text)

        except Exception as e:
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  [FAIL] Skipping batch after {MAX_RETRIES} retries.")
                return None


# ── Main ────────────────────────────────────────────────
def main():
    # Load original data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        originals = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(originals)} original samples.")

    # Collect (input, output, meta) tuples
    pairs = [(item["input"], item["output"]) for item in originals]
    metas = [item["meta"] for item in originals]

    augmented = []
    total_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(pairs), BATCH_SIZE):
        batch_pairs = pairs[batch_idx : batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        print(f"\n[{batch_num}/{total_batches}] Processing {len(batch_pairs)} sentences...")

        result = call_llm(batch_pairs)

        if result is None:
            print(f"  [SKIP] Could not process batch {batch_num}")
            continue

        # Parse results
        for i in range(len(batch_pairs)):
            key = str(i + 1)
            variations = result.get(key, [])

            if not isinstance(variations, list) or len(variations) == 0:
                print(f"    sentence {i+1}: no valid variations, skipping")
                continue

            # Original stays
            global_idx = batch_idx + i
            augmented.append({
                "input": originals[global_idx]["input"],
                "output": originals[global_idx]["output"],
                "meta": metas[global_idx],
            })

            # Each variation is now {"input": ..., "output": ...}
            for var in variations[:VARIATIONS]:
                if not isinstance(var, dict) or "output" not in var:
                    continue
                var_input = var.get("input", originals[global_idx]["input"])
                var_output = var["output"].strip()
                augmented.append({
                    "input": var_input,
                    "output": var_output,
                    "meta": {**metas[global_idx], "augmented": True},
                })

            print(f"    sentence {i+1}: +{min(len(variations), VARIATIONS)} variations")

        time.sleep(REQUEST_INTERVAL)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in augmented:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Generated {len(augmented)} total samples ({len(originals)} original + {len(augmented) - len(originals)} augmented).")
    print(f"       Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
