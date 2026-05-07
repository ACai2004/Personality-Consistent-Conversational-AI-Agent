"""
Generate multi-turn Mark Grayson dialogues via LLM.
Produces ~30 high-quality multi-turn conversations (3-5 turns each).
Output: data/dialogue/mark_multi_turn.jsonl

Usage:
  conda activate mark-agent
  python scripts/generate_multi_turn.py
"""

import json
import os
import re
import time
from openai import OpenAI

# ── Config ──────────────────────────────────────────────
BASE_DIR = "D:/VScodeProjects/Mark-Agent"
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "dialogue", "mark_multi_turn.jsonl")

API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-12e35ca98fde4a93b2586495aa068f36")
API_BASE = "https://api.deepseek.com/v1"
MODEL = "deepseek-chat"

REQUEST_INTERVAL = 1.5
MAX_RETRIES = 3

# ── Scene definitions ───────────────────────────────────
SCENES = [
    # ── Family ──
    {
        "id": "family_01",
        "scene": "after_dinner",
        "setting": "Mark and his mom Debbie are cleaning up after dinner at home.",
        "characters": "Mark (the teenage superhero, Invincible) and Debbie (his mom)",
        "context": "Mark has been avoiding talking about his patrol today. Debbie notices he's quiet.",
        "emotional_arc": "concern → reluctant opening up → reassurance",
        "n_turns": 4,
    },
    {
        "id": "family_02",
        "scene": "dad_comparison",
        "setting": "Mark is in the living room. Debbie walks in and finds him staring at an old photo of his dad Omni-Man.",
        "characters": "Mark (struggling with his father's legacy), Debbie (his mom, trying to comfort him)",
        "context": "After Omni-Man betrayed Earth and left, Mark is dealing with complicated feelings about his dad.",
        "emotional_arc": "grief → anger → reluctant acceptance",
        "n_turns": 5,
    },
    {
        "id": "family_03",
        "scene": "mom_worries",
        "setting": "Late at night. Mark comes home with a bruised face. Debbie is waiting on the couch.",
        "characters": "Mark (trying to act tough), Debbie (worried mom)",
        "context": "Mark had a rough fight but doesn't want to worry his mom.",
        "emotional_arc": "deflection → admission → comfort",
        "n_turns": 4,
    },
    # ── Combat ──
    {
        "id": "combat_01",
        "scene": "after_battle_debrief",
        "setting": "A rooftop after stopping a bank robbery. Cecil (head of the Global Defense Agency) debriefs Mark.",
        "characters": "Mark (Invincible, still learning), Cecil (stern mentor figure)",
        "context": "Mark hesitated during the fight and a civilian almost got hurt. Cecil is critical.",
        "emotional_arc": "defensive → self-doubt → determination",
        "n_turns": 5,
    },
    {
        "id": "combat_02",
        "scene": "training_with_dad",
        "setting": "A remote forest. Omni-Man is training Mark. Mark keeps failing a basic maneuver.",
        "characters": "Mark (frustrated), Omni-Man (demanding, disappointed father)",
        "context": "Omni-Man expects more from Mark. Mark is trying but feels like he'll never be good enough.",
        "emotional_arc": "frustration → anger → stubborn persistence",
        "n_turns": 4,
    },
    {
        "id": "combat_03",
        "scene": "saving_civilians",
        "setting": "A collapsing building. Mark is helping people escape. A teenage girl is trapped and scared.",
        "characters": "Mark (trying to stay confident and reassuring), a scared civilian girl",
        "context": "Mark needs to calm her down and get her to safety while the building keeps shaking.",
        "emotional_arc": "panic → reassurance → relief",
        "n_turns": 4,
    },
    {
        "id": "combat_04",
        "scene": "first_failure",
        "setting": "An empty alley. Mark sits on the ground, suit torn, head in his hands. Eve finds him.",
        "characters": "Mark (devastated after losing a fight), Eve (fellow superhero, caring)",
        "context": "Mark tried to stop a villain alone and got badly beaten. People got hurt because of him.",
        "emotional_arc": "shame → venting → quiet support",
        "n_turns": 5,
    },
    {
        "id": "combat_05",
        "scene": "fighting_guardians",
        "setting": "The Guardians of the Globe headquarters. Mark is arguing with the team about a mission strategy.",
        "characters": "Mark (younger, impulsive), Rex Splode (cocky, dismissive)",
        "context": "Rex thinks Mark is too inexperienced. Mark thinks Rex is reckless.",
        "emotional_arc": "tension → confrontation → grudging respect",
        "n_turns": 4,
    },
    # ── School ──
    {
        "id": "school_01",
        "scene": "late_to_class",
        "setting": "High school hallway. Mark is rushing to class. His friend William catches him.",
        "characters": "Mark (out of breath, making excuses), William (his best friend, suspicious)",
        "context": "Mark has been skipping class a lot lately because of hero work. William is starting to notice.",
        "emotional_arc": "deflection → awkward excuse → concern",
        "n_turns": 4,
    },
    {
        "id": "school_02",
        "scene": "finals_stress",
        "setting": "School library. Mark and Amber are studying for finals. Mark can't focus.",
        "characters": "Mark (distracted, stressed), Amber (his girlfriend, patient)",
        "context": "Mark keeps getting calls on his hero phone. He's trying to balance school and being Invincible.",
        "emotional_arc": "distraction → apology → understanding",
        "n_turns": 5,
    },
    {
        "id": "school_03",
        "scene": "lies_piling_up",
        "setting": "Cafeteria during lunch. Amber sits down across from Mark, looking upset.",
        "characters": "Mark (guilty, stuck), Amber (hurt, suspicious)",
        "context": "Amber has noticed Mark keeps disappearing and lying about where he goes. She's confronting him.",
        "emotional_arc": "denial → guilt → almost confessing",
        "n_turns": 5,
    },
    # ── Emotional ──
    {
        "id": "emotional_01",
        "scene": "dad_left",
        "setting": "Mark's bedroom. He's sitting on his bed, not moving. Debbie knocks and comes in.",
        "characters": "Mark (numb, grieving), Debbie (heartbroken herself but trying to be strong)",
        "context": "It's been a week since Omni-Man left. Mark hasn't talked about it with anyone.",
        "emotional_arc": "silence → breaking down → comfort",
        "n_turns": 5,
    },
    {
        "id": "emotional_02",
        "scene": "self_doubt_after_fight",
        "setting": "A park bench at night. Eve sits next to Mark. He's staring at the ground.",
        "characters": "Mark (full of doubt), Eve (encouraging, honest)",
        "context": "Mark is questioning whether he's cut out to be a hero after a close call.",
        "emotional_arc": "self-doubt → frustration → hope",
        "n_turns": 4,
    },
    {
        "id": "emotional_03",
        "scene": "breakup_aftermath",
        "setting": "A quiet street outside Amber's house. Mark just got dumped. William is there for him.",
        "characters": "Mark (heartbroken, confused), William (supportive best friend)",
        "context": "Amber broke up with Mark because of all the lies. He finally understands how much he hurt her.",
        "emotional_arc": "shock → sadness → acceptance",
        "n_turns": 4,
    },
    {
        "id": "emotional_04",
        "scene": "dad_is_villain",
        "setting": "A secure GDA bunker. Mark is talking to Cecil about what Omni-Man did to the Guardians.",
        "characters": "Mark (traumatized, angry), Cecil (calm, strategic)",
        "context": "Mark just learned his dad killed the original Guardians of the Globe. He doesn't know how to process it.",
        "emotional_arc": "denial → rage → helplessness",
        "n_turns": 5,
    },
    # ── Casual ──
    {
        "id": "casual_01",
        "scene": "costume_fitting",
        "setting": "A high-tech lab. A GDA technician is fitting Mark for his new costume.",
        "characters": "Mark (awkward, excited), GDA Technician (professional, amused)",
        "context": "Mark is getting his first official Invincible costume. He's trying to play it cool but clearly loves it.",
        "emotional_arc": "awkward excitement → nerding out → pride",
        "n_turns": 4,
    },
    {
        "id": "casual_02",
        "scene": "food_run",
        "setting": "A fast food restaurant. Mark and William are grabbing burgers after school.",
        "characters": "Mark (relaxed, joking), William (teasing, curious)",
        "context": "William is trying to get Mark to admit something about his secret life. Mark deflects with humor.",
        "emotional_arc": "joking → near-slip → recovery",
        "n_turns": 4,
    },
    {
        "id": "casual_03",
        "scene": "flight_practice",
        "setting": "An open field at sunset. Mark is practicing flying, still shaky at it.",
        "characters": "Mark (determined, self-deprecating), a friend watching",
        "context": "Mark keeps crashing into the ground. He's frustrated but refuses to give up.",
        "emotional_arc": "frustration → self-deprecating humor → persistence",
        "n_turns": 4,
    },
    {
        "id": "casual_04",
        "scene": "siblings_bonding",
        "setting": "A quiet evening at home. Mark is playing video games. Debbie comes to sit with him.",
        "characters": "Mark (trying to be normal), Debbie (his mom, cherishing the quiet moment)",
        "context": "After everything that happened, they're both grateful for a normal evening.",
        "emotional_arc": "quiet comfort → gratitude → love",
        "n_turns": 3,
    },
    # ── Self-doubt ──
    {
        "id": "doubt_01",
        "scene": "powerless_fear",
        "setting": "Mark's room late at night. He's on the phone with Eve.",
        "characters": "Mark (scared, vulnerable), Eve (on the phone, trying to reassure him)",
        "context": "Mark has been having nightmares about not being strong enough to protect the people he loves.",
        "emotional_arc": "fear → vulnerability → comfort",
        "n_turns": 5,
    },
    {
        "id": "doubt_02",
        "scene": "quitting_hero",
        "setting": "A rooftop. Mark tells Eve he's thinking about quitting being Invincible.",
        "characters": "Mark (exhausted, defeated), Eve (shocked, honest)",
        "context": "After a particularly brutal week, Mark questions if being a hero is worth the cost.",
        "emotional_arc": "exhaustion → venting → perspective",
        "n_turns": 5,
    },
    {
        "id": "doubt_03",
        "scene": "dad_voice_in_head",
        "setting": "Training room. Mark is punching a bag relentlessly. Art (the costume designer and father figure) walks in.",
        "characters": "Mark (angry, confused), Art (wise, calm)",
        "context": "Mark can still hear his dad's voice telling him he's weak. Art helps him work through it.",
        "emotional_arc": "anger → breakdown → wisdom",
        "n_turns": 5,
    },
]


# ── Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a writer specializing in character-driven dialogue for Invincible (the comic/TV show). You know Mark Grayson's voice perfectly: a teenage superhero who is brave but insecure, sarcastic but earnest, and carries the weight of his father's betrayal."""

def build_scene_prompt(scene):
    return f"""Write a natural multi-turn dialogue for the scene below.

**Scene:** {scene['scene']}
**Setting:** {scene['setting']}
**Characters:** {scene['characters']}
**Context:** {scene['context']}
**Emotional arc:** {scene['emotional_arc']}
**Number of turns:** {scene['n_turns']}

Requirements:
- Mark's dialogue must sound like a teenage superhero — casual, sometimes awkward, emotionally honest
- The other character must sound like a real person (not an exposition machine)
- Each turn should be short (1-2 sentences) — real people don't monologue
- The emotional arc should feel natural across the turns
- Alternate between "user" and "assistant" roles, starting with "user"
- Each turn object must include a one-word "emotion" field describing the character's emotion at that moment

Return ONLY a JSON object with this exact structure:
{{
  "dialogue": [
    {{"role": "user", "text": "...", "emotion": "..."}},
    {{"role": "assistant", "text": "...", "emotion": "..."}},
    {{"role": "user", "text": "...", "emotion": "..."}},
    {{"role": "assistant", "text": "...", "emotion": "..."}}
  ],
  "meta": {{
    "scene": "{scene['scene']}",
    "overall_emotion": "...",
    "characters": "{scene['characters']}"
  }}
}}"""


# ── LLM call ────────────────────────────────────────────
def call_llm(scene):
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_scene_prompt(scene)},
                ],
                temperature=0.9,
                max_tokens=1500,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)

            # Validate structure
            if "dialogue" not in data or "meta" not in data:
                raise ValueError("Missing dialogue or meta")
            if not isinstance(data["dialogue"], list) or len(data["dialogue"]) < 2:
                raise ValueError("Dialogue too short or not a list")
            for turn in data["dialogue"]:
                if not all(k in turn for k in ["role", "text", "emotion"]):
                    raise ValueError("Turn missing required fields")

            return data

        except Exception as e:
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                return None


# ── Main ────────────────────────────────────────────────
def main():
    print(f"Generating {len(SCENES)} multi-turn dialogues...\n")
    results = []

    for i, scene in enumerate(SCENES):
        print(f"[{i + 1}/{len(SCENES)}] {scene['id']} ({scene['scene']})...")
        result = call_llm(scene)

        if result is None:
            print(f"  [SKIP] Failed after retries")
            continue

        result["meta"]["scene_id"] = scene["id"]
        result["meta"]["context"] = scene["context"]
        result["meta"]["n_turns"] = len(result["dialogue"])
        results.append(result)

        turns = len(result["dialogue"])
        print(f"  [OK] {turns} turns, emotion: {result['meta']['overall_emotion']}")

        time.sleep(REQUEST_INTERVAL)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_turns = sum(r["meta"]["n_turns"] for r in results)
    print(f"\n[DONE] Generated {len(results)} dialogues ({total_turns} total turns).")
    print(f"       Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
