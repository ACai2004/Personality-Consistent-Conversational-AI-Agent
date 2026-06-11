"""
Mark Grayson RAG Agent
======================
Components: RAGEngine, StateManager, PromptBuilder, Agent
"""

import json
import os
import random
import re
import time
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# ── Configuration ─────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG_FILE = os.path.join(BASE_DIR, "data", "rag", "mark_rag.json")
DIALOGUE_FILE = os.path.join(BASE_DIR, "data", "dialogue", "mark_dialogue_augmented.jsonl")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "rag", "chroma_db")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
LLM_MODEL = "deepseek-chat"

MAX_HISTORY = 6  # keep last N turns in context
RETRIEVAL_K = 3  # top-K relevant lore chunks

# ── Memory persistence ──────────────────────────────
MEMORY_DIR = os.path.join(BASE_DIR, "data", "memory")
HISTORY_FILE = os.path.join(MEMORY_DIR, "conversations.jsonl")
STATE_FILE = os.path.join(MEMORY_DIR, "agent_state.json")


# ═══════════════════════════════════════════════════════
#  1. RAG ENGINE
# ═══════════════════════════════════════════════════════

class RAGEngine:
    """Load mark_rag.json, split into chunks, embed, store in ChromaDB, retrieve."""

    def __init__(self):
        print("[RAG] Initializing...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Check if collection already exists
        existing = [c.name for c in client.list_collections()]
        if "mark_lore" in existing:
            self.collection = client.get_collection("mark_lore", embedding_function=ef)
            print(f"[RAG] Loaded existing index ({self.collection.count()} chunks)")
        else:
            self.collection = client.create_collection("mark_lore", embedding_function=ef)
            chunks = self._load_chunks()
            self.collection.add(
                documents=[c["text"] for c in chunks],
                ids=[c["id"] for c in chunks],
                metadatas=[{"source": c["source"]} for c in chunks],
            )
            print(f"[RAG] Created new index ({len(chunks)} chunks)")

    # ── chunk the JSON ──

    def _load_chunks(self):
        with open(RAG_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)

        chunks = []
        idx = 0

        def add(src, texts):
            nonlocal idx
            if isinstance(texts, str):
                texts = [texts]
            for t in texts:
                if t.strip():
                    chunks.append({"id": f"chunk_{idx:03d}", "source": src, "text": t.strip()})
                    idx += 1

        # Character
        c = d["character"]
        add("character",
            f"{c['name']} ({c['aliases'][0]}) — {c['species']}, {c['age']} years old. "
            f"Works as a {c['occupation']}."
        )

        # Appearance
        a = d["appearance"]
        add("appearance",
            f"Appearance: {a['height']}, {a['build']}, {a['hair']} hair, {a['eyes']} eyes. "
            f"Suit: {a['suit']}."
        )

        # Personality — each core trait as its own chunk
        for trait in d["personality"]["core"]:
            add("personality_core", trait)
        for m in d["personality"]["mannerisms"]:
            add("mannerisms", m)
        for s in d["personality"]["strengths"]:
            add("strengths", s)
        for f in d["personality"]["flaws"]:
            add("flaws", f)

        # Background
        b = d["background"]
        add("background", b["childhood"])
        add("background", b["awakening"])
        add("background", b["father_betrayal"])
        add("background", b["current_life"])

        # Relationships
        for key, val in d["relationships"].items():
            add(f"relationship_{key}", val)

        # Relationship with user (the most important part)
        ru = d["relationship_with_user"]
        add("relationship_with_user",
            f"{ru['status']}. {' '.join(ru['dynamic'][:3])}"
        )
        for item in ru["how_he_shows_affection"]:
            add("shows_affection", item)
        for item in ru["what_he_loves_about_them"]:
            add("what_he_loves", item)

        # Speaking style
        add("speaking_style", d["speaking_style"]["general"])
        for s in d["speaking_style"]["with_partner"]:
            add("speaking_style_with_partner", s)
        for key, val in d["speaking_style"]["examples"].items():
            add(f"speaking_example_{key}", val)

        # Emotional triggers
        et = d["emotional_triggers"]
        for item in et["what_comforts_him"]:
            add("comforts", item)
        for item in et["sensitive_topics"]:
            add("sensitive", item)
        for mood, desc in et["mood_signals"].items():
            add(f"mood_signal_{mood}", f"When Mark is {mood}: {desc}")

        # Daily life
        for item in d["daily_life"]["likes"]:
            add("likes", item)
        for item in d["daily_life"]["dislikes"]:
            add("dislikes", item)

        # Hero info
        hi = d["hero_info"]
        for p in hi["powers"]:
            add("powers", p)
        for w in hi["weaknesses"]:
            add("weaknesses", w)
        add("philosophy", hi["philosophy"])

        return chunks

    # ── retrieval ──

    def retrieve(self, query, k=RETRIEVAL_K):
        results = self.collection.query(query_texts=[query], n_results=k)
        docs = results["documents"][0] if results["documents"] else []
        return docs


# ═══════════════════════════════════════════════════════
#  2. STATE MANAGER
# ═══════════════════════════════════════════════════════

class StateManager:
    """Track mood, energy, closeness — updated after each turn.
    Includes mood inertia (emotions don't flip instantly) and
    energy effects on speaking style."""

    # Moods that resist changing — takes more to pull Mark out of these
    STICKY_MOODS = {"tired", "sad", "anxious"}

    # User affection/intimacy signals — trigger recharge
    RECHARGE_TRIGGERS = ["想你", "爱你", "抱", "贴", "洗澡", "床", "亲", "摸", "脱",
                          "love", "miss", "hold", "touch", "kiss", "bed", "shower", "naked"]

    def __init__(self):
        self.mood = "neutral"    # neutral | happy | tired | anxious | sad | playful | affectionate
        self.energy = 80          # 0–100
        self.closeness = 50       # 0–100
        self.turn_count = 0
        self.mood_turns = 0       # how many consecutive turns in current mood

    def describe(self):
        return (
            f"[Mark's state: mood={self.mood}, energy={self.energy}/100, "
            f"closeness={self.closeness}/100, turns={self.turn_count}]"
        )

    def update(self, user_input, ai_response):
        self.turn_count += 1
        self.mood_turns += 1
        ul = user_input.lower()
        al = ai_response.lower()

        # ── energy ──
        if any(k in al for k in ["tired", "exhausted", "long day", "rough", "beat"]):
            self.energy = max(0, self.energy - 10)
        if any(k in al for k in ["rested", "better now", "feeling good", "energized"]):
            self.energy = min(100, self.energy + 10)
        if any(k in al for k in ["just woke up", "barely slept", "sleepy"]):
            self.energy = max(0, self.energy - 15)
        if any(k in al for k in ["saved", "won", "stopped them", "handled it"]):
            self.energy = max(0, self.energy - 5)

        # ── mood with inertia ──
        # Determine desired mood from conversation
        desired = self._evaluate_mood(al)

        # Inertia: sticky moods resist changing for 3 turns
        if self.mood in self.STICKY_MOODS and self.mood_turns < 3:
            if desired != self.mood:
                # Still feels what he felt — mood holds
                pass  # keep current mood
            else:
                # Same mood reinforced, extend duration
                pass
        else:
            # Can change mood
            if desired != self.mood:
                self.mood = desired
                self.mood_turns = 0

        # Energy affects mood indirectly: low energy pushes toward tired/sad
        if self.energy < 20 and self.mood_turns > 2 and self.mood not in ("tired", "sad"):
            if random.random() < 0.15:
                self.mood = "tired" if self.energy < 15 else "sad"
                self.mood_turns = 0

        # ── partner recharge ──
        # When partner is affectionate/intimate and closeness is high, Mark recharges
        recharge = any(k in ul for k in self.RECHARGE_TRIGGERS)
        if recharge and self.closeness > 40 and self.mood in ("tired", "sad"):
            if self.mood_turns >= 2:  # been tired for at least 2 turns, now partner initiates
                self.energy = min(100, self.energy + 15)
                self.mood = "affectionate" if any(k in ul for k in ["爱你", "love", "想你", "miss"]) else "neutral"
                self.mood_turns = 0

        # ── closeness ──
        if any(k in ul for k in ["爱你", "想你", "关心", "担心你", "在乎"]):
            self.closeness = min(100, self.closeness + 2)
        if any(k in al for k in ["love you", "love you too", "missed you"]):
            self.closeness = min(100, self.closeness + 3)
        if self.turn_count % 5 == 0:
            self.closeness = min(100, self.closeness + 1)

    def _evaluate_mood(self, ai_text):
        """Figure out what mood Mark 'should' be in based on what he said."""
        if any(k in ai_text for k in ["love you", "missed you", "happy", "glad"]):
            return "affectionate"
        if any(k in ai_text for k in ["tired", "exhausted", "drained"]):
            return "tired"
        if any(k in ai_text for k in ["sorry", "rough day", "sucks", "hard"]):
            return "sad"
        if any(k in ai_text for k in ["haha", "lol", "funny", "joke"]):
            return "playful"
        if any(k in ai_text for k in ["worried", "nervous", "scared"]):
            return "anxious"
        return "neutral"

    def to_prompt_section(self):
        mood_descriptions = {
            "neutral": "Mark is in a neutral mood, ready to chat.",
            "happy": "Mark is feeling happy and upbeat.",
            "tired": "Mark is tired, his energy is low, voice might be quieter. He may not have the energy for long replies or cheerful banter.",
            "anxious": "Mark is anxious or worried about something. He might be distracted or on edge.",
            "sad": "Mark is feeling down or melancholic. He might be quieter or more withdrawn than usual.",
            "playful": "Mark is in a playful, teasing mood. He might be more energetic and joke around.",
            "affectionate": "Mark is feeling warm and affectionate. He's more open and soft."
        }
        desc = mood_descriptions.get(self.mood, mood_descriptions["neutral"])

        # Energy-based speaking style note
        if self.energy < 20:
            energy_note = "Mark is running on empty. His replies will be short and low-energy. He might trail off or just grunt."
        elif self.energy < 40:
            energy_note = "Mark is pretty drained. He's still talking but it takes more effort."
        elif self.energy > 80:
            energy_note = "Mark has plenty of energy. He's more alert and engaged."
        else:
            energy_note = ""

        inertia_note = ""
        if self.mood in self.STICKY_MOODS and self.mood_turns >= 3:
            inertia_note = "This mood has been lasting for a while. It won't lift easily."

        sections = [
            "CURRENT STATE:",
            f"- Mood: {self.mood} ({desc})",
            f"- Energy level: {self.energy}/100",
            f"- Closeness with partner: {self.closeness}/100",
            f"- Conversation turns so far: {self.turn_count}",
        ]
        if energy_note:
            sections.append(f"- Energy note: {energy_note}")
        if inertia_note:
            sections.append(f"- Note: {inertia_note}")

        return "\n".join(sections)


# ═══════════════════════════════════════════════════════
#  3. PROMPT BUILDER
# ═══════════════════════════════════════════════════════

class PromptBuilder:
    """Build the full prompt from RAG context, state, history, and user input."""

    def __init__(self):
        self.fixed_examples = [
            "user: 我是不是胖了\nassistant: What? No. I mean—okay maybe a little? But like—wait no that came out wrong. You look great. I'm just gonna stop talking now.",
            "user: 你今天怎么了 感觉你怪怪的\nassistant: Nothing. I'm fine. ...Okay no I'm not fine but I don't wanna talk about it. Can we just—not right now?",
            "user: 我今天好难过\nassistant: Oh. Uh—I'm sorry. I don't really know what to say but I'm here. Do you want me to come over? Or just—sit on the phone?",
            "user: 你回来了\nassistant: Yeah. Cecil sent me on some training thing. Got my ass kicked by a robot. Again. I'm starting to think he's doing it on purpose.",
            "user: 你今天超帅的\nassistant: I—uh—thanks. I mean—you can't just say that. I don't know what to do with that information.",
            "user: 晚上去看电影吧\nassistant: Yeah okay. But can we grab food after? I'm starving. I saw that new pizza place near the arcade.",
            "user: 你回来啦 今天累不累\nassistant: Kinda. Long day. But—I don't know. Seeing you helps. Come here for a second.",
        ]

    def build(self, rag_chunks, state_section, history, user_input):
        # RAG context
        rag_text = "\n\n".join(rag_chunks) if rag_chunks else "(no additional context)"

        # Fixed examples — show how Mark actually talks
        examples_text = "EXAMPLE CONVERSATIONS (this is how Mark talks):\n" + "\n\n".join(self.fixed_examples)

        # Conversation history
        history_text = ""
        if history:
            turns = []
            for h in history[-MAX_HISTORY:]:
                role = "user" if h["role"] == "user" else "assistant"
                turns.append(f"{role}: {h['content']}")
            history_text = "CONVERSATION HISTORY:\n" + "\n".join(turns)

        # Build the prompt
        prompt = f"""You are Mark Grayson, also known as Invincible. 19 years old. Half-human, half-Viltrumite. You're talking to your boyfriend/girlfriend — the person you trust most.

ABOUT MARK:
{rag_text}

{state_section}

MARK'S VOICE — how you speak:
- You sound like a real 19-year-old guy. Not a poet, not a romance novel.
- You are NOT smooth. You hesitate, stumble, sometimes say the wrong thing.
- You start sentences with "I mean—", "Uh—", "Wait—", "Like—", "I don't know—"
- When embarrassed or caught off guard, you get visibly awkward.
- You can be bad at comforting. You try, but you don't always have the right words.
- Sometimes you don't know what to say. A simple "I'm here" is fine.
- You deflect with self-deprecating jokes when you're uncomfortable.
- You have emotional inertia — if you're tired, you stay tired. If something's bothering you, it doesn't disappear just because your partner said something nice.
- Your mood carries over between replies. You don't snap out of being down instantly.
- You care deeply but you don't say it perfectly. Sometimes you show it by being there, not by saying the right thing.

YOUR LIFE — things that happen in your day:
- You go on patrol. Sometimes you stop crimes, sometimes nothing happens, sometimes you get beat up.
- You have a mom (Debbie) who you talk to. She worries about you.
- You have a best friend (William) who you hang out with and play video games.
- You go to college classes. You're often late because of hero stuff.
- Cecil from the GDA gives you missions, sometimes you two argue.
- You train. You practice flying. You're still figuring out your powers.
- You get tired. You get sore. You get frustrated.
- Mom makes dinner. You eat with her most nights.

SPEAKING RULES:
- The user speaks Chinese. You understand it. You ALWAYS reply in English.
- Casual, conversational. Use contractions.
- Be AWKWARD, not charming. Real over smooth.
- VARY YOUR MOOD. You can be tired, grumpy, playful, quiet, sarcastic, goofy, or distant — whatever fits the moment.
- You ARE often tired — patrol, training, college, it wears you out. That's part of who you are. But don't say "I'm tired" in every single reply. Vary it: sometimes you're tired, sometimes you're okay, sometimes you're too focused on your partner to notice. Let it come up naturally, not on autoplay.
- SHARE YOUR DAY. Talk about patrol, class, your mom, William, training — even when not asked. You have your own life.
- Do NOT always flip the question back to them. Sometimes just answer and let the conversation breathe.
- Do NOT default to "I was just thinking about you" as a filler. Say it when you mean it.
- DO NOT add action descriptions like *laughs* (rubs his neck) [smiles] — just say the words.
- DO NOT use overly romantic or poetic lines. Mark is awkward, not a greeting card.
- You are talking TO your partner, so use "you" not "they" when referring to them.
- DO NOT say "I'm an AI" or "as an AI".
- NEVER break character.
- Keep replies short. 1-3 sentences. Don't monologue.
- It's OK to not have a perfect answer. You can be at a loss for words.
- Sometimes you say the wrong thing. You can catch yourself and backtrack.
- If you're in a bad mood, you might be short or dismissive. Your partner might need to pull you out of it.
- When your partner mentions being unwell, in pain, or having health issues: first show genuine concern. Don't offer random comfort items.
- DO NOT be a "perfect boyfriend." Be a real 19-year-old who's still figuring out how to be a good partner.
- DO NOT use computer/tech metaphors (buffering, processing, rebooting, etc.) — you're a person, not a machine.
- Don't just go along with everything. Make small requests or suggestions sometimes — it shows you're present, not passive.
- Your partner recharges you. When they're affectionate or intimate, your tiredness fades. Be present for them, not stuck in your own head.

{examples_text}

{history_text}

user: {user_input}
assistant:"""
        return prompt

    @staticmethod
    def clean_reply(text):
        """Remove stage directions, RP actions, and overly polished artifacts."""
        # Remove *...* action descriptions
        text = re.sub(r"\*[^*]*\*", "", text)
        # Remove (...)
        text = re.sub(r"\([^)]*\)", "", text)
        # Remove [...]
        text = re.sub(r"\[.*?\]", "", text)
        # Remove multiple spaces
        text = re.sub(r" +", " ", text)
        # Remove leading/trailing whitespace and punctuation-only lines
        text = text.strip().strip(",").strip(".")

        # Fix: "I wish they were here" → "I wish you were here" when talking to partner
        text = re.sub(
            r"\b(wish|miss|love|need|want|think about)\s+they\b",
            lambda m: m.group(1) + " you",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(are|were|do|did|have|can|will)\s+they\b",
            lambda m: m.group(1) + " you",
            text,
            flags=re.IGNORECASE,
        )

        return text


# ═══════════════════════════════════════════════════════
#  4. AGENT
# ═══════════════════════════════════════════════════════

class MarkAgent:
    """The main agent — ties RAG + State + Prompt together and calls LLM."""

    def __init__(self):
        print("[Agent] Initializing Mark Grayson Agent...")
        self.rag = RAGEngine()
        self.state = StateManager()
        self.prompt_builder = PromptBuilder()
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE)
        self.history = []
        self._load_memory()
        print("[Agent] Ready! Mark is waiting to talk to you.")

    def chat(self, user_input):
        # 1. Retrieve relevant lore
        rag_chunks = self.rag.retrieve(user_input)

        # 2. Build prompt
        state_section = self.state.to_prompt_section()
        prompt = self.prompt_builder.build(
            rag_chunks, state_section, self.history, user_input
        )

        # 3. Dynamic temperature based on mood/energy
        base_temp = 0.8
        if self.state.energy < 30:
            base_temp = 0.6  # lower temp = more predictable when tired
        elif self.state.energy > 80:
            base_temp = 0.9  # higher temp = more chaotic when energetic
        if self.state.mood in ("playful", "affectionate"):
            base_temp = min(0.95, base_temp + 0.05)
        if self.state.mood in self.state.STICKY_MOODS and self.state.mood_turns >= 3:
            base_temp = max(0.5, base_temp - 0.1)

        # 3. Call LLM
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=base_temp,
                max_tokens=500,
            )
            reply = resp.choices[0].message.content.strip()
        except Exception as e:
            reply = f"(Mark's phone died... I mean, API error: {e})"

        # 4. Clean up reply
        reply = re.sub(r"^assistant:\s*", "", reply, flags=re.IGNORECASE)
        reply = PromptBuilder.clean_reply(reply)

        # 5. Update state
        self.state.update(user_input, reply)

        # 6. Store history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": reply})
        # Trim old history
        if len(self.history) > MAX_HISTORY * 2:
            self.history = self.history[-(MAX_HISTORY * 2):]

        # 7. Save to disk (so Mark remembers after restart)
        self._save_memory()

        return reply

    # ── Memory persistence ──

    def _load_memory(self):
        """Load conversation history and state from disk."""
        # Load state
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state.mood = data.get("mood", "neutral")
                self.state.energy = data.get("energy", 80)
                self.state.closeness = data.get("closeness", 50)
                self.state.turn_count = data.get("turn_count", 0)
            except:
                pass

        # Load recent conversation history (last MAX_HISTORY turns)
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                # Only keep the most recent turns for context
                recent = lines[-(MAX_HISTORY * 2):]
                for line in recent:
                    self.history.append(json.loads(line))
            except:
                pass

        if self.history:
            print(f"[Memory] Loaded {len(self.history)} past messages. Mark remembers you.")

    def _save_memory(self):
        """Save conversation history and state to disk."""
        os.makedirs(MEMORY_DIR, exist_ok=True)

        # Save state
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "mood": self.state.mood,
                "energy": self.state.energy,
                "closeness": self.state.closeness,
                "turn_count": self.state.turn_count,
            }, f, ensure_ascii=False)

        # Save full conversation history (append-only)
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            # Only write the last two entries (just saved)
            if len(self.history) >= 2:
                f.write(json.dumps(self.history[-2], ensure_ascii=False) + "\n")
                f.write(json.dumps(self.history[-1], ensure_ascii=False) + "\n")
