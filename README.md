# Personality-Consistent Conversational AI Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> A stateful AI conversational agent with **long-term memory**, **emotion state management**, and **dynamic reasoning control** — designed to maintain personality consistency across 80+ turns of dialogue.

---

## ✨ Features

- 🧠 **RAG-based Knowledge Retrieval** — Character lore (background, relationships, personality traits) chunked into semantic pieces, embedded and stored in ChromaDB for Top-K retrieval.
- 🎭 **Three-Dimensional Emotion State** — Maintains Mood (emotional category), Energy (vitality level), and Closeness (relationship depth) with emotional inertia and long-term persistence.
- ⚡ **Dynamic Prompt Pipeline** — Assembles RAG results, few-shot examples, state info, and conversation history into a context-aware System Prompt at runtime.
- 🌡️ **State-Driven Inference Control** — Dynamically adjusts LLM 	emperature (0.5–0.95) based on agent state: lower randomness when low energy, higher diversity in intense emotional states.
- 🔄 **LLM-Augmented Data Pipeline** — Raw script corpus → style scoring → dialogue pair extraction → GPT-based data augmentation (5 input variations per sample), yielding ~800 training samples across 7 interaction categories.
- 📱 **WeChat Integration** — Deployed via wcferry / wxauto bridges for real-time conversational testing.
- 💾 **Long-Term Memory Persistence** — Conversation history and agent state survive restarts via JSONL append-only storage and state snapshots.

---

## 🏗️ Architecture

`
┌─────────────────────────────────────────────────────┐
│                    User Input                        │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│              RAG Engine (ChromaDB)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Embedding│→│ Top-K Ret│→│ Style/Metadata     │   │
│  │ (MiniLM) │  │  (K=3)   │  │ Filter (optional)│   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│           Dynamic Prompt Builder                     │
│  ┌───────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ RAG Chunks│ │ Few-Shots│ │ State + History    │  │
│  └─────┬─────┘ └────┬─────┘ └──────────┬─────────┘  │
│        └────────────┴──────────────────┘             │
│                      ▼                               │
│             System Prompt                            │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│         State-Driven Inference (DeepSeek API)        │
│  mood/energy → dynamic temperature (0.5 ~ 0.95)     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│         State Manager (Update + Persist)             │
│  mood · energy · closeness · turn_count             │
└─────────────────────────────────────────────────────┘
`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- A DeepSeek API key (get one at https://platform.deepseek.com)

### Installation

`ash
# Clone and install dependencies
git clone <your-repo-url>
cd Personality-Consistent-Conversational-AI-Agent
pip install -r requirements.txt
`

### Configuration

Create a .env file in the project root:

`env
DEEPSEEK_API_KEY=your-api-key-here
`

> ℹ️ The project uses os.getenv("DEEPSEEK_API_KEY") — no API key is hardcoded.

### Usage

#### CLI Chat

`ash
python run_agent.py
`

#### WeChat Bridge (Optional)

Requires a running WeChat desktop client on Windows.

`ash
# Using wcferry
python wechat_bridge.py

# Using wxauto
python wechat_wxauto_bridge.py
`

---

## 📂 Project Structure

`
├── run_agent.py                # CLI entry point
├── wechat_bridge.py            # WeChat integration (wcferry)
├── wechat_wxauto_bridge.py     # WeChat integration (wxauto)
├── requirements.txt            # Dependencies
├── .env.example                # Environment variable template
│
├── scripts/
│   ├── mark_agent.py           # Core agent: RAG + State + Prompt + LLM
│   ├── augment_dialogue.py     # LLM-based data augmentation (5x per sample)
│   ├── build_dialogue.py       # Dialogue pair construction with style scoring
│   ├── build_rag.py            # RAG index builder
│   ├── clean_data.py           # Raw script corpus cleaning
│   └── generate_multi_turn.py  # Multi-turn scenario expansion
│
├── data/
│   ├── rag/
│   │   ├── mark_rag.json       # Character lore (persona, relationships, etc.)
│   │   └── chroma_db/          # Vector store (gitignored)
│   ├── dialogue/
│   │   ├── mark_dialogue_v2.jsonl        # Base dialogue pairs
│   │   └── mark_dialogue_augmented.jsonl # LLM-augmented (~800 samples)
│   └── memory/
│       ├── conversations.jsonl # Persistent conversation history (gitignored)
│       └── agent_state.json    # Persistent agent state (gitignored)
│
└── prompts/
    └── mark_prompt.txt         # Prompt template
`

---

## 🔬 Design Decisions

### Why RAG over fine-tuning?
RAG allows real-time persona updates without retraining. Character lore (45 semantic chunks) is indexed in ChromaDB, enabling instant knowledge retrieval across new conversation topics.

### Emotion State Evolution (V1 → V3)
- **V1**: 6-dimensional emotion model — users couldn't perceive fine-grained distinctions
- **V2**: Reduced to **3 core dimensions** — Mood, Energy, Closeness — with emotional inertia preventing jarring mood swings
- **V3**: Added long-term state persistence so the agent maintains personality continuity across sessions

### Chunking Strategy
Character knowledge was split into ~45 chunks by category: personality traits (each as an independent chunk), background events (one per story beat), relationships (one per character), and lore items. This granularity prevents a single oversized persona from dominating the prompt context.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| openai | DeepSeek API client (OpenAI-compatible) |
| chromadb | Vector database for RAG retrieval |
| sentence-transformers | Embedding model (paraphrase-multilingual-MiniLM-L12-v2) |
| 
lpaug | Text augmentation utilities |
| wcferry (optional) | WeChat robot bridge |
| wxauto4 (optional) | Alternative WeChat automation |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) file for details.
