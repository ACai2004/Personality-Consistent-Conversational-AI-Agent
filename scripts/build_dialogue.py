import json
import os
import random
import re

BASE_DIR = "D:/VScodeProjects/Mark-Agent"
INPUT_FILE = os.path.join(BASE_DIR, "data", "cleaned", "mark_cleaned_v3.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "dialogue")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "mark_dialogue_v2.jsonl")

# ❌ 低质量 blacklist
BLACKLIST = {
    "yeah", "uh", "huh", "okay", "ok", "yes", "no"
}

# ✅ 去重
seen_outputs = set()

# ------------------------
# ✅ 分类（升级版）
# ------------------------
def classify(line):
    l = line.lower()

    if any(k in l for k in ["dad", "mom", "family"]):
        return "family"
    if any(k in l for k in ["sorry", "apologize"]):
        return "apology"
    if any(k in l for k in ["love", "miss", "care"]):
        return "emotional"
    if any(k in l for k in ["hit", "fight", "kill", "stop", "hurt"]):
        return "combat"
    if any(k in l for k in ["school", "class", "finals"]):
        return "school"
    if any(k in l for k in ["what if", "i can't", "not ready"]):
        return "self_doubt"

    return "casual"


# ------------------------
# ✅ 情绪识别
# ------------------------
def detect_emotion(line):
    l = line.lower()

    if "sorry" in l:
        return "guilt"
    if any(k in l for k in ["love", "care"]):
        return "warm"
    if any(k in l for k in ["can't", "what if", "not ready"]):
        return "anxious"
    if any(k in l for k in ["fight", "stop", "hurt"]):
        return "intense"
    if any(k in l for k in ["fuck", "shit"]):
        return "angry"

    return "neutral"


# ------------------------
# ✅ 生成“用户输入”（核心升级）
# ------------------------
def generate_user_input(line, category):
    l = line.lower()

    # 🎯 强规则（优先）
    if "sorry" in l:
        return random.choice([
            "Why are you late?",
            "What happened?",
            "You okay?"
        ])

    if "dad" in l:
        return random.choice([
            "Did you talk to your dad?",
            "What happened with your dad?",
            "Are things okay at home?"
        ])

    if category == "combat":
        return random.choice([
            "What happened out there?",
            "Are you hurt?",
            "Did you win?"
        ])

    if category == "self_doubt":
        return random.choice([
            "What's wrong?",
            "Why are you hesitating?",
            "You okay?"
        ])

    if category == "emotional":
        return random.choice([
            "What do you mean?",
            "Are you serious?",
            "Why are you saying that?"
        ])

    if category == "school":
        return random.choice([
            "How's school going?",
            "Are you ready for finals?",
        ])

    # 🎯 fallback（非常重要）
    return random.choice([
        "What's going on?",
        "What are you doing?",
        "You good?"
    ])


# ------------------------
# ✅ 过滤低质量
# ------------------------
def is_valid(line):
    words = line.split()

    if len(words) < 4:
        return False

    if line.lower() in BLACKLIST:
        return False

    # 去掉全是疑问词/废话
    if re.fullmatch(r"[?!. ]+", line):
        return False

    return True


# ------------------------
# 🚀 主流程
# ------------------------
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    dataset = []

    for line in lines:

        # ❌ 过滤
        if not is_valid(line):
            continue

        # ❌ 去重
        if line in seen_outputs:
            continue
        seen_outputs.add(line)

        # ✅ 分类
        category = classify(line)

        # ✅ 情绪
        emotion = detect_emotion(line)

        # ✅ 生成用户输入
        user_input = generate_user_input(line, category)

        data = {
            "input": user_input,
            "output": line,
            "meta": {
                "type": category,
                "emotion": emotion
            }
        }

        dataset.append(data)

    # 写入
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(dataset)} high-quality dialogue samples.")


if __name__ == "__main__":
    main()