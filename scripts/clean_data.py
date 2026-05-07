import re

INPUT_PATH = "../data/raw/mark_s1.txt"
OUTPUT_PATH = "../data/cleaned/mark_cleaned_v3.txt"

BLACKLIST = {
    "yeah", "huh", "okay", "ok", "uh", "um",
    "yup", "nope", "hmm", "ah", "oh"
}

# =========================
# 🔥 新增：Mark风格正向评分
# =========================
def mark_style_score(line: str) -> int:
    lower = line.lower()
    score = 0

    # 第一人称（核心特征）
    if re.search(r"\b(i|i'm|i've|i'll|i'd)\b", lower):
        score += 2

    # 家庭相关（强特征）
    if "dad" in lower or "mom" in lower:
        score += 2

    # 情绪 / 犹豫
    if any(w in lower for w in ["i think", "i guess", "i don't know", "sorry", "scared"]):
        score += 2

    # 青少年语气
    if any(w in lower for w in ["dude", "oh man", "crap", "shit"]):
        score += 1

    # 疑问句（不确定性）
    if "?" in line:
        score += 1

    # 长一点更可能是有效语义
    if len(line.split()) > 6:
        score += 1

    return score


# =========================
# 🔥 新增：反向过滤（不像Mark）
# =========================
def is_not_mark_style(line: str) -> bool:
    lower = line.lower()

    # ❌ 过于成熟/领导者语气（像他爸）
    if "we must" in lower or "we will" in lower:
        return True

    # ❌ 战术/组织性语言
    if "we need to" in lower and "i" not in lower:
        return True

    # ❌ 纯命令句（大概率不是Mark）
    if re.match(r"^[A-Z][a-z]+ (him|her|them|this|that)", line):
        return True

    # ❌ 太“官方”的表达
    if "this is unacceptable" in lower or "stay together" in lower:
        return True

    return False


def clean_line(line: str) -> str | None:
    line = line.strip()

    if not line:
        return None

    # 去括号内容
    line = re.sub(r"\(.*?\)", "", line)

    # 标准化空格
    line = re.sub(r"\s+", " ", line)

    # 标点-only
    if re.fullmatch(r"[^\w]+", line):
        return None

    lower = line.lower()

    # 黑名单
    if lower in BLACKLIST:
        return None

    # 词数过滤
    if len(lower.split()) < 3:
        return None

    # ❌ 超长（多角色拼接）
    if len(line) > 150:
        return None

    # ❌ 多句拼接
    if line.count('.') > 2:
        return None

    # ❌ 反向过滤（关键）
    if is_not_mark_style(line):
        return None

    # ✅ 风格评分（核心）
    score = mark_style_score(line)

    if score < 3:
        return None

    return line


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = []
    seen = set()

    for line in lines:
        cl = clean_line(line)
        if cl and cl not in seen:
            cleaned.append(cl)
            seen.add(cl)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")

    print(f"✅ Done. Cleaned {len(cleaned)} lines.")


if __name__ == "__main__":
    main()