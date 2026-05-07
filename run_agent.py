"""
Mark Grayson Agent — CLI入口
==============================
用法: conda activate mark-agent && python run_agent.py
输入中文，Mark用英文回复。输入 exit 或 quit 退出。
"""

from scripts.mark_agent import MarkAgent


def main():
    agent = MarkAgent()

    print("\n" + "=" * 60)
    print("  Mark Grayson (Invincible) — RAG Agent")
    print("  你说中文，他回英文。输入 exit 退出。")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\n💬 你说 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Bye! Take care.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("\n👋 Mark: Later, babe. Don't miss me too much.")
            break

        reply = agent.chat(user_input)
        print(f"\n🦸 Mark  > {reply}")


if __name__ == "__main__":
    main()
