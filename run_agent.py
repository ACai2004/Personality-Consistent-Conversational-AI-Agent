"""
Mark Grayson Agent — CLI入口
==============================
用法: conda activate ai-agent && python run_agent.py
"""

from scripts.mark_agent import MarkAgent


def main():
    agent = MarkAgent()

    print("\n" + "=" * 60)
    print("  Personality-Consistent AI Agent")
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
            print("\n👋 Mark: See you next time!")
            break

        reply = agent.chat(user_input)
        print(f"\n🦸 Agent > {reply}")


if __name__ == "__main__":
    main()

