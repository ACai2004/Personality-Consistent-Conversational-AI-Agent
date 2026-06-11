"""
WeChat (wxauto4) Bridge for Mark Grayson Agent
================================================
基于 wxauto4 的微信接入方案，适用于微信 4.1.x。
使用 UI 自动化（非 DLL 注入），通过轮询获取新消息。

用法：
  1. 登录微信电脑版
  2. 运行本脚本，首次会提示输入联系人昵称
  3. 给该联系人发消息，Mark 自动回复
  4. Ctrl+C 退出
"""

import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from wxauto4 import WeChat
except ImportError:
    print("请先安装 wxauto4: pip install wxauto4")
    sys.exit(1)

from scripts.mark_agent import MarkAgent


def load_config():
    """加载配置，首次运行引导用户输入联系人昵称"""
    config_file = BASE_DIR / "data" / "wechat_wxauto_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("=" * 50)
    print("首次运行配置")
    print("=" * 50)
    print("\n联系人昵称：你的微信号在 Mark 好友列表里显示的名字")
    print("（就是 Mark 微信里点开和你的聊天框，顶部显示的名字）\n")
    target = input("联系人昵称 > ").strip()

    if not target:
        print("昵称不能为空！")
        sys.exit(1)

    config = {"target_contact": target}

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"配置已保存到 {config_file}")
    return config


def main():
    print("=" * 50)
    print("  Mark Grayson — WeChat Bridge")
    print("  基于 wxauto4  |  说中文 Mark 回英文")
    print("=" * 50)

    config = load_config()
    target = config["target_contact"]

    print(f"\n[Bridge] 目标联系人: {target}")
    print("[Bridge] 正在连接微信...")

    try:
        wx = WeChat()
    except Exception as e:
        print(f"[Bridge] 连接微信失败: {e}")
        print("请确保微信已登录且窗口已打开")
        sys.exit(1)

    # 获取当前登录账号信息
    try:
        me = wx.GetMyInfo()
        nickname = me.get("display_name", me.get("name", "未知"))
        print(f"[Bridge] 当前微信账号: {nickname}")
    except Exception:
        print("[Bridge] 无法获取账号信息（不影响运行）")

    print("[Bridge] 正在唤醒 Mark...")
    agent = MarkAgent()

    # 切换到目标联系人聊天窗口
    print(f"[Bridge] 切换到 [{target}] 的聊天窗口...")
    try:
        wx.ChatWith(target)
    except Exception as e:
        print(f"[Bridge] 切换到 [{target}] 失败: {e}")
        print("可能的原因：找不到该联系人，请检查昵称是否正确")
        sys.exit(1)

    print(f"\n[Bridge] ✅ Mark 已上线！正在监听 [{target}] 的消息...")
    print("[Bridge] ⚠️ 请保持微信窗口打开（可最小化）")
    print("[Bridge] 按 Ctrl+C 退出\n")

    # ── 轮询监听 ──
    check_interval = 1.5  # 每 1.5 秒检查一次
    last_text = ""  # 记录最后一条消息内容，避免重复处理

    try:
        while True:
            try:
                messages = wx.GetAllMessage()
                if messages:
                    latest = messages[-1]
                    content = latest.content.strip() if latest.content else ""
                    sender = getattr(latest, "sender", "")

                    # 有新的文本消息 + 不是自己发的(self) + 不是系统消息 + 不是重复的
                    if content and sender not in ("self", "system") and content != last_text:
                        last_text = content
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"\n💬 [{timestamp}] {sender}: {content}")

                        try:
                            reply = agent.chat(content)
                        except Exception as e:
                            reply = f"(Mark 大脑短路了一下… {e})"
                            print(f"[Bridge] 错误: {e}")

                        wx.SendMsg(reply, target)
                        print(f"🦸 Mark: {reply}")

            except Exception:
                # 偶尔的错误（如切换窗口时）不影响运行
                pass

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n[Bridge] 收到退出信号，Mark 去睡觉了...")
    except Exception as e:
        print(f"\n[Bridge] 运行出错: {e}")
    finally:
        print("[Bridge] 已安全退出")


if __name__ == "__main__":
    main()
