"""
WeChat Bridge for Mark Grayson Agent
======================================
把 Mark 接入微信，在备用手机上和他聊天。

用法：
  1. 装好指定版本微信并登录备用号
  2. 把这个备用号加为好友
  3. 运行本脚本
  4. 在备用机上给这个号发消息，Mark 自动回复

注意：需要 pywxdll 和 WeChat 版本匹配。
"""

import json
import os
import sys
import time
import threading
from pathlib import Path

# 把项目根目录加入路径
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from scripts.mark_agent import MarkAgent
from wcferry import client as wcf_client


def load_config():
    """加载配置。如果没有，引导用户配置。"""
    config_file = BASE_DIR / "data" / "wechat_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("=" * 50)
    print("首次运行，需要配置你的微信信息")
    print("=" * 50)
    print("\n请先完成以下步骤：")
    print("1. 在电脑上安装微信 3.9.12.51（或其他兼容版本）")
    print("2. 登录你的备用微信号")
    print("3. 把这台电脑上的微信号加为好友")
    print("\n完成后，输入你的完整 wxid（在微信「我」→ 设置 → 账号与安全 → 微信号 查看）：")
    print("注意：不是微信号，是 wxid_xxxxxxxx 格式的 ID\n")

    my_wxid = input("你的 wxid > ").strip()

    config = {
        "my_wxid": my_wxid,
        "reply_delay": 1.0,  # 模拟打字延迟（秒）
        "auto_accept_friend": True,
    }

    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n配置已保存到 {config_file}")
    return config


def main():
    config = load_config()
    my_wxid = config["my_wxid"]

    print("\n[Bridge] 正在连接微信...")
    wcf = wcf_client.Wcf(debug=False)

    # 检查登录状态
    if not wcf.is_login():
        print("[Bridge] 微信未登录，请在电脑上扫码登录")
        # 尝试获取二维码
        qr = wcf.get_qrcode()
        if qr:
            print(f"[Bridge] 二维码已生成：{qr}")
        # 等待登录
        while not wcf.is_login():
            time.sleep(2)
        print("[Bridge] 微信登录成功！")

    self_info = wcf.get_user_info()
    print(f"[Bridge] 登录账号：{self_info.get('name', '未知')} ({self_info.get('wxid', '未知')})")

    # 初始化 Mark Agent
    print("[Bridge] 正在唤醒 Mark...")
    agent = MarkAgent()

    # 获取好友列表确认目标存在
    friends = wcf.get_friends()
    target_found = any(f.get("wxid") == my_wxid for f in friends)
    if target_found:
        print(f"[Bridge] 已找到目标好友 ✓")
    else:
        print(f"[Bridge] 警告：好友列表中未找到 {my_wxid}")
        print("        请确认已添加该账号为好友")

    # 开启消息接收
    wcf.enable_receiving_msg()
    print(f"[Bridge] Mark 已上线！发消息给备用号即可聊天")

    # 消息处理循环
    while True:
        try:
            msg = wcf.get_msg(block=True)

            # 只处理文本消息
            if msg.type != 1:
                continue

            # 只处理来自目标用户的消息（且不是群聊）
            if msg.sender != my_wxid or msg.roomid:
                continue

            content = msg.content.strip()
            if not content:
                continue

            # 显示收到的消息
            print(f"\n💬 [{time.strftime('%H:%M:%S')}] 你说: {content}")

            # 模拟打字延迟
            delay = min(config.get("reply_delay", 1.0) + len(content) * 0.02, 3.0)
            time.sleep(delay)

            # 调用 Mark Agent
            try:
                reply = agent.chat(content)
            except Exception as e:
                reply = f"(Mark 的大脑短路了一下… {e})"

            # 发送回复
            wcf.send_text(reply, my_wxid)
            print(f"🦸 Mark: {reply}")

        except KeyboardInterrupt:
            print("\n[Bridge] 收到退出信号，Mark 去睡觉了...")
            break
        except Exception as e:
            print(f"[Bridge] 错误: {e}")
            time.sleep(2)

    wcf.cleanup()


if __name__ == "__main__":
    main()
