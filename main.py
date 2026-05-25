"""CLI 入口：运行猫行为咨询对话。

用法示例：
    python main.py                          # 运行第一个 benchmark case
    python main.py --case-id cat_single_case_01
    python main.py --case-id cat_single_case_02
    python main.py --list-cases             # 列出所有可用 case
    python main.py --no-routing             # 关闭路由（消融实验）
    python main.py --free-chat              # 自由对话模式（输入 q/退出 结束）
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator

import Config
from agents import memory_store
from agents.behaviorist_agent import BehavioristAgent
from agents.user_agent import UserAgent
from utils.benchmark_loader import BenchmarkCase, load_benchmark, select_case
from utils.llm_client import LLMClient, LLMClientError


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _stream_turn(role: str, stream: Iterator[str], is_think: bool = False) -> str:
    """流式打印一轮对话，返回完整文本。

    思考模型的 <think>...</think> 内容会以暗色展示。
    """
    print(f"\n{role}")
    print("-" * 50)

    full_text: list[str] = []
    buffer = ""
    in_think = False

    for chunk in stream:
        buffer += chunk
        full_text.append(chunk)

        # 检测思考块开始
        if is_think and not in_think and "<think>" in buffer:
            in_think = True
            before, _, after = buffer.partition("<think>")
            if before:
                print(before, end="", flush=True)
            print("\n\033[2m[思考过程]\033[0m", flush=True)
            buffer = after
            continue

        # 检测思考块结束
        if is_think and in_think and "</think>" in buffer:
            in_think = False
            think_content, _, after = buffer.partition("</think>")
            print(f"\033[2m{think_content}\033[0m", end="", flush=True)
            print("\n\033[2m[思考结束]\033[0m\n", flush=True)
            buffer = after
            continue

        if buffer:
            if in_think and is_think:
                print(f"\033[2m{buffer}\033[0m", end="", flush=True)
            else:
                print(buffer, end="", flush=True)
            buffer = ""

    if buffer:
        if in_think and is_think:
            print(f"\033[2m{buffer}\033[0m", end="", flush=True)
        else:
            print(buffer, end="", flush=True)

    print()
    return "".join(full_text)


def _print_static_turn(role: str, content: str) -> None:
    """打印非流式的一轮（benchmark 初始消息用）。"""
    print(f"\n{role}")
    print("-" * 50)
    print(content)


# ---------------------------------------------------------------------------
# Benchmark consultation loop
# ---------------------------------------------------------------------------

def run_consultation(
    case: BenchmarkCase,
    llm_fast: LLMClient,
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
) -> None:
    user_agent = UserAgent(case, llm_fast, llm_strong)
    behaviorist = BehavioristAgent(llm_strong, llm_think, memory=memory)

    print(f"\n{'=' * 50}")
    print(f"Case: {case.case_id}")
    routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
    memory_status = "启用" if Config.memory_enabled else "关闭"
    print(f"路由: {routing_status}  |  记忆: {memory_status}")
    print(f"{'=' * 50}")

    current_msg = case.initial_user_message

    while True:
        _print_static_turn("【猫主人】", current_msg)

        stream, is_done, is_think = behaviorist.ask(current_msg)
        header = "【行为专家 · 深度分析】" if is_think else "【行为专家】"
        behaviorist_text = _stream_turn(header, stream, is_think=is_think)

        if is_done:
            break

        user_stream = user_agent.respond(behaviorist_text)
        current_msg = _stream_turn("【猫主人】", user_stream)

    fallback = behaviorist.round_count > Config.max_conversation_rounds
    print(f"\n{'=' * 50}")
    print(
        f"对话结束 | 共 {behaviorist.round_count} 轮"
        + ("（已触发最大轮数限制）" if fallback else "")
    )
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Free-chat loop
# ---------------------------------------------------------------------------

def run_free_chat(
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
) -> None:
    behaviorist = BehavioristAgent(llm_strong, llm_think, memory=memory)

    routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
    memory_status = "启用" if Config.memory_enabled else "关闭"

    print(f"\n{'=' * 50}")
    print("自由对话模式 — 直接与猫行为专家对话")
    print(f"路由: {routing_status}  |  记忆: {memory_status}")
    print("输入 q / quit / 退出 结束对话")
    print(f"{'=' * 50}")

    while True:
        try:
            user_input = input("\n【你】 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n（对话中断）")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit", "退出"):
            print("（对话结束）")
            break

        stream, is_done, is_think = behaviorist.ask(user_input)
        header = "【行为专家 · 深度分析】" if is_think else "【行为专家】"
        _stream_turn(header, stream, is_think=is_think)

        if is_done:
            print("\n（专家已给出最终建议，对话结束）")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="猫行为咨询 CLI")
    parser.add_argument(
        "--case-id",
        default=None,
        help="指定 benchmark case ID（默认使用第一个，--free-chat 时忽略）",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="列出所有可用的 case ID 后退出",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="关闭模型路由，全部使用 consultation_model（消融实验用）",
    )
    parser.add_argument(
        "--free-chat", "-f",
        action="store_true",
        help="自由对话模式：直接与行为专家对话，不加载 benchmark case",
    )
    return parser.parse_args()


def main() -> None:
    Config.setup_logging()
    args = parse_args()

    if args.no_routing:
        Config.routing_enabled = False

    # --list-cases 不需要 LLM，提前处理
    if args.list_cases:
        cases = load_benchmark()
        print("可用的 case ID：")
        for c in cases:
            print(f"  - {c.case_id}")
        sys.exit(0)

    try:
        llm_fast   = LLMClient.build_for_role("fast")
        llm_strong = LLMClient.build_for_role("strong")
        llm_think  = LLMClient.build_for_role("think")
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    memory = memory_store.get_memory() if Config.memory_enabled else ""

    try:
        if args.free_chat:
            run_free_chat(llm_strong, llm_think, memory=memory)
        else:
            cases = load_benchmark()
            case = select_case(cases, args.case_id)
            run_consultation(case, llm_fast, llm_strong, llm_think, memory=memory)
    except LLMClientError as e:
        print(f"[LLM 错误] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
