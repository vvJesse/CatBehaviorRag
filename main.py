"""CLI 入口：运行猫行为咨询对话。

用法示例：
    python main.py                          # 运行第一个 benchmark case
    python main.py --case-id cat_single_case_01
    python main.py --case-id cat_single_case_02
    python main.py --list-cases             # 列出所有可用 case
"""

from __future__ import annotations

import argparse
import sys

import Config
from agents.behaviorist_agent import BehavioristAgent
from agents.user_agent import UserAgent
from utils.benchmark_loader import BenchmarkCase, load_benchmark, select_case
from utils.llm_client import LLMClient, LLMClientError


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_turn(role: str, content: str) -> None:
    print(f"\n{role}")
    print("-" * 40)
    print(content)


# ---------------------------------------------------------------------------
# Core consultation loop
# ---------------------------------------------------------------------------

def run_consultation(case: BenchmarkCase, llm: LLMClient) -> None:
    user_agent = UserAgent(case, llm)
    behaviorist = BehavioristAgent(llm)

    print(f"\n{'=' * 50}")
    print(f"Case: {case.case_id}")
    print(f"{'=' * 50}")

    current_msg = case.initial_user_message

    while True:
        _print_turn("【猫主人】", current_msg)

        response, is_done = behaviorist.ask(current_msg)
        _print_turn("【行为专家】", response)

        if is_done:
            break

        current_msg = user_agent.respond(response)

    fallback = behaviorist.round_count > Config.max_conversation_rounds
    print(f"\n{'=' * 50}")
    print(f"对话结束 | 共 {behaviorist.round_count} 轮" + ("（已触发最大轮数限制）" if fallback else ""))
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="猫行为咨询 CLI")
    parser.add_argument(
        "--case-id",
        default=None,
        help="指定 benchmark case ID（默认使用第一个）",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="列出所有可用的 case ID 后退出",
    )
    return parser.parse_args()


def main() -> None:
    Config.setup_logging()
    args = parse_args()

    cases = load_benchmark()

    if args.list_cases:
        print("可用的 case ID：")
        for c in cases:
            print(f"  - {c.case_id}")
        sys.exit(0)

    case = select_case(cases, args.case_id)

    try:
        llm = LLMClient.build_from_config()
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)

    try:
        run_consultation(case, llm)
    except LLMClientError as e:
        print(f"[LLM 错误] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
