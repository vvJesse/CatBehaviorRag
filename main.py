"""CLI 入口：运行猫行为咨询对话。

用法示例：
    python main.py                          # 运行第一个 benchmark case
    python main.py --case-id 1
    python main.py --case-id 2
    python main.py --list-cases             # 列出所有可用 case
    python main.py --no-routing             # 关闭路由（消融实验）
    python main.py --free-chat              # 自由对话模式（输入 q/退出 结束）
    python main.py --batch-eval             # 批量评测所有 case
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterator

import Config
from agents import memory_store
from agents.behaviorist_agent import BehavioristAgent
from agents.user_agent import UserAgent
from utils.benchmark_loader import BenchmarkCase, load_benchmark, select_case
from utils.llm_client import LLMClient, LLMClientError


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConsultationResult:
    case_id: int
    conversation_history: list[dict]
    question_turns: int
    final_conclusion: str


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _stream_turn(role: str, stream: Iterator[str], is_think: bool = False) -> str:
    """流式打印一轮对话，返回完整文本。

    思考模型的 <|thinks|>...<|/thinks|> 内容会以暗色展示。
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
        if is_think and not in_think and "<|thinks|>" in buffer:
            in_think = True
            before, _, after = buffer.partition("<|thinks|>")
            if before:
                print(before, end="", flush=True)
            print("\n\033[2m[思考过程]\033[0m", flush=True)
            buffer = after
            continue

        # 检测思考块结束
        if is_think and in_think and "<|/thinks|>" in buffer:
            in_think = False
            think_content, _, after = buffer.partition("<|/thinks|>")
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


def _consume_stream(stream: Iterator[str]) -> str:
    """消费流式迭代器但不打印，返回完整文本。用于 silent 模式。"""
    chunks: list[str] = []
    for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


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
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
    silent: bool = False,
) -> ConsultationResult:
    user_agent = UserAgent(case, llm_strong)
    behaviorist = BehavioristAgent(llm_strong, llm_think, memory=memory)
    question_turns = 0

    if not silent:
        print(f"\n{'=' * 50}")
        print(f"Case: {case.case_id}")
        routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
        memory_status = "启用" if Config.memory_enabled else "关闭"
        print(f"路由: {routing_status}  |  记忆: {memory_status}")
        print(f"{'=' * 50}")

    current_msg = case.initial_user_message

    if not silent:
        _print_static_turn("【猫主人】", current_msg)

    while True:
        stream, is_done, is_think = behaviorist.ask(current_msg)

        if not silent:
            header = "【行为专家 · 深度分析】" if is_think else "【行为专家】"
            behaviorist_text = _stream_turn(header, stream, is_think=is_think)
        else:
            behaviorist_text = _consume_stream(stream)

        if is_done:
            break

        question_turns += 1

        user_stream = user_agent.respond(behaviorist_text)
        if not silent:
            current_msg = _stream_turn("【猫主人】", user_stream)
        else:
            current_msg = _consume_stream(user_stream)

    final_conclusion = behaviorist.final_conclusion or ""

    if not silent:
        fallback = behaviorist.round_count > Config.max_conversation_rounds
        print(f"\n{'=' * 50}")
        print(
            f"对话结束 | 共 {behaviorist.round_count} 轮"
            + ("（已触发最大轮数限制）" if fallback else "")
        )
        print(f"{'=' * 50}\n")

    return ConsultationResult(
        case_id=case.case_id,
        conversation_history=behaviorist.conversation_history,
        question_turns=question_turns,
        final_conclusion=final_conclusion,
    )


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
    print("输入 q / quit / exit / 退出 结束对话")
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
        type=int,
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
    parser.add_argument(
        "--batch-eval", "-b",
        action="store_true",
        help="批量评测模式：运行所有 benchmark case 并计算评估指标",
    )
    parser.add_argument(
        "--no-tracing",
        action="store_true",
        help="强制关闭 LangSmith tracing，即使环境变量已设置",
    )
    return parser.parse_args()


def main() -> None:
    # Windows 控制台默认使用 GBK 编码，无法输出 LLM 返回的 emoji 字符
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    Config.setup_logging()
    args = parse_args()

    if not args.no_tracing:
        Config.setup_tracing()

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
        if args.batch_eval:
            from evaluation.batch_eval import run_batch_evaluation
            result = run_batch_evaluation(llm_fast, llm_strong, llm_think, memory=memory)
            _print_evaluation_report(result)
        elif args.free_chat:
            run_free_chat(llm_strong, llm_think, memory=memory)
        else:
            cases = load_benchmark()
            case = select_case(cases, args.case_id)
            run_consultation(case, llm_strong, llm_think, memory=memory)
    except LLMClientError as e:
        print(f"[LLM 错误] {e}")
        sys.exit(1)


def _print_evaluation_report(result) -> None:
    """打印批量评测报告。"""
    from evaluation.batch_eval import BatchEvaluationResult
    print(f"\n{'=' * 55}")
    print("  Batch Evaluation Report")
    print(f"{'=' * 55}")
    print(f"  Total cases: {result.total_cases}")
    print()
    print(f"  AQT (Average Question Turns): {result.aqt:.2f}")
    print(f"  UFS (User-Friendliness, norm): {result.mean_ufs_norm:.2f}")
    print(f"  DCS (Direction Coverage):      {result.mean_dcs:.2f}")
    print(f"  AS  (Actionability, norm):     {result.mean_as_norm:.2f}")
    print(f"  UHA (Uncertainty Handling):    {result.uha:.2f}")
    print()
    print("  Per-Case Details:")
    print(f"  {'-' * 49}")
    for e in result.case_evaluations:
        uha_str = "correct" if e.uha_correct else "incorrect"
        print(
            f"  Case {e.case_id}: "
            f"QT={e.question_turns}, "
            f"UFS={e.ufs_raw}({e.ufs_norm:.2f}), "
            f"DCS={e.dcs:.2f}, "
            f"AS={e.as_raw}({e.as_norm:.2f}), "
            f"UHA={uha_str}"
        )
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
