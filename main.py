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
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import Config
from agents import memory_store
from agents.consultant_agent import (
    ConsultState,
    ConsultantAgent,
    _update_state_with_tool_result,
)
from agents.user_agent import UserAgent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
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
# Free-chat loop
# ---------------------------------------------------------------------------

def run_free_chat(
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
) -> None:
    """自由对话模式：用户直接与 ConsultantAgent 对话，从 stdin 读取输入。"""
    consultant = ConsultantAgent(llm_strong, llm_think)

    routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
    memory_status = "启用" if Config.memory_enabled else "关闭"
    print(f"\n{'=' * 55}")
    print("自由对话模式 — 直接与猫行为顾问对话")
    print(f"路由: {routing_status}  |  记忆: {memory_status}")
    print("输入 q / quit / exit / 退出 结束对话")
    print(f"{'=' * 55}")

    try:
        initial_input = input("\n【你】 ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n（对话中断）")
        return
    if not initial_input or initial_input.lower() in ("q", "quit", "exit", "退出"):
        return

    session = RunSession.create(None)
    print(f"[Run 目录] {session.folder}")

    rewritten = consultant.rewrite_initial_query(initial_input)
    print(f"\n[Query Rewrite]\n{rewritten}")

    state = ConsultState(
        user_initial_query=initial_input,
        rewritten_initial_query=rewritten,
    )
    session.log_state("init", 0, state, initial_message=initial_input, rewritten=rewritten)

    history: list[BaseMessage] = []
    cur_round = 0
    user_input = initial_input

    while cur_round < Config.max_conversation_rounds:
        state.user_response_this_round = user_input
        history.append(HumanMessage(content=user_input))
        session.log_history(cur_round, "user", user_input)
        session.log_state("user_input", cur_round, state)

        state = _tool_loop(consultant, state, history, session, cur_round, silent=False)

        consult_response = consultant.generate_response(state, history, trajectory=None)
        history.append(AIMessage(content=consult_response.text))
        session.log_history(cur_round, "consultant", consult_response.text)
        session.log_state("consultant_response", cur_round, state,
                          text=consult_response.text, end=consult_response.end)

        print(f"\n【咨询员】")
        print("-" * 50)
        print(consult_response.text)

        if consult_response.end:
            session.log_state("consult_end", cur_round, state)
            break

        cur_round += 1
        try:
            user_input = input("\n【你】 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n（对话中断）")
            break
        if not user_input or user_input.lower() in ("q", "quit", "exit", "退出"):
            print("（对话结束）")
            break

    # 最终建议
    state.resolved = True
    think_stream = consultant.think(state, history)
    _stream_turn("【行为专家 · 深度分析】", think_stream, is_think=True)
    session.log_state("final_state", -1, state)
    print(f"\n[Session 记录已写入: {session.folder}]")



# ---------------------------------------------------------------------------
# Consultant loop (new architecture)
# ---------------------------------------------------------------------------

@dataclass
class RunSession:
    """每次运行创建独立文件夹，分别记录 state / history / trajectory。"""
    folder: Path

    def __post_init__(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._state_path      = self.folder / "state.jsonl"
        self._history_path    = self.folder / "history.jsonl"
        self._trajectory_path = self.folder / "trajectory.jsonl"
        # 清空（本次 run 全新开始）
        for p in (self._state_path, self._history_path, self._trajectory_path):
            p.write_text("", encoding="utf-8")

    # --- 三个追加写入方法 ---

    def log_state(self, event: str, round: int, state: ConsultState, **extra) -> None:
        entry = {"round": round, "event": event, "state": state.model_dump(), **extra}
        self._append(self._state_path, entry)

    def log_history(self, round: int, role: str, content: str) -> None:
        entry = {"round": round, "role": role, "content": content}
        self._append(self._history_path, entry)

    def log_trajectory(self, round: int, step: dict) -> None:
        entry = {"round": round, **step}
        self._append(self._trajectory_path, entry)

    @staticmethod
    def _append(path: Path, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @staticmethod
    def create(case_id: int | None) -> "RunSession":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"case{case_id}" if case_id is not None else "free"
        folder = Config.project_root / "run" / f"{timestamp}_{label}"
        return RunSession(folder=folder)


def _tool_loop(
    consultant: ConsultantAgent,
    state: ConsultState,
    history: list[BaseMessage],
    session: RunSession,
    round_num: int,
    silent: bool,
) -> ConsultState:
    """内循环：让 consultant 决定是否调用工具，最多 max_intermediate_turn 次。

    工具调用结果作为 AIMessage 写入 history，同时更新 state.collected_evidence。
    每次变更后立即写文件。
    """
    max_intermediate_turn = 3
    tool_trajectory: list[dict] = []

    for _ in range(max_intermediate_turn):
        intermediate = consultant.generate_intermediate_response(state, history, tool_trajectory)

        if intermediate.end_tool_call():
            break

        if intermediate.is_tool_call():
            tool_result = consultant.execute_tool(intermediate)

            # 工具结果写入 history（AIMessage）
            tool_msg = f"[{intermediate.tool_name}] {tool_result}"
            history.append(AIMessage(content=tool_msg))

            # 更新 state
            state = _update_state_with_tool_result(state, intermediate.tool_name, tool_result)

            step = {
                "thought": intermediate.thought,
                "tool_name": intermediate.tool_name,
                "tool_args": intermediate.tool_args,
                "observation": tool_result,
            }
            tool_trajectory.append(step)

            session.log_trajectory(round_num, step)
            session.log_state("tool_result", round_num, state)
            if not silent:
                print(f"\n[工具: {intermediate.tool_name}] {tool_result}")

    return state


def run_consultant_loop(
    case: BenchmarkCase,
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
    silent: bool = False,
) -> ConsultationResult:
    """Consultant 对话循环，遵循 prompt-flow.txt 伪代码架构。

    外循环驱动用户交互：
      1. UserAgent 提供用户输入 → 写入 history
      2. 内循环（tool loop）刷满工具状态
      3. ConsultantAgent 生成正式回复 → 写入 history
      4. 如 consultant 认为信息已足够（end=True），退出外循环
      5. 最终调用 consultant.think() 流式输出深度建议

    state 和 history 每次变更后立即写入独立 run 文件夹的三个 JSONL 文件。
    返回 ConsultationResult 供评测使用。
    """
    session = RunSession.create(case.case_id)
    if not silent:
        print(f"[Run 目录] {session.folder}")

    consultant = ConsultantAgent(llm_strong, llm_think)
    user_agent = UserAgent(case, llm_strong)

    history: list[BaseMessage] = []
    conv_history: list[dict] = []   # 纯用户/咨询员消息，供评测使用
    question_turns = 0

    # 1. 获取初始消息
    initial_message = case.initial_user_message
    if not silent:
        print(f"\n{'=' * 55}")
        print(f"Case {case.case_id}")
        routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
        memory_status = "启用" if Config.memory_enabled else "关闭"
        print(f"路由: {routing_status}  |  记忆: {memory_status}")
        print(f"{'=' * 55}")
        _print_static_turn("【猫主人（初始问题）】", initial_message)

    # 2. Rewrite 初始 query
    rewritten = consultant.rewrite_initial_query(initial_message)
    if not silent:
        print(f"\n[Query Rewrite]\n{rewritten}")

    # 3. 初始化 state（含假设）
    initial_hypotheses = consultant.initialize_hypotheses(initial_message, rewritten)
    state = ConsultState(
        user_initial_query=initial_message,
        rewritten_initial_query=rewritten,
        hypothesis=initial_hypotheses,
    )
    session.log_state("init", 0, state, initial_message=initial_message, rewritten=rewritten)

    max_consult_round = Config.max_conversation_rounds

    for consult_cur_round in range(max_consult_round):
        # 4. 用户输入（第 0 轮用初始消息，之后由 UserAgent 回答上一轮 consultant 的话）
        if consult_cur_round == 0:
            user_input = initial_message
        else:
            last_consultant_msg = history[-1].content if history else ""
            user_stream = user_agent.respond(last_consultant_msg)
            if not silent:
                user_input = _stream_turn("【猫主人】", user_stream)
            else:
                user_input = _consume_stream(user_stream)
            question_turns += 1

        state.user_response_this_round = user_input
        history.append(HumanMessage(content=user_input))
        conv_history.append({"role": "user", "content": user_input})
        session.log_history(consult_cur_round, "user", user_input)
        session.log_state("user_input", consult_cur_round, state)

        # 5. 内循环：刷满工具状态
        state = _tool_loop(consultant, state, history, session, consult_cur_round, silent)

        # 5b. 更新假设置信度
        state = consultant.update_state(state, history, user_input)
        session.log_state("state_updated", consult_cur_round, state)

        # 6. 生成正式回复
        consult_response = consultant.generate_response(state, history, trajectory=None)
        history.append(AIMessage(content=consult_response.text))
        conv_history.append({"role": "assistant", "content": consult_response.text})
        session.log_history(consult_cur_round, "consultant", consult_response.text)
        session.log_state("consultant_response", consult_cur_round, state,
                          text=consult_response.text, end=consult_response.end)

        if not silent:
            print(f"\n【咨询员】")
            print("-" * 50)
            print(consult_response.text)

        if consult_response.end:
            session.log_state("consult_end", consult_cur_round, state)
            break

    else:
        if not silent:
            print(f"\n[已达最大轮数 {max_consult_round}，进入最终建议]")

    # 7. 最终 think：流式输出深度建议
    state.resolved = True
    think_stream = consultant.think(state, history)
    if not silent:
        final_conclusion = _stream_turn("【行为专家 · 深度分析】", think_stream, is_think=True)
    else:
        final_conclusion = _consume_stream(think_stream)
    session.log_state("final_state", -1, state)

    if not silent:
        print(f"\n[Session 记录已写入: {session.folder}]")

    return ConsultationResult(
        case_id=case.case_id,
        conversation_history=conv_history,
        question_turns=question_turns,
        final_conclusion=final_conclusion,
    )


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
            run_consultant_loop(case, llm_strong, llm_think, memory=memory)
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
