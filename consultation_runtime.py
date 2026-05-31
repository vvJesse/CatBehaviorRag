from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import Config
from agents.consultant_agent import ConsultResponse, ConsultState, ConsultantAgent, _update_state_with_tool_result
from agents.user_agent import UserAgent
from checkpoint_store import CheckpointStore, RunCheckpoint, history_from_records
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from utils.benchmark_loader import BenchmarkCase, load_benchmark, select_case
from utils.llm_client import LLMClient


@dataclass
class ConsultationResult:
    case_id: int
    conversation_history: list[dict]
    question_turns: int
    final_conclusion: str


@dataclass
class RuntimeContext:
    state: ConsultState
    history: list[BaseMessage]
    conversation_history: list[dict]
    question_turns: int
    case_id: int | None
    mode: str


def _stream_turn(role: str, stream: Iterator[str], is_think: bool = False) -> str:
    print(f"\n{role}")
    print("-" * 50)

    full_text: list[str] = []
    buffer = ""
    in_think = False

    for chunk in stream:
        buffer += chunk
        full_text.append(chunk)

        if is_think and not in_think and "<|thinks|>" in buffer:
            in_think = True
            before, _, after = buffer.partition("<|thinks|>")
            if before:
                print(before, end="", flush=True)
            print("\n\033[2m[思考过程]\033[0m", flush=True)
            buffer = after
            continue

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
    chunks: list[str] = []
    for chunk in stream:
        chunks.append(chunk)
    return "".join(chunks)


def _print_static_turn(role: str, content: str) -> None:
    print(f"\n{role}")
    print("-" * 50)
    print(content)


def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _log_state(folder: Path, event: str, round_num: int, state: ConsultState, **extra) -> None:
    _append_jsonl(
        folder / "state.jsonl",
        {"round": round_num, "event": event, "state": state.model_dump(), **extra},
    )


def _log_history(folder: Path, round_num: int, role: str, content: str) -> None:
    _append_jsonl(folder / "history.jsonl", {"round": round_num, "role": role, "content": content})


def _log_trajectory(folder: Path, round_num: int, step: dict) -> None:
    _append_jsonl(folder / "trajectory.jsonl", {"round": round_num, **step})


def _ensure_log_files(folder: Path, reset: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("state.jsonl", "history.jsonl", "trajectory.jsonl"):
        path = folder / filename
        if reset:
            path.write_text("", encoding="utf-8")
        elif not path.exists():
            path.write_text("", encoding="utf-8")


def restore_case_from_store(store: CheckpointStore) -> BenchmarkCase:
    manifest = store.load_manifest()
    case_id = manifest.get("case_id")
    if case_id is None:
        raise ValueError("当前 run 不对应 benchmark case，暂不支持该模式恢复")
    return select_case(load_benchmark(), case_id)


def _context_from_checkpoint(checkpoint: RunCheckpoint) -> RuntimeContext:
    return RuntimeContext(
        state=checkpoint.state.model_copy(deep=True),
        history=history_from_records(checkpoint.history),
        conversation_history=list(checkpoint.conversation_history),
        question_turns=checkpoint.question_turns,
        case_id=checkpoint.case_id,
        mode=checkpoint.mode,
    )


def _create_initial_context(
    consultant: ConsultantAgent,
    initial_input: str,
    case_id: int | None,
    mode: str,
    silent: bool,
    consultation_date: str | None = None,
    consultation_latitude: str | None = None,
    consultation_longitude: str | None = None,
) -> RuntimeContext:
    rewritten = consultant.rewrite_initial_query(initial_input)
    if not silent:
        print(f"\n[Query Rewrite]\n{rewritten}")
    initial_hypotheses = consultant.initialize_hypotheses(initial_input, rewritten)
    return RuntimeContext(
        state=ConsultState(
            user_initial_query=initial_input,
            consultation_date=consultation_date,
            consultation_latitude=consultation_latitude,
            consultation_longitude=consultation_longitude,
            rewritten_initial_query=rewritten,
            hypothesis=initial_hypotheses,
        ),
        history=[],
        conversation_history=[],
        question_turns=0,
        case_id=case_id,
        mode=mode,
    )


def _apply_user_input(
    folder: Path,
    ctx: RuntimeContext,
    round_num: int,
    user_input: str,
) -> RuntimeContext:
    ctx.state.user_response_this_round = user_input
    ctx.history.append(HumanMessage(content=user_input))
    ctx.conversation_history.append({"role": "user", "content": user_input})
    _log_history(folder, round_num, "user", user_input)
    _log_state(folder, "user_input", round_num, ctx.state)
    return ctx


def _run_tool_phase_core(
    consultant: ConsultantAgent,
    ctx: RuntimeContext,
    trajectory: list[dict] | None = None,
) -> tuple[RuntimeContext, dict]:
    tool_trajectory = list(trajectory or [])
    steps: list[dict] = []
    for _ in range(3):
        intermediate = consultant.generate_intermediate_response(ctx.state, ctx.history, tool_trajectory)
        intermediate_payload = intermediate.model_dump()
        if intermediate.end_tool_call():
            return ctx, {
                "intermediate": intermediate_payload,
                "steps": steps,
                "trajectory": tool_trajectory,
            }
        if not intermediate.is_tool_call():
            steps.append({"intermediate": intermediate_payload})
            continue
        tool_result = consultant.execute_tool(intermediate)
        ctx.history.append(AIMessage(content=f"[{intermediate.tool_name}] {tool_result}"))
        ctx.state = _update_state_with_tool_result(ctx.state, intermediate.tool_name, tool_result)
        step = {
            "thought": intermediate.thought,
            "tool_name": intermediate.tool_name,
            "tool_args": intermediate.tool_args,
            "observation": tool_result,
        }
        tool_trajectory.append(step)
        steps.append(
            {
                "intermediate": intermediate_payload,
                "tool_result": tool_result,
                "state_after_tool": ctx.state.model_dump(),
                "step": step,
            }
        )
    return ctx, {
        "intermediate": None,
        "steps": steps,
        "trajectory": tool_trajectory,
    }


def _execute_tool_phase(
    consultant: ConsultantAgent,
    folder: Path,
    ctx: RuntimeContext,
    round_num: int,
    silent: bool,
) -> RuntimeContext:
    ctx, payload = _run_tool_phase_core(consultant, ctx)
    for item in payload["steps"]:
        step = item.get("step")
        if step is None:
            continue
        _log_trajectory(folder, round_num, step)
        _log_state(folder, "tool_result", round_num, ctx.state)
        if not silent:
            print(f"\n[工具: {step['tool_name']}] {step['observation']}")
    return ctx


def _run_state_update_phase_core(
    consultant: ConsultantAgent,
    ctx: RuntimeContext,
) -> tuple[RuntimeContext, dict]:
    before_state = ctx.state.model_dump()
    user_response = ctx.state.user_response_this_round or ""
    ctx.state = consultant.update_state(ctx.state, ctx.history, user_response)
    return ctx, {
        "user_response": user_response,
        "before_state": before_state,
        "after_state": ctx.state.model_dump(),
    }


def _execute_state_update(
    consultant: ConsultantAgent,
    folder: Path,
    ctx: RuntimeContext,
    round_num: int,
) -> RuntimeContext:
    ctx, _ = _run_state_update_phase_core(consultant, ctx)
    _log_state(folder, "state_updated", round_num, ctx.state)
    return ctx


def _run_consult_response_phase_core(
    consultant: ConsultantAgent,
    ctx: RuntimeContext,
) -> tuple[RuntimeContext, ConsultResponse, dict]:
    consult_response = consultant.generate_response(ctx.state, ctx.history, trajectory=None)
    ctx.history.append(AIMessage(content=consult_response.text))
    ctx.conversation_history.append({"role": "assistant", "content": consult_response.text})
    return ctx, consult_response, {
        "response": consult_response.model_dump(),
        "updated_context": {
            "state": ctx.state.model_dump(),
            "history": [{"type": message.type, "content": message.content} for message in ctx.history],
            "conversation_history": list(ctx.conversation_history),
        },
    }


def _execute_consult_response(
    consultant: ConsultantAgent,
    folder: Path,
    ctx: RuntimeContext,
    round_num: int,
    silent: bool,
) -> tuple[RuntimeContext, ConsultResponse]:
    ctx, consult_response, _ = _run_consult_response_phase_core(consultant, ctx)
    _log_history(folder, round_num, "consultant", consult_response.text)
    _log_state(
        folder,
        "consultant_response",
        round_num,
        ctx.state,
        text=consult_response.text,
        end=consult_response.end,
    )
    if not silent:
        print(f"\n【咨询员】")
        print("-" * 50)
        print(consult_response.text)
    return ctx, consult_response


def _execute_round(
    consultant: ConsultantAgent,
    folder: Path,
    ctx: RuntimeContext,
    round_num: int,
    user_input: str,
    silent: bool,
) -> tuple[RuntimeContext, bool]:
    ctx = _apply_user_input(folder, ctx, round_num, user_input)
    ctx = _execute_tool_phase(consultant, folder, ctx, round_num, silent)
    ctx = _execute_state_update(consultant, folder, ctx, round_num)
    ctx, consult_response = _execute_consult_response(consultant, folder, ctx, round_num, silent)
    return ctx, consult_response.end


def _run_final_think(
    consultant: ConsultantAgent,
    folder: Path,
    store: CheckpointStore,
    ctx: RuntimeContext,
    round_num: int,
    silent: bool,
) -> str:
    ctx.state.resolved = True
    think_stream = consultant.think(ctx.state, ctx.history)
    final_conclusion = (
        _stream_turn("【行为专家 · 深度分析】", think_stream, is_think=True)
        if not silent
        else _consume_stream(think_stream)
    )
    _log_state(folder, "final_state", -1, ctx.state, final_conclusion=final_conclusion)
    if Config.checkpoint_enabled:
        store.save_final(
            round_num=round_num,
            case_id=ctx.case_id,
            mode=ctx.mode,
            state=ctx.state,
            history=ctx.history,
            conversation_history=ctx.conversation_history,
            question_turns=ctx.question_turns,
            final_conclusion=final_conclusion,
        )
    return final_conclusion


def _run_session_loop(
    consultant: ConsultantAgent,
    checkpoint_store: CheckpointStore,
    ctx: RuntimeContext,
    get_user_input: Callable[[int, RuntimeContext, bool], str | None],
    silent: bool,
    start_round: int,
) -> tuple[RuntimeContext, int]:
    last_round = start_round - 1
    for consult_cur_round in range(start_round, Config.max_conversation_rounds):
        user_input = get_user_input(consult_cur_round, ctx, silent)
        if user_input is None:
            break
        if consult_cur_round > 0:
            ctx.question_turns += 1

        ctx, should_end = _execute_round(
            consultant=consultant,
            folder=checkpoint_store.folder,
            ctx=ctx,
            round_num=consult_cur_round,
            user_input=user_input,
            silent=silent,
        )
        last_round = consult_cur_round
        if Config.checkpoint_enabled:
            checkpoint_store.save_round(
                round_num=consult_cur_round,
                case_id=ctx.case_id,
                mode=ctx.mode,
                state=ctx.state,
                history=ctx.history,
                conversation_history=ctx.conversation_history,
                question_turns=ctx.question_turns,
            )
        if should_end:
            _log_state(checkpoint_store.folder, "consult_end", consult_cur_round, ctx.state)
            break
    return ctx, last_round


def _build_benchmark_user_input_provider(
    case: BenchmarkCase,
    user_agent: UserAgent,
) -> Callable[[int, RuntimeContext, bool], str | None]:
    def provider(round_num: int, ctx: RuntimeContext, silent: bool) -> str | None:
        if round_num == 0:
            return case.initial_user_message
        user_stream = user_agent.respond(list(ctx.conversation_history))
        return _stream_turn("【猫主人】", user_stream) if not silent else _consume_stream(user_stream)

    return provider


def _build_free_chat_user_input_provider() -> Callable[[int, RuntimeContext, bool], str | None]:
    def provider(round_num: int, ctx: RuntimeContext, silent: bool) -> str | None:
        if round_num == 0:
            return ctx.state.user_initial_query
        try:
            user_input = input("\n【你】 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n（对话中断）")
            return None
        if not user_input or user_input.lower() in ("q", "quit", "exit", "退出"):
            print("（对话结束）")
            return None
        return user_input

    return provider


def run_consultant_loop(
    case: BenchmarkCase,
    llm_fast: LLMClient,
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
    silent: bool = False,
    checkpoint_store: CheckpointStore | None = None,
    resume_checkpoint: RunCheckpoint | None = None,
) -> ConsultationResult:
    mode = "benchmark"
    checkpoint_store = checkpoint_store or CheckpointStore.create(case.case_id, mode=mode)
    _ensure_log_files(checkpoint_store.folder, reset=False)
    if not silent:
        print(f"[Run 目录] {checkpoint_store.folder}")

    consultant = ConsultantAgent(llm_fast, llm_strong, llm_think)
    user_agent = UserAgent(case, llm_strong)

    if resume_checkpoint is None:
        if not silent:
            print(f"\n{'=' * 55}")
            print(f"Case {case.case_id}")
            routing_status = "启用" if Config.routing_enabled else "关闭（消融模式）"
            memory_status = "启用" if Config.memory_enabled else "关闭"
            print(f"路由: {routing_status}  |  记忆: {memory_status}")
            print(f"{'=' * 55}")
            _print_static_turn("【猫主人（初始问题）】", case.initial_user_message)
        ctx = _create_initial_context(
            consultant=consultant,
            initial_input=case.initial_user_message,
            case_id=case.case_id,
            mode=mode,
            silent=silent,
            consultation_date=case.date,
            consultation_latitude=case.latitude,
            consultation_longitude=case.longitude,
        )
        start_round = 0
    else:
        ctx = _context_from_checkpoint(resume_checkpoint)
        if resume_checkpoint.checkpoint_type == "final_state" and resume_checkpoint.final_conclusion is not None:
            return ConsultationResult(
                case_id=case.case_id,
                conversation_history=ctx.conversation_history,
                question_turns=ctx.question_turns,
                final_conclusion=resume_checkpoint.final_conclusion,
            )
        start_round = resume_checkpoint.round + 1

    ctx, last_round = _run_session_loop(
        consultant=consultant,
        checkpoint_store=checkpoint_store,
        ctx=ctx,
        get_user_input=_build_benchmark_user_input_provider(case, user_agent),
        silent=silent,
        start_round=start_round,
    )
    if last_round < start_round and not silent:
        print("\n[未执行新的 round，直接进入最终建议]")
    elif last_round == Config.max_conversation_rounds - 1 and not ctx.state.resolved and not silent:
        print(f"\n[已达最大轮数 {Config.max_conversation_rounds}，进入最终建议]")

    final_conclusion = _run_final_think(
        consultant=consultant,
        folder=checkpoint_store.folder,
        store=checkpoint_store,
        ctx=ctx,
        round_num=last_round,
        silent=silent,
    )
    if not silent:
        print(f"\n[Session 记录已写入: {checkpoint_store.folder}]")
    return ConsultationResult(
        case_id=case.case_id,
        conversation_history=ctx.conversation_history,
        question_turns=ctx.question_turns,
        final_conclusion=final_conclusion,
    )


def run_free_chat(
    llm_fast: LLMClient,
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
) -> None:
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

    store = CheckpointStore.create(None, mode="free")
    _ensure_log_files(store.folder, reset=False)
    print(f"[Run 目录] {store.folder}")

    consultant = ConsultantAgent(llm_fast, llm_strong, llm_think)
    ctx = _create_initial_context(
        consultant=consultant,
        initial_input=initial_input,
        case_id=None,
        mode="free",
        silent=False,
        consultation_date=None,
        consultation_latitude=None,
        consultation_longitude=None,
    )
    ctx, last_round = _run_session_loop(
        consultant=consultant,
        checkpoint_store=store,
        ctx=ctx,
        get_user_input=_build_free_chat_user_input_provider(),
        silent=False,
        start_round=0,
    )
    _run_final_think(
        consultant=consultant,
        folder=store.folder,
        store=store,
        ctx=ctx,
        round_num=last_round,
        silent=False,
    )
    print(f"\n[Session 记录已写入: {store.folder}]")