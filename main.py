"""CLI 入口：运行猫行为咨询对话。"""

from __future__ import annotations

import argparse
import sys

import Config
from agents import memory_store
from checkpoint_store import CheckpointStore
from consultation_runtime import (
    run_consultant_loop,
    run_free_chat,
    restore_case_from_store,
)
from utils.benchmark_loader import load_benchmark, select_case
from utils.llm_client import LLMClient, LLMClientError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="猫行为咨询 CLI")
    parser.add_argument("--case-id", type=int, default=None, help="指定 benchmark case ID（默认使用第一个）")
    parser.add_argument("--list-cases", action="store_true", help="列出所有可用的 case ID 后退出")
    parser.add_argument("--no-routing", action="store_true", help="关闭模型路由，全部使用 consultation_model")
    parser.add_argument("--free-chat", "-f", action="store_true", help="自由对话模式")
    parser.add_argument("--batch-eval", "-b", action="store_true", help="批量评测模式")
    parser.add_argument("--no-tracing", action="store_true", help="强制关闭 LangSmith tracing")
    parser.add_argument("--resume-run", type=str, default=None, help="从指定 run 目录恢复 benchmark 流程")
    parser.add_argument("--checkpoint-id", type=str, default=None, help="恢复指定 checkpoint；默认最新")
    parser.add_argument("--list-checkpoints", action="store_true", help="列出指定 run 的 checkpoint 摘要后退出")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    Config.setup_logging()
    args = parse_args()

    if not args.no_tracing:
        Config.setup_tracing()

    if args.no_routing:
        Config.routing_enabled = False

    if args.list_cases:
        cases = load_benchmark()
        print("可用的 case ID：")
        for c in cases:
            print(f"  - {c.case_id}")
        sys.exit(0)

    if args.list_checkpoints:
        if not args.resume_run:
            raise ValueError("--list-checkpoints 需要配合 --resume-run")
        store = CheckpointStore.open_existing(Config.project_root / "run" / args.resume_run)
        for summary in store.checkpoint_summaries():
            print(summary)
        sys.exit(0)

    try:
        llm_fast = LLMClient.build_for_role("fast")
        llm_strong = LLMClient.build_for_role("strong")
        llm_think = LLMClient.build_for_role("think")
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
        elif args.resume_run:
            source_store = CheckpointStore.open_existing(Config.project_root / "run" / args.resume_run)
            checkpoint = source_store.load_checkpoint(args.checkpoint_id)
            case = restore_case_from_store(source_store)
            checkpoint_store = CheckpointStore.create_resume(
                case_id=case.case_id,
                mode="benchmark",
                source_run=source_store.folder.name,
                source_checkpoint_id=checkpoint.checkpoint_id,
            )
            run_consultant_loop(
                case,
                llm_strong,
                llm_think,
                memory=memory,
                checkpoint_store=checkpoint_store,
                resume_checkpoint=checkpoint,
            )
        else:
            cases = load_benchmark()
            case = select_case(cases, args.case_id)
            run_consultant_loop(
                case,
                llm_strong,
                llm_think,
                memory=memory,
            )
    except LLMClientError as e:
        print(f"[LLM 错误] {e}")
        sys.exit(1)


def _print_evaluation_report(result) -> None:
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