from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.consultant_agent import ConsultantAgent
from consultation_runtime import _run_consult_response_phase_core, _run_state_update_phase_core, _run_tool_phase_core
from tests.flow.phase_case_loader import build_runtime_context, load_fixture
from utils.llm_client import LLMClient


def _build_consultant() -> ConsultantAgent:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("DASHSCOPE_API_KEY 未设置，无法运行 prompt 调试脚本。")
    llm_fast = LLMClient.build_for_role("fast")
    llm_strong = LLMClient.build_for_role("strong")
    llm_think = LLMClient.build_for_role("think")
    return ConsultantAgent(llm_fast, llm_strong, llm_think)


def _dump(title: str, payload: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _matches_case(case_name: str, keyword: str | None) -> bool:
    if not keyword:
        return True
    return keyword.lower() in case_name.lower()


def run_tool_phase_debug(consultant: ConsultantAgent, fixture: dict, case_keyword: str | None) -> None:
    for case in fixture.get("tool_phase", []):
        if not _matches_case(case["name"], case_keyword):
            continue
        ctx = build_runtime_context(case)
        payload = {
            "case": case["name"],
            "context": {
                "state": ctx.state.model_dump(),
                "history": case.get("history", []),
            },
            "output": _run_tool_phase_core(consultant, ctx, trajectory=case.get("trajectory", []))[1],
        }
        _dump(f"tool_phase::{case['name']}", payload)


def run_state_update_phase_debug(consultant: ConsultantAgent, fixture: dict, case_keyword: str | None) -> None:
    for case in fixture.get("state_update", []):
        if not _matches_case(case["name"], case_keyword):
            continue
        ctx = build_runtime_context(case)
        payload = {
            "case": case["name"],
            "context": {
                "state": ctx.state.model_dump(),
                "history": case.get("history", []),
            },
            "output": _run_state_update_phase_core(consultant, ctx)[1],
        }
        _dump(f"state_update::{case['name']}", payload)


def run_consult_response_debug(consultant: ConsultantAgent, fixture: dict, case_keyword: str | None) -> None:
    for case in fixture.get("consult_response", []):
        if not _matches_case(case["name"], case_keyword):
            continue
        ctx = build_runtime_context(case)
        payload = {
            "case": case["name"],
            "context": {
                "state": ctx.state.model_dump(),
                "history": case.get("history", []),
            },
            "output": _run_consult_response_phase_core(consultant, ctx)[2],
        }
        _dump(f"consult_response::{case['name']}", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt phase debug runner")
    parser.add_argument("--phase", choices=["tool", "state_update", "consult", "all"], default="all")
    parser.add_argument("--case", dest="case_keyword", help="只运行名称包含该关键字的场景")
    args = parser.parse_args()

    fixture = load_fixture("prompt_phase_cases.json")
    consultant = _build_consultant()

    if args.phase in ("tool", "all"):
        run_tool_phase_debug(consultant, fixture, args.case_keyword)
    if args.phase in ("state_update", "all"):
        run_state_update_phase_debug(consultant, fixture, args.case_keyword)
    if args.phase in ("consult", "all"):
        run_consult_response_debug(consultant, fixture, args.case_keyword)


if __name__ == "__main__":
    main()