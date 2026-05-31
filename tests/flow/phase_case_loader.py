from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.consultant_agent import ConsultState, HypothesisItem
from consultation_runtime import RuntimeContext
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str) -> dict[str, Any]:
    path = _FIXTURE_PATH / name
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(records: list[dict[str, str]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for record in records:
        role = record["role"]
        content = record["content"]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            raise ValueError(f"Unsupported role in fixture: {role}")
    return messages


def build_state(payload: dict[str, Any]) -> ConsultState:
    data = dict(payload)
    if "hypothesis" in data:
        data["hypothesis"] = [HypothesisItem(**item) for item in data["hypothesis"]]
    return ConsultState(**data)


def build_runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    history_records = payload.get("history", [])
    return RuntimeContext(
        state=build_state(payload["state"]),
        history=build_messages(history_records),
        conversation_history=payload.get("conversation_history", history_records),
        question_turns=payload.get("question_turns", 0),
        case_id=payload.get("case_id", 1),
        mode=payload.get("mode", "benchmark"),
    )