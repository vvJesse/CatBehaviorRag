from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import Config
from agents.consultant_agent import ConsultState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


@dataclass
class RunCheckpoint:
    checkpoint_id: str
    checkpoint_type: str
    case_id: int | None
    mode: str
    round: int
    state: ConsultState
    history: list[dict]
    conversation_history: list[dict]
    question_turns: int
    final_conclusion: str | None = None


def history_to_records(history: list[BaseMessage]) -> list[dict]:
    records: list[dict] = []
    for message in history:
        if isinstance(message, HumanMessage):
            msg_type = "human"
        elif isinstance(message, AIMessage):
            msg_type = "ai"
        else:
            msg_type = message.type
        records.append({"type": msg_type, "content": message.content})
    return records


def history_from_records(records: list[dict]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for item in records:
        msg_type = item["type"]
        content = item["content"]
        if msg_type == "human":
            history.append(HumanMessage(content=content))
        elif msg_type == "ai":
            history.append(AIMessage(content=content))
        else:
            raise ValueError(f"不支持恢复的消息类型: {msg_type}")
    return history


def _serialize_checkpoint(checkpoint: RunCheckpoint) -> dict:
    return {
        "version": 1,
        "checkpoint_id": checkpoint.checkpoint_id,
        "checkpoint_type": checkpoint.checkpoint_type,
        "case_id": checkpoint.case_id,
        "mode": checkpoint.mode,
        "round": checkpoint.round,
        "state": checkpoint.state.model_dump(),
        "history": checkpoint.history,
        "conversation_history": checkpoint.conversation_history,
        "question_turns": checkpoint.question_turns,
        "final_conclusion": checkpoint.final_conclusion,
    }


def _deserialize_checkpoint(payload: dict) -> RunCheckpoint:
    checkpoint_type = payload.get("checkpoint_type")
    if checkpoint_type is None:
        legacy_event = payload.get("event", "legacy_event")
        checkpoint_type = "final_state" if legacy_event == "final_state" else "legacy_event"
    return RunCheckpoint(
        checkpoint_id=payload["checkpoint_id"],
        checkpoint_type=checkpoint_type,
        case_id=payload.get("case_id"),
        mode=payload["mode"],
        round=payload["round"],
        state=ConsultState.model_validate(payload["state"]),
        history=payload.get("history", []),
        conversation_history=payload.get("conversation_history", []),
        question_turns=payload.get("question_turns", 0),
        final_conclusion=payload.get("final_conclusion"),
    )


@dataclass
class CheckpointStore:
    folder: Path
    reset: bool = True

    def __post_init__(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.folder / "manifest.json"
        self.checkpoint_dir = self.folder / Config.checkpoint_dirname
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(
        self,
        *,
        case_id: int | None,
        mode: str,
        source_run: str | None = None,
        source_checkpoint_id: str | None = None,
    ) -> None:
        payload = {
            "version": 1,
            "run_id": self.folder.name,
            "case_id": case_id,
            "mode": mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_run": source_run,
            "source_checkpoint_id": source_checkpoint_id,
        }
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_round(
        self,
        *,
        round_num: int,
        case_id: int | None,
        mode: str,
        state: ConsultState,
        history: list[BaseMessage],
        conversation_history: list[dict],
        question_turns: int,
    ) -> RunCheckpoint:
        checkpoint = RunCheckpoint(
            checkpoint_id=f"round_{round_num:03d}",
            checkpoint_type="round_end",
            case_id=case_id,
            mode=mode,
            round=round_num,
            state=state.model_copy(deep=True),
            history=history_to_records(history),
            conversation_history=list(conversation_history),
            question_turns=question_turns,
        )
        self._write_checkpoint(checkpoint)
        return checkpoint

    def save_final(
        self,
        *,
        round_num: int,
        case_id: int | None,
        mode: str,
        state: ConsultState,
        history: list[BaseMessage],
        conversation_history: list[dict],
        question_turns: int,
        final_conclusion: str,
    ) -> RunCheckpoint:
        checkpoint = RunCheckpoint(
            checkpoint_id="final_state",
            checkpoint_type="final_state",
            case_id=case_id,
            mode=mode,
            round=round_num,
            state=state.model_copy(deep=True),
            history=history_to_records(history),
            conversation_history=list(conversation_history),
            question_turns=question_turns,
            final_conclusion=final_conclusion,
        )
        self._write_checkpoint(checkpoint)
        return checkpoint

    def list_checkpoints(self) -> list[RunCheckpoint]:
        checkpoints: list[RunCheckpoint] = []
        for path in sorted(self.checkpoint_dir.glob("*.json")):
            checkpoints.append(_deserialize_checkpoint(json.loads(path.read_text(encoding="utf-8"))))
        checkpoints.sort(key=lambda cp: (cp.round, cp.checkpoint_type != "round_end"))
        return checkpoints

    def load_checkpoint(self, checkpoint_id: str | None = None) -> RunCheckpoint:
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            raise ValueError(f"run 目录中没有 checkpoint: {self.folder}")
        if checkpoint_id is None:
            return checkpoints[-1]
        for checkpoint in checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                return checkpoint
        raise ValueError(f"未找到 checkpoint: {checkpoint_id}")

    def checkpoint_summaries(self) -> list[str]:
        summaries: list[str] = []
        for checkpoint in self.list_checkpoints():
            summaries.append(
                f"{checkpoint.checkpoint_id} | type={checkpoint.checkpoint_type} | round={checkpoint.round} | history={len(checkpoint.history)}"
            )
        return summaries

    @staticmethod
    def create(case_id: int | None, mode: str) -> "CheckpointStore":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"case{case_id}" if case_id is not None else mode
        folder = Config.project_root / "run" / f"{timestamp}_{label}"
        store = CheckpointStore(folder=folder, reset=True)
        store.write_manifest(case_id=case_id, mode=mode)
        return store

    @staticmethod
    def create_resume(
        *,
        case_id: int | None,
        mode: str,
        source_run: str,
        source_checkpoint_id: str,
    ) -> "CheckpointStore":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"case{case_id}" if case_id is not None else mode
        folder = Config.project_root / "run" / f"{timestamp}_{label}_resume"
        store = CheckpointStore(folder=folder, reset=True)
        store.write_manifest(
            case_id=case_id,
            mode=mode,
            source_run=source_run,
            source_checkpoint_id=source_checkpoint_id,
        )
        return store

    @staticmethod
    def open_existing(folder: Path) -> "CheckpointStore":
        if not folder.exists():
            raise ValueError(f"run 目录不存在: {folder}")
        return CheckpointStore(folder=folder, reset=False)

    def _write_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.json"
        path.write_text(
            json.dumps(_serialize_checkpoint(checkpoint), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )