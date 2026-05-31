from __future__ import annotations

import logging
from typing import Iterator

from langsmith import traceable
from utils.benchmark_loader import BenchmarkCase
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class UserAgent:
    """模拟猫主人，根据 BenchmarkCase 的 user_setting 回答咨询 agent 的提问。"""

    def __init__(self, case: BenchmarkCase, llm_strong: LLMClient) -> None:
        self.case = case
        self._llm_strong = llm_strong
        self._conversation_history: list[dict] = []
        self._system_prompt = self._build_system_prompt()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "你是一位普通的猫主人，正在向猫行为学专家咨询问题。\n\n"
            "以下是你对自家猫情况的全部认知：\n"
            "---\n"
            f"{self.case.user_setting}\n"
            "---\n\n"
            "【回答原则】\n"
            "- 先判断你的认知是否足够回答专家的问题\n"
            "- 如果足够回答，则仅仅回答专家的问题，不做额外的发挥。\n"
            "- 如果不足够回答，则说不知道，或者这个问题无法回答。\n"
            "- 用中文回答。"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="user_agent_respond", run_type="chain")
    def respond(self, behaviorist_message: str) -> Iterator[str]:
        """根据行为专家的消息流式生成猫主人的回复。

        调用方需消费完整个迭代器；内部会在迭代结束后自动更新对话历史。
        """
        messages: list[dict] = [{"role": "system", "content": self._system_prompt}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": behaviorist_message})

        accumulated: list[str] = []

        for chunk in self._llm_strong.stream_chat(messages):
            accumulated.append(chunk)
            yield chunk

        full_response = "".join(accumulated)
        self._conversation_history.append({"role": "user", "content": behaviorist_message})
        self._conversation_history.append({"role": "assistant", "content": full_response})
