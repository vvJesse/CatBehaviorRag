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
            "以下是你对自家猫情况的全部认知（包含你确定的、模糊感觉到的、以及你不知道的边界）：\n"
            "---\n"
            f"{self.case.user_setting}\n"
            "---\n\n"
            "【回答原则】\n"
            "- 严格基于上述认知回答。对于「能明确观察到」的部分，你回答时很确定。\n"
            "- 对于「能模糊感觉到但不确定」的部分，只有当专家具体追问时，你才会以不确定的语气提及"
            "（如「好像……」「我不太确定，但……」），不会主动提起。\n"
            "- 对于「无法可靠回答」的部分，你会坦诚说不清楚、没注意过，不会编造信息。\n"
            "- 你的回答会自然流露出「主观倾向」中的看法，即使这些看法可能不专业。\n"
            "- 你绝不会主动提供「不会主动提供隐藏线索」中提到的信息，除非专家用非常具体的问题追问到，"
            "你才可能想起来并提及。\n"
            "- 说话自然口语化，像一个普通猫主人，不要太正式、太专业。\n"
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
