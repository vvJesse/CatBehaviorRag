from __future__ import annotations

from utils.llm_client import LLMClient


class LLMJudge:
    """Quick-mode LLM judge，使用快速模型进行非流式评判。"""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._llm.chat(messages).strip()
