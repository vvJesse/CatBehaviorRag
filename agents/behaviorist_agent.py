from __future__ import annotations

import logging

import Config
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

_ASSESSMENT_COMPLETE = "[ASSESSMENT_COMPLETE]"
_FORCE_CONCLUDE_MSG = (
    "你已经进行了足够多轮的询问，请现在给出你的完整行为分析和建议，不必再继续提问。"
)


class BehavioristAgent:
    """模拟猫行为学专家，通过提问收集信息，时机成熟时给出建议。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._conversation_history: list[dict] = []
        self._round = 0
        self._done = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return (
            "你是一位专业的猫行为学顾问，具有丰富的猫行为诊断和干预经验。\n\n"
            "【你的任务】\n"
            "通过与猫主人对话收集足够的信息，然后给出专业的行为分析和具体建议。\n\n"
            "【对话策略】\n"
            "- 每次提问不超过 2 个问题，循序渐进地了解情况。\n"
            "- 先了解基本情况（症状、时间线、环境），再深入询问细节。\n"
            "- 注意识别主人可能存在的认知误区，并在适当时候温和地纠正。\n"
            "- 当你认为已收集到足够信息，可以做出判断时：\n"
            f"  在回复末尾加上 {_ASSESSMENT_COMPLETE}，然后给出完整的分析和建议。\n\n"
            "【输出要求】\n"
            "- 分析和建议要具体、可操作。\n"
            "- 用中文回复。"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, user_message: str) -> tuple[str, bool]:
        """
        处理猫主人的消息，返回 (回复文本, 是否结束对话)。
        """
        self._round += 1

        # 10 轮强制结束
        if self._round > Config.max_conversation_rounds and not self._done:
            logger.info("已达到最大轮数 %d，强制给出最终建议。", Config.max_conversation_rounds)
            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "user", "content": _FORCE_CONCLUDE_MSG})

            messages = [{"role": "system", "content": self._build_system_prompt()}]
            messages.extend(self._conversation_history)

            response = self.llm.chat(messages)
            response = response.replace(_ASSESSMENT_COMPLETE, "").strip()
            self._conversation_history.append({"role": "assistant", "content": response})
            self._done = True
            return response, True

        # 正常轮次
        self._conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(self._conversation_history)

        response = self.llm.chat(messages)

        is_done = _ASSESSMENT_COMPLETE in response
        if is_done:
            response = response.replace(_ASSESSMENT_COMPLETE, "").strip()
            self._done = True

        self._conversation_history.append({"role": "assistant", "content": response})
        return response, is_done

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def round_count(self) -> int:
        return self._round
