from __future__ import annotations

import logging

from utils.benchmark_loader import BenchmarkCase
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


class UserAgent:
    """模拟猫主人，根据 BenchmarkCase 的 user_state 逐步披露信息。"""

    def __init__(self, case: BenchmarkCase, llm: LLMClient) -> None:
        self.case = case
        self.llm = llm
        self._conversation_history: list[dict] = []
        self._revealed_indices: set[int] = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, extra_facts: list[str] | None = None) -> str:
        us = self.case.user_state

        known_lines = "\n".join(f"- {f}" for f in us.initially_known)
        belief_lines = "\n".join(f"- {b}" for b in us.user_beliefs) if us.user_beliefs else "（无）"

        prompt = (
            "你是一位普通的猫主人，正在向猫行为学专家咨询问题。\n\n"
            "【你目前确定知道的情况】\n"
            f"{known_lines}\n\n"
            "【你对猫行为的固有认知（可能不准确，这是你的真实想法）】\n"
            f"{belief_lines}\n\n"
            "【注意事项】\n"
            "- 只根据你实际知道的信息回答，不要主动透露你没有提过的细节。\n"
            "- 回答要自然，像普通人说话，不要过于正式或专业。\n"
            "- 如果专家问到你不了解的事情，可以说不清楚或不确定。\n"
            "- 用中文回答。"
        )

        if extra_facts:
            facts_text = "；".join(extra_facts)
            prompt += f"\n\n【你刚刚想起的额外情况】\n{facts_text}\n如果对话内容与这些相关，请自然地提及。"

        return prompt

    def _check_and_reveal_facts(self, behaviorist_question: str) -> list[str]:
        """检查行为专家的问题是否触发了尚未披露的可发现事实。"""
        newly_revealed: list[str] = []
        us = self.case.user_state

        for i, df in enumerate(us.discoverable_facts):
            if i in self._revealed_indices:
                continue

            topics = "、".join(df.revealed_when_asked_about)
            classifier_messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一个判断助手。请判断下面的问题是否涉及给定话题中的任意一个。"
                        "只回答\"是\"或\"否\"，不需要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"行为专家的问题：\"{behaviorist_question}\"\n\n"
                        f"话题标签：{topics}\n\n"
                        "这个问题是否涉及上述话题中的任意一个？只回答\"是\"或\"否\"。"
                    ),
                },
            ]

            try:
                answer = self.llm.chat(classifier_messages).strip()
                if "是" in answer:
                    self._revealed_indices.add(i)
                    newly_revealed.append(df.fact)
                    logger.debug("披露新事实：%s", df.fact)
            except Exception as e:
                logger.warning("事实判断调用失败，跳过：%s", e)

        return newly_revealed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def respond(self, behaviorist_message: str) -> str:
        """根据行为专家的消息生成猫主人的回复。"""
        newly_revealed = self._check_and_reveal_facts(behaviorist_message)

        system_prompt = self._build_system_prompt(extra_facts=newly_revealed if newly_revealed else None)

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._conversation_history)
        messages.append({"role": "user", "content": behaviorist_message})

        response = self.llm.chat(messages)

        self._conversation_history.append({"role": "user", "content": behaviorist_message})
        self._conversation_history.append({"role": "assistant", "content": response})

        return response
