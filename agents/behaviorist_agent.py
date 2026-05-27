from __future__ import annotations

import logging
from typing import Iterator

import Config
from langsmith import traceable
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

_ASSESSMENT_COMPLETE = "[ASSESSMENT_COMPLETE]"
_FORCE_CONCLUDE_MSG = (
    "你已经进行了足够多轮的询问，请现在给出你的完整行为分析和建议，不必再继续提问。"
)


class BehavioristAgent:
    """模拟猫行为学专家，通过提问收集信息，时机成熟时给出深度思考建议。"""

    def __init__(self, llm_strong: LLMClient, llm_think: LLMClient, memory: str = "") -> None:
        self._llm_strong = llm_strong  # 问答轮次
        self._llm_think = llm_think    # 最终建议
        self._memory = memory          # 历史记忆（可为空）
        self._conversation_history: list[dict] = []
        self._round = 0
        self._done = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        prompt = (
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

        if self._memory:
            prompt += (
                "\n\n【相关历史记忆】\n"
                f"{self._memory}\n"
                "以上是你对该主人的历史了解，可作为背景参考。"
            )

        return prompt

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._build_system_prompt()}] + list(
            self._conversation_history
        )

    def _stream_buffered(self, llm: LLMClient, messages: list[dict]) -> tuple[Iterator[str], str]:
        """先完整收集响应，再作为迭代器返回，同时返回完整文本供内部检查。

        对于强模型（问答阶段），需要先缓冲以检测 [ASSESSMENT_COMPLETE]，
        然后再决定是否切换到思考模型。
        """
        chunks: list[str] = []
        for chunk in llm.stream_chat(messages):
            chunks.append(chunk)
        full_text = "".join(chunks)
        return iter(chunks), full_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @traceable(name="behaviorist_ask", run_type="chain")
    def ask(self, user_message: str) -> tuple[Iterator[str], bool, bool]:
        """处理猫主人的消息。

        返回：(token_stream, is_done, is_think)
        - token_stream: 流式文本迭代器
        - is_done: 是否为最终回答（对话结束）
        - is_think: 是否使用了思考模型（用于 main.py 展示思考过程）
        """
        self._round += 1

        # ---- 强制结束（超出最大轮数） ----
        if self._round > Config.max_conversation_rounds and not self._done:
            logger.info("已达到最大轮数 %d，强制使用思考模型给出最终建议。", Config.max_conversation_rounds)
            self._conversation_history.append({"role": "user", "content": user_message})
            self._conversation_history.append({"role": "user", "content": _FORCE_CONCLUDE_MSG})

            messages = self._build_messages()
            stream = self._stream_think_final(messages)
            self._done = True
            return stream, True, True

        # ---- 正常轮次：先用强模型缓冲响应 ----
        self._conversation_history.append({"role": "user", "content": user_message})
        messages = self._build_messages()

        _, full_text = self._stream_buffered(self._llm_strong, messages)

        if _ASSESSMENT_COMPLETE in full_text:
            # 强模型认为信息已足够 → 丢弃该响应，改用思考模型输出最终建议
            logger.info("检测到 %s，切换到思考模型输出最终建议。", _ASSESSMENT_COMPLETE)
            final_messages = messages + [
                {
                    "role": "assistant",
                    "content": "（信息已足够，准备给出完整分析）",
                },
                {
                    "role": "user",
                    "content": "请现在给出你完整的行为分析和具体建议。",
                },
            ]
            stream = self._stream_think_final(final_messages)
            self._done = True
            return stream, True, True
        else:
            # 正常问答轮次，直接流式返回强模型响应
            self._conversation_history.append({"role": "assistant", "content": full_text})
            return iter(full_text), False, False

    @traceable(name="behaviorist_think_final", run_type="llm")
    def _stream_think_final(self, messages: list[dict]) -> Iterator[str]:
        """使用思考模型流式输出，并在结束后更新历史。"""
        accumulated: list[str] = []

        def _gen() -> Iterator[str]:
            for chunk in self._llm_think.stream_chat(messages):
                accumulated.append(chunk)
                yield chunk
            # 流消费完毕后更新历史
            self._conversation_history.append(
                {"role": "assistant", "content": "".join(accumulated)}
            )

        return _gen()

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def round_count(self) -> int:
        return self._round

    @property
    def final_conclusion(self) -> str | None:
        """Return the final conclusion text, or None if consultation not done."""
        if not self._done or not self._conversation_history:
            return None
        for msg in reversed(self._conversation_history):
            if msg["role"] == "assistant":
                return msg["content"]
        return None

    @property
    def conversation_history(self) -> list[dict]:
        """Return a copy of the full conversation history."""
        return list(self._conversation_history)
