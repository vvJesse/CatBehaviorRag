from __future__ import annotations

import logging
from typing import Optional

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import Config

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class LLMClient:
    """DashScope LLM 的薄封装，使用 ChatTongyi。"""

    def __init__(self, model: str, api_key: str, temperature: float = 0.7) -> None:
        self._llm = ChatTongyi(model=model, api_key=api_key, temperature=temperature)

    @classmethod
    def build_from_config(cls) -> "LLMClient":
        if not Config.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置，请在环境变量中配置或修改 Config.py。"
            )
        return cls(model=Config.consultation_model, api_key=Config.dashscope_api_key)

    def chat(self, messages: list[dict]) -> str:
        """
        发送对话请求并返回模型回复文本。

        messages 格式：[{"role": "system"/"user"/"assistant", "content": "..."}]
        """
        lc_messages = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                raise LLMClientError(f"未知的 role: {role}")

        try:
            response = self._llm.invoke(lc_messages)
            return response.content
        except Exception as e:
            logger.error("LLM 调用失败：%s", e)
            raise LLMClientError(f"LLM 调用失败：{e}") from e
