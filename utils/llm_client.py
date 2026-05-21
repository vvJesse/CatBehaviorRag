from __future__ import annotations

import logging
from typing import Iterator

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import Config

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    pass


class LLMClient:
    """DashScope LLM 的薄封装，使用 ChatTongyi。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ) -> None:
        kwargs: dict = {"model": model, "api_key": api_key, "temperature": temperature}
        if enable_thinking:
            kwargs["model_kwargs"] = {"enable_thinking": True}
        self._llm = ChatTongyi(**kwargs)
        self._model = model
        self._enable_thinking = enable_thinking

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def build_from_config(cls) -> "LLMClient":
        """向后兼容的工厂方法，返回 strong 模型。"""
        if not Config.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置，请在环境变量中配置或修改 Config.py。"
            )
        return cls(model=Config.consultation_model, api_key=Config.dashscope_api_key)

    @classmethod
    def build_for_role(cls, role: str) -> "LLMClient":
        """按角色构建对应模型的客户端。role: 'fast' | 'strong' | 'think'"""
        if not Config.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置，请在环境变量中配置或修改 Config.py。"
            )
        model = Config.resolve_model(role)
        enable_thinking = role == "think" and Config.model_think_enable_thinking
        logger.debug("构建 LLMClient: role=%s model=%s think=%s", role, model, enable_thinking)
        return cls(
            model=model,
            api_key=Config.dashscope_api_key,
            enable_thinking=enable_thinking,
        )

    # ------------------------------------------------------------------
    # Internal: message conversion
    # ------------------------------------------------------------------

    def _to_lc_messages(self, messages: list[dict]):
        lc = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                lc.append(SystemMessage(content=content))
            elif role == "user":
                lc.append(HumanMessage(content=content))
            elif role == "assistant":
                lc.append(AIMessage(content=content))
            else:
                raise LLMClientError(f"未知的 role: {role}")
        return lc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict]) -> str:
        """非流式调用，返回完整回复字符串（用于分类等轻量任务）。"""
        try:
            response = self._llm.invoke(self._to_lc_messages(messages))
            return response.content
        except Exception as e:
            logger.error("LLM 调用失败：%s", e)
            raise LLMClientError(f"LLM 调用失败：{e}") from e

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """流式调用，逐块 yield 文本。

        对于思考模型，思考过程以 <think>...</think> 包裹在内容流中一起输出，
        调用方负责格式化展示。
        """
        try:
            for chunk in self._llm.stream(self._to_lc_messages(messages)):
                text = chunk.content
                if text:
                    yield text
        except Exception as e:
            logger.error("LLM 流式调用失败：%s", e)
            raise LLMClientError(f"LLM 流式调用失败：{e}") from e
