from __future__ import annotations

from evaluation.metrics.base import LLMJudge
from evaluation.metrics.utils import format_history

UHA_SYSTEM_PROMPT_UNCERTAINTY = """你是一位专业的猫行为学评估专家。你需要判断顾问是否正确处理了不确定性。

当前案例具有不确定性（uncertainty=true）。正确的处理方式：
- 承认信息不足
- 保留多个可能方向
- 避免过度确定性
- 不应给出"唯一确诊"式的结论

错误的处理方式：
- 过度确定单一结论
- 忽略其他可能性
- 以确信语气排除其他方向

只输出"正确"或"错误"，不要输出其他内容。"""

UHA_SYSTEM_PROMPT_CERTAINTY = """你是一位专业的猫行为学评估专家。你需要判断顾问是否正确处理了确定性。

当前案例不具有重大不确定性（uncertainty=false）。正确的处理方式：
- 明确聚焦核心问题
- 给出清晰的诊断方向
- 可以提及次要因素，但应体现明显主次关系
- 不应大篇幅讨论多个竞争性解释

错误的处理方式：
- 大篇幅讨论多个竞争性解释而不分主次
- 过度犹豫，该下结论时不下
- 核心结论模糊

只输出"正确"或"错误"，不要输出其他内容。"""


def compute_uha(
    judge: LLMJudge,
    conversation_history: list[dict],
    final_conclusion: str,
    uncertainty: bool,
) -> bool:
    """Uncertainty Handling Accuracy. Returns True if correctly handled."""
    history_text = format_history(conversation_history)
    system_prompt = UHA_SYSTEM_PROMPT_UNCERTAINTY if uncertainty else UHA_SYSTEM_PROMPT_CERTAINTY
    user_prompt = (
        f"【对话历史】\n{history_text}\n\n"
        f"【最终建议】\n{final_conclusion}\n\n"
        "请判断处理是否正确："
    )
    response = judge.judge(system_prompt, user_prompt)
    return "正确" in response
