from __future__ import annotations

from evaluation.metrics.base import LLMJudge
from evaluation.metrics.utils import format_history, parse_score

AS_SYSTEM_PROMPT = """你是一位专业的行动建议评估专家。你的任务是评估猫行为顾问给出的最终建议是否真正"可以执行"。

请关注以下方面：
- 是否给出了具体可执行的行动步骤
- 是否能真正落地（普通猫主人能否实际做到）
- 是否避免了空泛的安慰性语言
- 是否有明确的优先级（先做什么后做什么）
- 是否让普通用户立刻知道下一步该做什么

评分标准：
5：高度具体，可直接执行
4：大部分可执行，少量抽象
3：方向正确，但缺少落地细节
2：较空泛，难以执行
1：几乎不可执行

只输出一个1-5的整数评分，不要输出其他内容。"""


def compute_actionability(judge: LLMJudge, conversation_history: list[dict], final_conclusion: str) -> int:
    """Actionability Score (raw 1-5)."""
    history_text = format_history(conversation_history)
    user_prompt = (
        f"【对话历史】\n{history_text}\n\n"
        f"【最终建议】\n{final_conclusion}\n\n"
        "请给出1-5的评分："
    )
    response = judge.judge(AS_SYSTEM_PROMPT, user_prompt)
    return parse_score(response, min_val=1, max_val=5)


def compute_actionability_normalized(judge: LLMJudge, conversation_history: list[dict], final_conclusion: str) -> float:
    """Actionability Score (normalized 0-1)."""
    raw = compute_actionability(judge, conversation_history, final_conclusion)
    return (raw - 1) / 4
