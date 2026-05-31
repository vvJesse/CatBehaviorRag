from __future__ import annotations

from evaluation.metrics.base import LLMJudge
from evaluation.metrics.utils import format_history, parse_direction_coverage

DCS_SYSTEM_PROMPT = """你是一位专业的猫行为学评估专家。

你的任务是判断：
咨询顾问的最终建议，
是否真正覆盖了指定的关键方向。

对于每个方向，
判断顾问是否：

- 明确识别了该方向，或
- 进行了普通读者可以合理理解的明确暗示

注意：

- 不要求完全相同措辞
- 不要求完全一致推理路径
- 允许不同表达方式
- 不要求逐字匹配

但：

- 不要仅因提到相关关键词、
  场景或表面概念，
  就判定为覆盖
- 需要判断是否真正涉及该方向背后的：
  - 核心风险
  - 核心因果
  - 核心行为学含义
- 如果只是弱相关、
  表层提及、
  或语义关联较弱，
  应判定为“未覆盖”

请对每个方向输出：
“覆盖”或“未覆盖”

格式：
方向1: 覆盖
方向2: 未覆盖
..."""


def compute_dcs(
    judge: LLMJudge,
    conversation_history: list[dict],
    final_conclusion: str,
    required_directions: list[str],
) -> tuple[float, list[bool]]:
    """Direction Coverage Score. Returns (score, per-direction coverage list)."""
    if not required_directions:
        return 1.0, []

    history_text = format_history(conversation_history)
    directions_text = "\n".join(f"方向{i+1}: {d}" for i, d in enumerate(required_directions))
    user_prompt = (
        f"【对话历史】\n{history_text}\n\n"
        f"【最终建议】\n{final_conclusion}\n\n"
        f"【需要覆盖的方向】\n{directions_text}\n\n"
        "请逐一判断每个方向是否被覆盖："
    )
    response = judge.judge(DCS_SYSTEM_PROMPT, user_prompt)
    covered = parse_direction_coverage(response, len(required_directions))
    score = sum(covered) / len(required_directions)
    return score, covered
