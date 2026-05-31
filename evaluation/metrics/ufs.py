from __future__ import annotations

from evaluation.metrics.base import LLMJudge
from evaluation.metrics.utils import format_history, parse_score

UFS_SYSTEM_PROMPT = """你是一位专业的沟通体验评估专家。

你的任务是评估：
一位猫行为咨询 agent 的最终回复，
对于普通猫主人来说，
是否自然、友好、易理解，
并且真正具有“人与人沟通”的感觉。

你评估的是“沟通体验”，
不是知识正确性。

即使内容专业、完整、正确，
如果表达方式像AI报告、信息堆积或让用户读起来很累，
也不应给高分。

请重点关注以下问题：

1. AI lecture感
是否存在明显“AI整理报告感”：
- 机械式编号、分层、Checklist堆叠
- 像分析报告或科普文章
- 缺少自然过渡
- 缺少互动感
- 缺少情绪承接
- 整体不像真实咨询交流

2. 信息负担
是否一次性提供过多信息：
- 信息密度过高
- 步骤堆积
- 用户难以抓住重点
- 普通用户读起来容易疲惫

即使内容正确，
如果阅读负担明显过高，
也应降低评分。

3. 术语与表达
是否：
- 大量使用专业术语
- 表达生硬
- 缺少自然解释
- 像在“上课”

允许必要专业内容，
但应自然、易理解。

4. 情绪体验
是否：
- 存在说教感或居高临下感
- 制造明显焦虑、压迫感
- 过度强调严重后果
- 缺少对用户情绪的自然回应

5. 自然交流感
高分回答通常：
- 像真实咨询
- 有自然的人际沟通感
- 不只是“信息输出”
- 不会让用户感觉被AI教育

重要：
以下情况即使内容专业完整，也不应给5分：
- 明显AI报告感
- 机械式结构堆积
- 信息过载
- 缺少交流感
- 缺少情绪承接
- 像“给方案”而不是“在沟通”

评分标准：

5：
自然、友好、易理解，
明显具有人际沟通感，
专业但不生硬，
信息量合适，
读起来像真实咨询。

4：
整体良好，
有少量AI感或表达偏硬，
但总体仍较自然。

3：
明显存在AI感、
信息堆积、
机械式结构、
轻度说教，
交流感一般。

2：
阅读负担明显较高，
术语堆积，
明显像AI报告或lecture，
缺少自然沟通感。

1：
严重不友好、
压迫、
生硬或难以理解，
明显不像人与人交流。

只输出一个1-5整数。
不要输出任何解释、分析或其他内容。"""


def compute_ufs(judge: LLMJudge, conversation_history: list[dict], final_conclusion: str) -> int:
    """User-Friendliness Score (raw 1-5)."""
    history_text = format_history(conversation_history)
    user_prompt = (
        f"【对话历史】\n{history_text}\n\n"
        f"【最终建议】\n{final_conclusion}\n\n"
        "请给出1-5的评分："
    )
    response = judge.judge(UFS_SYSTEM_PROMPT, user_prompt)
    return parse_score(response, min_val=1, max_val=5)


def compute_ufs_normalized(judge: LLMJudge, conversation_history: list[dict], final_conclusion: str) -> float:
    """User-Friendliness Score (normalized 0-1)."""
    raw = compute_ufs(judge, conversation_history, final_conclusion)
    return (raw - 1) / 4
