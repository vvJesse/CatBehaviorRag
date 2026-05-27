from __future__ import annotations

import re


def format_history(conversation_history: list[dict]) -> str:
    """Format conversation history for judge prompt."""
    lines = []
    role_map = {"user": "猫主人", "assistant": "行为专家"}
    for msg in conversation_history:
        role = role_map.get(msg["role"], msg["role"])
        lines.append(f"【{role}】{msg['content']}")
    return "\n\n".join(lines)


def parse_score(response: str, min_val: int = 1, max_val: int = 5) -> int:
    """Extract integer score from LLM response."""
    match = re.search(r'\d+', response)
    if match:
        score = int(match.group())
        return max(min_val, min(max_val, score))
    return min_val


def parse_direction_coverage(response: str, n_directions: int) -> list[bool]:
    """Parse per-direction coverage from judge response."""
    results: list[bool] = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if '未覆盖' in line:
            results.append(False)
        elif '覆盖' in line:
            results.append(True)
    # Pad or truncate to match expected count
    while len(results) < n_directions:
        results.append(False)
    return results[:n_directions]
