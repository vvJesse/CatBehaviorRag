from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import Config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    case_id: int
    initial_user_message: str
    user_setting: str
    reference_answer: str
    uncertainty: bool
    required_directions: list[str]


def load_benchmark(path: Optional[Path] = None) -> list[BenchmarkCase]:
    """加载并解析 benchmark v2 JSON 文件，返回 BenchmarkCase 列表。"""
    p = path or Config.benchmark_path
    logger.info("加载 benchmark 文件：%s", p)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    cases: list[BenchmarkCase] = []
    for item in data:
        cases.append(
            BenchmarkCase(
                case_id=item["id"],
                initial_user_message=item["initial_user_message"],
                user_setting=item["user_setting"],
                reference_answer=item["reference_answer"],
                uncertainty=item["uncertainty"],
                required_directions=item.get("required_directions", []),
            )
        )
    logger.info("共加载 %d 个 benchmark case", len(cases))
    return cases


def select_case(cases: list[BenchmarkCase], case_id: Optional[int] = None) -> BenchmarkCase:
    """按 case_id 查找，若为 None 则返回第一个。"""
    if case_id is None:
        return cases[0]
    for case in cases:
        if case.case_id == case_id:
            return case
    valid = [c.case_id for c in cases]
    raise ValueError(f"未找到 case_id='{case_id}'，可用：{valid}")
