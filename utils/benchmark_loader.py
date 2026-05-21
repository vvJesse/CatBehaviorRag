from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import Config

logger = logging.getLogger(__name__)


@dataclass
class DiscoverableFact:
    fact: str
    revealed_when_asked_about: list[str]


@dataclass
class UserState:
    initially_known: list[str]
    discoverable_facts: list[DiscoverableFact]
    user_beliefs: list[str]


@dataclass
class GroundTruth:
    primary_issue: str
    critical_facts: list[str]
    accepted_conclusions: list[str]
    rejected_conclusions: list[str]


@dataclass
class BenchmarkCase:
    case_id: str
    initial_user_message: str
    user_state: UserState
    ground_truth: GroundTruth
    reference_solution: str


def load_benchmark(path: Optional[Path] = None) -> list[BenchmarkCase]:
    """加载并解析 benchmark JSON 文件，返回 BenchmarkCase 列表。"""
    p = path or Config.benchmark_path
    logger.info("加载 benchmark 文件：%s", p)
    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    cases: list[BenchmarkCase] = []
    for item in data:
        us = item["user_state"]
        gt = item["ground_truth"]
        cases.append(
            BenchmarkCase(
                case_id=item["case_id"],
                initial_user_message=item["initial_user_message"],
                user_state=UserState(
                    initially_known=us["initially_known"],
                    discoverable_facts=[
                        DiscoverableFact(
                            fact=df["fact"],
                            revealed_when_asked_about=df["revealed_when_asked_about"],
                        )
                        for df in us.get("discoverable_facts", [])
                    ],
                    user_beliefs=us.get("user_beliefs", []),
                ),
                ground_truth=GroundTruth(
                    primary_issue=gt["primary_issue"],
                    critical_facts=gt["critical_facts"],
                    accepted_conclusions=gt["accepted_conclusions"],
                    rejected_conclusions=gt["rejected_conclusions"],
                ),
                reference_solution=item["reference_solution"],
            )
        )
    logger.info("共加载 %d 个 benchmark case", len(cases))
    return cases


def select_case(cases: list[BenchmarkCase], case_id: Optional[str] = None) -> BenchmarkCase:
    """按 case_id 查找，若为 None 则返回第一个。"""
    if case_id is None:
        return cases[0]
    for case in cases:
        if case.case_id == case_id:
            return case
    valid = [c.case_id for c in cases]
    raise ValueError(f"未找到 case_id='{case_id}'，可用：{valid}")
