from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from main import ConsultationResult, run_consultant_loop
from utils.benchmark_loader import load_benchmark
from utils.llm_client import LLMClient

from evaluation.metrics.aqt import compute_aqt
from evaluation.metrics.as_ import compute_actionability, compute_actionability_normalized
from evaluation.metrics.base import LLMJudge
from evaluation.metrics.dcs import compute_dcs
from evaluation.metrics.ufs import compute_ufs, compute_ufs_normalized
from evaluation.metrics.uha import compute_uha

logger = logging.getLogger(__name__)


@dataclass
class CaseEvaluation:
    case_id: int
    question_turns: int
    ufs_raw: int
    ufs_norm: float
    dcs: float
    dcs_details: list[bool]
    as_raw: int
    as_norm: float
    uha_correct: bool


@dataclass
class BatchEvaluationResult:
    total_cases: int
    aqt: float
    mean_ufs_norm: float
    mean_dcs: float
    mean_as_norm: float
    uha: float
    case_evaluations: list[CaseEvaluation]


def run_batch_evaluation(
    llm_fast: LLMClient,
    llm_strong: LLMClient,
    llm_think: LLMClient,
    memory: str = "",
) -> BatchEvaluationResult:
    """运行所有 benchmark case 并计算评估指标。"""
    cases = load_benchmark()
    judge = LLMJudge(llm_fast)

    results: list[tuple] = []  # (case, ConsultationResult)

    for case in cases:
        logger.info("运行 case %s ...", case.case_id)
        consultation = run_consultant_loop(
            case, llm_strong, llm_think, memory=memory, silent=True
        )
        results.append((case, consultation))

    # 计算各 case 的指标
    case_evals: list[CaseEvaluation] = []
    consultations: list[ConsultationResult] = []
    for case, consultation in results:
        ufs_raw = compute_ufs(judge, consultation.conversation_history, consultation.final_conclusion)
        dcs_score, dcs_details = compute_dcs(
            judge, consultation.conversation_history, consultation.final_conclusion,
            case.required_directions,
        )
        as_raw = compute_actionability(judge, consultation.conversation_history, consultation.final_conclusion)
        uha_correct = compute_uha(
            judge, consultation.conversation_history, consultation.final_conclusion,
            case.uncertainty,
        )

        case_evals.append(CaseEvaluation(
            case_id=case.case_id,
            question_turns=consultation.question_turns,
            ufs_raw=ufs_raw,
            ufs_norm=(ufs_raw - 1) / 4,
            dcs=dcs_score,
            dcs_details=dcs_details,
            as_raw=as_raw,
            as_norm=(as_raw - 1) / 4,
            uha_correct=uha_correct,
        ))
        consultations.append(consultation)

    # 汇总
    n = len(case_evals)
    result = BatchEvaluationResult(
        total_cases=n,
        aqt=compute_aqt([ConsultationResult(
            case_id=e.case_id,
            conversation_history=[],
            question_turns=e.question_turns,
            final_conclusion="",
        ) for e in case_evals]),
        mean_ufs_norm=sum(e.ufs_norm for e in case_evals) / n if n else 0.0,
        mean_dcs=sum(e.dcs for e in case_evals) / n if n else 0.0,
        mean_as_norm=sum(e.as_norm for e in case_evals) / n if n else 0.0,
        uha=sum(1 for e in case_evals if e.uha_correct) / n if n else 0.0,
        case_evaluations=case_evals,
    )

    # 保存结果
    _save_result(result, case_evals, consultations)
    return result


def _save_result(
    result: BatchEvaluationResult,
    case_evals: list[CaseEvaluation],
    consultations: list[ConsultationResult],
) -> None:
    """保存评测结果：metrics 汇总 + 每个 case 的对话记录和 conclusion。"""
    result_dir = Path(__file__).parent / "result"
    result_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 指标汇总
    summary_path = result_dir / f"eval_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    logger.info("评测结果已保存至 %s", summary_path)

    # 2. 每个 case 的对话记录 + conclusion + 该 case 的各项指标
    eval_by_case = {e.case_id: e for e in case_evals}
    detail = []
    for consultation in consultations:
        e = eval_by_case[consultation.case_id]
        detail.append({
            "case_id": consultation.case_id,
            "question_turns": consultation.question_turns,
            "metrics": {
                "ufs_raw": e.ufs_raw,
                "ufs_norm": e.ufs_norm,
                "dcs": e.dcs,
                "dcs_details": e.dcs_details,
                "as_raw": e.as_raw,
                "as_norm": e.as_norm,
                "uha_correct": e.uha_correct,
            },
            "final_conclusion": consultation.final_conclusion,
            "conversation_history": consultation.conversation_history,
        })

    detail_path = result_dir / f"eval_detail_{timestamp}.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)
    logger.info("对话详情已保存至 %s", detail_path)
