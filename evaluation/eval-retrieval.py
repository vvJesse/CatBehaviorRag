import os
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import ContextRelevance

from VectorRetriever import VectorRetriever

try:
    from langchain_community.chat_models import ChatTongyi
except ImportError:
    from langchain_community.chat_models.tongyi import ChatTongyi

import Config
from datetime import datetime



dashscope_api_key = os.getenv("DASHSCOPE_API_KEY") or Config.dashscope_api_key
if not dashscope_api_key:
    raise ValueError("未配置 DASHSCOPE_API_KEY，请先在环境变量或 Config.py 中设置。")

evaluation_dir = Path(__file__).resolve().parent

class RagasRetrievalEvaluator(object):
    def __init__(
            self,
            dataset: EvaluationDataset,
    ):
        self.dataset = dataset
        evaluator_model = os.getenv("EVAL_LLM_MODEL", "qwen-plus")
        base_llm = ChatTongyi(
            model=evaluator_model,
            api_key=dashscope_api_key,
        )
        evaluator_llm = LangchainLLMWrapper(base_llm)
        self.metrics = [ContextRelevance(llm=evaluator_llm)]

    def run(self):
        result = evaluate(
            dataset=self.dataset,
            metrics=self.metrics,
        )
        return result


def setup_evaluation_config(
        embedding_provider: str="local",
        eval_model_name: str="qwen-plus",
        eval_dataset: str="syn-clear",
):
    Config.embedding_provider = embedding_provider
    Config.eval_model = eval_model_name
    Config.eval_dataset = eval_dataset

def resolve_data_path(data_name: str):
    if data_name == "syn-clear":
        return evaluation_dir / "synthesis" / "clear.json"
    elif data_name == "syn-ambiguous":
        return evaluation_dir / "synthesis" / "ambiguous.json"
    elif data_name == "real-clear":
        return evaluation_dir / "real" / "clear.json"
    elif data_name == "real-ambiguous":
        return evaluation_dir / "real" / "ambiguous.json"
    else:
        raise ValueError(f"Unknown data name: {data_name}")

def save_result(result_df, data, retrieval_results):
    filename = f"eval-{Config.eval_dataset}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    result_records = result_df.to_dict(orient="records")
    output_records = [
        {
            **record,
            "question": item.get("question"),
            "retrieved_chunks": [
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                }
                for doc, score in retrieved_docs
            ],
        }
        for item, retrieved_docs, record in zip(data, retrieval_results, result_records)
    ]
    os.makedirs(evaluation_dir / "result", exist_ok=True)
    (evaluation_dir / "result" / filename).write_text(
        json.dumps(output_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Result saved to {evaluation_dir / 'result' / filename}")


if __name__ == "__main__":
    setup_evaluation_config(
        eval_model_name="qwen-plus",
        embedding_provider="local",
        eval_dataset="syn-clear"
    )

    with (resolve_data_path(Config.eval_dataset)).open("r", encoding="utf-8") as f:
        data = json.load(f)
    Config.embedding_provider = "local"
    retriever = VectorRetriever()
    logger.info("Building vector store...")
    retriever.build_vector_store(force_rebuild=False)
    logger.info("Vector store built. Starting document retrieval...")
    retrieval_results = [
        retriever.search_with_scores(item.get("question"), k=3)
        for item in data
    ]
    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=item.get("question"),
                retrieved_contexts=[
                    doc[0].page_content
                    for doc in retrieved_docs
                ],
            )
            for item, retrieved_docs in zip(data, retrieval_results)
        ]
    )
    evaluator = RagasRetrievalEvaluator(dataset)
    logger.info("Starting evaluation...")
    result = evaluator.run()
    result_df = result.to_pandas()
    print(result_df)
    # 把结果转换为json
    save_result(result_df, data, retrieval_results)


