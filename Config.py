
from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s - %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging(level: str | None = None) -> None:
    """为整个项目配置统一的日志格式。

    在应用入口调用一次即可；后续各模块通过 ``logging.getLogger(__name__)`` 取用。
    """
    logging.basicConfig(
        level=level or log_level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
    )


# 项目根目录，用于拼接其余相对路径配置。
project_root = Path(__file__).resolve().parent

# 文件上传模块支持的文件类型。
support_file_types = ["pdf", "txt"]

# 上传文档转全文后的落盘目录。
full_text_path = project_root / "rag_document_uploader" / "full_data"

# 切分后文档快照导出路径，便于人工查看 page_content 与 metadata。
chunked_documents_json_path = Path(
	os.getenv("CHUNKED_DOCUMENTS_JSON_PATH", str(project_root / "data" / "chunked_documents.json"))
)

# 向量库持久化目录。
vector_store_path = project_root / ".chroma" / "line_documents"

# Chroma 集合名称。
vector_collection_name = "cat_behavior_line_documents"

# Embedding Provider 配置。
# 支持: dashscope / local
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").strip().lower()

# DashScope 向量模型配置。
# provider=dashscope 时生效。如果不想把密钥写死，也可以在环境变量中设置 DASHSCOPE_API_KEY。
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope_embedding_model = "text-embedding-v1"

# 本地向量模型配置（HuggingFace / sentence-transformers）。
# provider=local 时生效；有显卡时可将 LOCAL_EMBEDDING_DEVICE 设为 cuda。
local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
local_embedding_device = os.getenv("LOCAL_EMBEDDING_DEVICE", "cuda")
local_embedding_encode_kwargs = {
	"normalize_embeddings": True,
}

# 单条文档允许直接入库的最大字符数；超过后将触发文本切分。
max_doc_length = 600

# 文本切分器配置。
text_splitter_chunk_size = 500
text_splitter_chunk_overlap = 50
text_splitter_separators = [
	"\n\n",
	"\n",
	"。",
	"！",
	"？",
	"；",
	"，",
	" ",
	"",
]

eval_model = "qwen-plus"
eval_dataset = "syn-clear"

# --- Consultation System ---
consultation_model = os.getenv("CONSULTATION_MODEL", "qwen-plus")
max_conversation_rounds = int(os.getenv("MAX_CONVERSATION_ROUNDS", "10"))
benchmark_path = project_root / "Data" / "enquiries_benchmark.json"