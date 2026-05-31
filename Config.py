
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
support_file_types = ["pdf", "txt", "epub"]

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

# Planner/State 循环模式的 state 和 history 输出路径（便于调试查看运行过程）
state_history_output_path = Path(
    os.getenv("STATE_HISTORY_OUTPUT_PATH", str(project_root / "data" / "session_state_history.jsonl"))
)

# --- Consultation System ---
consultation_model = os.getenv("CONSULTATION_MODEL", "qwen-plus")
max_conversation_rounds = int(os.getenv("MAX_CONVERSATION_ROUNDS", "10"))
benchmark_path = project_root / "data" / "enquiries_benchmark_v2.json"

# --- Checkpoint ---
checkpoint_enabled = os.getenv("CHECKPOINT_ENABLED", "true").lower() == "true"
checkpoint_dirname = os.getenv("CHECKPOINT_DIRNAME", "checkpoints")

# --- Model Routing ---
# 路由总开关：True 启用路由，False 全部使用 consultation_model（消融实验用）
routing_enabled = os.getenv("ROUTING_ENABLED", "true").lower() == "true"

# 小而快的模型：用于低成本分类任务（UserAgent 事实判断）
model_fast = os.getenv("MODEL_FAST", "qwen-turbo")
# 强一些的模型：用于问答对话轮次
model_strong = os.getenv("MODEL_STRONG", "qwen-plus")
# 深度思考模型：用于最终建议
model_think = os.getenv("MODEL_THINK", "qwen3-32b")
# 思考模型是否启用 enable_thinking（DashScope qwen3 系列专用参数）
model_think_enable_thinking = os.getenv("MODEL_THINK_ENABLE_THINKING", "true").lower() == "true"

# --- Consultation Phase Routing ---
# 这些配置控制问诊各阶段默认使用哪一档模型；仅需改这里，无需改命令行参数。
rewrite_phase_role = os.getenv("REWRITE_PHASE_ROLE", "fast").strip().lower()
init_hypothesis_phase_role = os.getenv("INIT_HYPOTHESIS_PHASE_ROLE", "strong").strip().lower()
tool_calling_phase_role = os.getenv("TOOL_CALLING_PHASE_ROLE", "strong").strip().lower()
state_update_phase_role = os.getenv("STATE_UPDATE_PHASE_ROLE", "fast").strip().lower()
consult_response_phase_role = os.getenv("CONSULT_RESPONSE_PHASE_ROLE", "strong").strip().lower()
final_think_phase_role = os.getenv("FINAL_THINK_PHASE_ROLE", "think").strip().lower()

_PHASE_ROLE_MAP = {
    "rewrite": rewrite_phase_role,
    "init_hypothesis": init_hypothesis_phase_role,
    "tool_calling": tool_calling_phase_role,
    "state_update": state_update_phase_role,
    "consult_response": consult_response_phase_role,
    "final_think": final_think_phase_role,
}

_ALLOWED_MODEL_ROLES = {"fast", "strong", "think"}


def resolve_model(role: str) -> str:
    """根据角色返回对应模型名。路由关闭时全部返回 consultation_model。

    role: 'fast' | 'strong' | 'think'
    """
    if not routing_enabled:
        return consultation_model
    return {"fast": model_fast, "strong": model_strong, "think": model_think}.get(
        role, consultation_model
    )


def resolve_phase_role(phase: str) -> str:
    """返回指定问诊阶段对应的模型档位。"""
    role = _PHASE_ROLE_MAP.get(phase, "strong")
    if role not in _ALLOWED_MODEL_ROLES:
        raise ValueError(f"未知的 phase role 配置: phase={phase} role={role}")
    return role


# --- Memory ---
# 记忆总开关：True 启用（将 memory_store 中的内容注入行为专家 system prompt）
memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

# --- LangSmith Tracing ---
langchain_tracing_v2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
langchain_api_key: str    = os.getenv("LANGCHAIN_API_KEY", "")
langchain_project: str    = os.getenv("LANGCHAIN_PROJECT", "CatBehaviorRag")


def setup_tracing() -> bool:
    """Propagate LangSmith config into os.environ before any LangChain object is built.

    Must be called before LLMClient.build_for_role(). Returns True if tracing activated.
    """
    import logging as _logging
    _log = _logging.getLogger(__name__)

    if langchain_tracing_v2.lower() != "true":
        return False
    if not langchain_api_key:
        _log.warning("LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set — tracing disabled.")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = langchain_api_key
    os.environ["LANGCHAIN_PROJECT"]     = langchain_project
    _log.info("LangSmith tracing enabled (project=%s)", langchain_project)
    return True