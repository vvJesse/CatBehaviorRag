
from __future__ import annotations

import os
from pathlib import Path


# 项目根目录，用于拼接其余相对路径配置。
project_root = Path(__file__).resolve().parent

# 文件上传模块支持的文件类型。
support_file_types = ["pdf", "txt"]

# 上传文档转全文后的落盘目录。
full_text_path = project_root / "RagDocumentUploader" / "full_data"

# 向量库持久化目录。
vector_store_path = project_root / ".chroma" / "line_documents"

# Chroma 集合名称。
vector_collection_name = "cat_behavior_line_documents"

# DashScope 向量模型配置。
# 如果不想把密钥写死，也可以在环境变量中设置 DASHSCOPE_API_KEY。
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
dashscope_embedding_model = "text-embedding-v1"

# 单条文档允许直接入库的最大字符数；超过后将触发文本切分。
max_doc_length = 300

# 文本切分器配置。
text_splitter_chunk_size = 200
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