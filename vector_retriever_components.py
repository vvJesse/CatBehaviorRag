from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import logging

import Config

logger = logging.getLogger(__name__)

EmbeddingProvider = Literal["dashscope", "local"]


@dataclass(frozen=True)
class VectorStoreConfig:
    """聚合向量检索所需的配置，避免运行期分散读取全局 Config。"""

    full_data_dir: Path
    persist_dir: Path
    embedding_provider: EmbeddingProvider
    embedding_model_name: str
    collection_name: str
    dashscope_api_key: str
    local_embedding_device: str
    local_embedding_encode_kwargs: dict[str, Any]
    max_doc_length: int
    text_splitter_chunk_size: int
    text_splitter_chunk_overlap: int
    text_splitter_separators: list[str]

    @classmethod
    def from_sources(
        cls,
        *,
        full_data_dir: str | Path | None = None,
        persist_dir: str | Path | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_model_name: str | None = None,
        collection_name: str | None = None,
        dashscope_api_key: str | None = None,
        local_embedding_device: str | None = None,
    ) -> VectorStoreConfig:
        resolved_provider = (embedding_provider or Config.embedding_provider).lower().strip()
        default_embedding_model = (
            Config.local_embedding_model if resolved_provider == "local" else Config.dashscope_embedding_model
        )
        return cls(
            full_data_dir=Path(full_data_dir) if full_data_dir else Path(Config.full_text_path),
            persist_dir=Path(persist_dir) if persist_dir else Path(Config.vector_store_path),
            embedding_provider=resolved_provider,
            embedding_model_name=embedding_model_name or default_embedding_model,
            collection_name=collection_name or Config.vector_collection_name,
            dashscope_api_key=dashscope_api_key if dashscope_api_key is not None else Config.dashscope_api_key,
            local_embedding_device=local_embedding_device or Config.local_embedding_device,
            local_embedding_encode_kwargs=dict(Config.local_embedding_encode_kwargs),
            max_doc_length=Config.max_doc_length,
            text_splitter_chunk_size=Config.text_splitter_chunk_size,
            text_splitter_chunk_overlap=Config.text_splitter_chunk_overlap,
            text_splitter_separators=Config.text_splitter_separators,
        )

    def validate(self) -> None:
        if self.embedding_provider not in {"dashscope", "local"}:
            raise ValueError("embedding_provider 仅支持 'dashscope' 或 'local'")

        if self.embedding_provider == "dashscope" and not self.dashscope_api_key:
            raise ValueError("未配置 DashScope API Key，请在 Config.py 或环境变量 DASHSCOPE_API_KEY 中设置。")

        if self.embedding_provider == "local" and not self.embedding_model_name:
            raise ValueError("provider=local 时必须配置本地 embedding 模型名。")

        logger.debug(
            "配置校验通过 | provider=%s model=%s",
            self.embedding_provider,
            self.embedding_model_name,
        )


def create_embeddings(config: VectorStoreConfig) -> Embeddings:
    """按配置创建 embedding 实例，支持 DashScope 和本地 HuggingFace。"""
    if config.embedding_provider == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings

        logger.info("使用 DashScope embedding | model=%s", config.embedding_model_name)
        return DashScopeEmbeddings(
            model=config.embedding_model_name,
            dashscope_api_key=config.dashscope_api_key,
        )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ValueError(
            "provider=local 需要安装 langchain-huggingface 和 sentence-transformers。"
        ) from exc

    logger.info(
        "使用本地 HuggingFace embedding | model=%s device=%s",
        config.embedding_model_name,
        config.local_embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": config.local_embedding_device},
        encode_kwargs=config.local_embedding_encode_kwargs,
    )


class DocumentChunkingStrategy:
    """负责长文本切分策略与分块元数据补齐。"""

    def __init__(
        self,
        *,
        max_doc_length: int,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ):
        self.max_doc_length = max_doc_length
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def build_documents(self, text: str, base_metadata: dict[str, object]) -> list[Document]:
        """按文本长度决定是否切分，并返回统一构造好的 Document 列表。"""
        if len(text) <= self.max_doc_length:
            return [Document(page_content=text, metadata=base_metadata)]

        chunks = self._text_splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    **base_metadata,
                    "is_split": True,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "original_length": len(text),
                },
            )
            for chunk_index, chunk in enumerate(chunks, start=1)
        ]


class DocumentPreprocessor:
    """负责行文本清洗和 Document 构造。"""

    def __init__(self, chunking_strategy: DocumentChunkingStrategy):
        self.chunking_strategy = chunking_strategy

    @staticmethod
    def normalize_line(raw_line: str) -> str:
        return raw_line.replace("\ufeff", "").strip()

    def build_documents_from_line(self, line_text: str, file_path: Path, line_number: int) -> list[Document]:
        base_metadata = {
            "source": file_path.name,
            "path": str(file_path),
            "line_number": line_number,
        }
        return self.chunking_strategy.build_documents(line_text, base_metadata)


class TextDocumentLoader:
    """负责从全文目录读取文本文件并交给预处理器。"""

    def __init__(self, preprocessor: DocumentPreprocessor):
        self.preprocessor = preprocessor

    def load_from_directory(self, full_data_dir: Path) -> list[Document]:
        if not full_data_dir.exists():
            raise FileNotFoundError(f"全文目录不存在: {full_data_dir}")

        logger.info("开始加载文档 | 目录=%s", full_data_dir)
        documents: list[Document] = []
        for file_path in sorted(full_data_dir.glob("*.txt")):
            with file_path.open("r", encoding="utf-8") as file:
                total_lines = sum(1 for _ in file)
                file.seek(0)
                for line_number, raw_line in enumerate(
                    tqdm(file, total=total_lines, desc=file_path.name, unit="line"),
                    start=1,
                ):
                    line_text = self.preprocessor.normalize_line(raw_line)
                    if not line_text:
                        continue

                    documents.extend(self.preprocessor.build_documents_from_line(line_text, file_path, line_number))

        if not documents:
            raise ValueError(f"全文目录中没有可用文本行: {full_data_dir}")

        logger.info("文档加载完成 | 共 %d 个片段", len(documents))
        return documents


class VectorStoreManager:
    """负责 Chroma 向量库的构建、持久化和打开。"""

    def __init__(self, *, persist_dir: Path, collection_name: str, embeddings):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embeddings = embeddings

    def build_from_documents(self, documents: list[Document], force_rebuild: bool = True) -> Chroma:
        if force_rebuild and self.persist_dir.exists():
            logger.info("删除旧向量库 | 路径=%s", self.persist_dir)
            shutil.rmtree(self.persist_dir)

        logger.info(
            "开始构建向量库 | 片段数=%d collection=%s",
            len(documents),
            self.collection_name,
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_name=self.collection_name,
        )
        logger.info("向量库构建完成 | 路径=%s", self.persist_dir)
        return vector_store

    def get_or_create(self, document_supplier: Callable[[], list[Document]]) -> Chroma:
        if not self.persist_dir.exists():
            logger.info("向量库不存在，自动构建 | 路径=%s", self.persist_dir)
            return self.build_from_documents(document_supplier(), force_rebuild=True)

        logger.info("加载已有向量库 | 路径=%s collection=%s", self.persist_dir, self.collection_name)
        return Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )