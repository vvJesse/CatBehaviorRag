from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from vector_retriever_components import (
	DocumentChunkingStrategy,
	DocumentPreprocessor,
	TextDocumentLoader,
	VectorStoreConfig,
	VectorStoreManager,
)


class VectorRetriever:
	"""负责把全文目录中的文本构建为向量库，并提供检索能力。"""

	def __init__(
		self,
		full_data_dir: str | Path | None = None,
		persist_dir: str | Path | None = None,
		embedding_model_name: str | None = None,
		collection_name: str | None = None,
		dashscope_api_key: str | None = None,
	):
		"""初始化向量检索器及其依赖配置。"""
		self.config = VectorStoreConfig.from_sources(
			full_data_dir=full_data_dir,
			persist_dir=persist_dir,
			embedding_model_name=embedding_model_name,
			collection_name=collection_name,
			dashscope_api_key=dashscope_api_key,
		)
		self.config.validate()

		# 为兼容现有外部代码，保留常用属性访问。
		self.full_data_dir = self.config.full_data_dir
		self.persist_dir = self.config.persist_dir
		self.collection_name = self.config.collection_name
		self.embedding_model_name = self.config.embedding_model_name
		self.dashscope_api_key = self.config.dashscope_api_key

		# 向量模型统一切换为 DashScopeEmbeddings，避免本地模型依赖。
		self._embeddings = DashScopeEmbeddings(
			model=self.embedding_model_name,
			dashscope_api_key=self.dashscope_api_key,
		)

		chunking_strategy = DocumentChunkingStrategy(
			max_doc_length=self.config.max_doc_length,
			chunk_size=self.config.text_splitter_chunk_size,
			chunk_overlap=self.config.text_splitter_chunk_overlap,
			separators=self.config.text_splitter_separators,
		)
		self._preprocessor = DocumentPreprocessor(chunking_strategy)
		self._document_loader = TextDocumentLoader(self._preprocessor)
		self._vector_store_manager = VectorStoreManager(
			persist_dir=self.persist_dir,
			collection_name=self.collection_name,
			embeddings=self._embeddings,
		)

	@staticmethod
	def _normalize_line(raw_line: str) -> str:
		"""清理每一行文本，显式过滤空白行和 BOM 等无效字符。"""
		return DocumentPreprocessor.normalize_line(raw_line)

	def _build_documents_from_line(self, line_text: str, file_path: Path, line_number: int) -> list[Document]:
		"""按长度决定是否切分当前文本行，并为每个片段补齐元数据。"""
		return self._preprocessor.build_documents_from_line(line_text, file_path, line_number)

	def load_documents(self) -> list[Document]:
		"""把全文目录中的每一行文本转成可检索的 Document。"""
		return self._document_loader.load_from_directory(self.full_data_dir)

	def build_vector_store(self, force_rebuild: bool = True) -> Chroma:
		"""根据全文目录重建本地 Chroma 向量库。"""
		documents = self.load_documents()
		return self._vector_store_manager.build_from_documents(documents, force_rebuild=force_rebuild)

	def get_vector_store(self) -> Chroma:
		"""返回可用向量库；不存在时自动构建。"""
		return self._vector_store_manager.get_or_create(self.load_documents)

	def search(self, query: str, k: int = 5) -> list[Document]:
		"""返回与查询最相近的文档片段。"""
		if not query or not query.strip():
			raise ValueError("query 不能为空")

		vector_store = self.get_vector_store()
		return vector_store.similarity_search(query=query.strip(), k=k)

	def search_with_scores(self, query: str, k: int = 5) -> list[tuple[Document, float]]:
		"""返回相似文档及其分数，便于调试和阈值分析。"""
		if not query or not query.strip():
			raise ValueError("query 不能为空")

		vector_store = self.get_vector_store()
		return vector_store.similarity_search_with_score(query=query.strip(), k=k)


if __name__ == "__main__":
	# 本地调试入口：重建向量库并执行一次简单检索。
	retriever = VectorRetriever()
	retriever.build_vector_store(force_rebuild=True)

	sample_query = "猫咪在家里乱尿怎么办"
	results = retriever.search_with_scores(sample_query, k=5)
	for index, (document, score) in enumerate(results, start=1):
		print(f"[{index}] score={score:.4f}")
		print(document.page_content)
		print(document.metadata)
		print()
