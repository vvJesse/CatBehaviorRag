from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

from tqdm import tqdm

import Config
from vector_retriever_components import DocumentChunkingStrategy, DocumentPreprocessor


@dataclass(frozen=True)
class DocStatistics:
	"""单个 doc 的关键信息；这里的 doc 指 full_data 中的一条非空行。"""

	source: str
	line_number: int
	content_char_count: int
	retrieval_document_count: int
	extra_chunk_count: int


@dataclass(frozen=True)
class CorpusStatistics:
	"""按“每个非空行就是一个 doc”的口径统计全文语料。"""

	doc_count: int
	total_content_char_count: int
	average_doc_char_count: float
	median_doc_char_count: float
	p95_doc_char_count: int
	max_doc_char_count: int
	total_retrieval_document_count: int
	total_extra_chunk_count: int
	split_doc_count: int
	longest_doc_source: str
	longest_doc_line_number: int
	longest_doc_char_count: int


@dataclass(frozen=True)
class FileStatistics:
	"""单个文件下 doc 分布的摘要，用于观察不同文件的内容密度。"""

	source: str
	doc_count: int
	total_content_char_count: int
	average_doc_char_count: float
	max_doc_char_count: int
	split_doc_count: int
	total_extra_chunk_count: int


class FullDataStatistician:
	"""读取 full_data 中的所有 doc，并按当前 RAG 配置计算统计信息。"""

	def __init__(self, full_data_dir: str | Path | None = None):
		self.full_data_dir = Path(full_data_dir) if full_data_dir else Path(Config.full_text_path)
		chunking_strategy = DocumentChunkingStrategy(
			max_doc_length=Config.max_doc_length,
			chunk_size=Config.text_splitter_chunk_size,
			chunk_overlap=Config.text_splitter_chunk_overlap,
			separators=Config.text_splitter_separators,
		)
		self._preprocessor = DocumentPreprocessor(chunking_strategy)

	@staticmethod
	def _count_content_chars(text: str) -> int:
		"""把空白排除后再计数，更接近日常理解中的“字数”。"""
		return sum(1 for char in text if not char.isspace())

	def _collect_file_docs(self, file_path: Path) -> list[DocStatistics]:
		doc_statistics: list[DocStatistics] = []
		with file_path.open("r", encoding="utf-8") as file:
			for line_number, raw_line in enumerate(file, start=1):
				line_text = self._preprocessor.normalize_line(raw_line)
				if not line_text:
					continue

				documents = self._preprocessor.build_documents_from_line(line_text, file_path, line_number)
				doc_statistics.append(
					DocStatistics(
						source=file_path.name,
						line_number=line_number,
						content_char_count=self._count_content_chars(line_text),
						retrieval_document_count=len(documents),
						extra_chunk_count=len(documents) - 1,
					)
				)

		return doc_statistics

	def collect_doc_statistics(self) -> list[DocStatistics]:
		if not self.full_data_dir.exists():
			raise FileNotFoundError(f"全文目录不存在: {self.full_data_dir}")

		file_paths = sorted(self.full_data_dir.glob("*.txt"))
		if not file_paths:
			raise ValueError(f"全文目录中没有可统计的 txt 文件: {self.full_data_dir}")

		doc_statistics: list[DocStatistics] = []
		for file_path in tqdm(file_paths, desc="full_data_files", unit="file"):
			doc_statistics.extend(self._collect_file_docs(file_path))

		if not doc_statistics:
			raise ValueError(f"全文目录中没有可统计的非空行: {self.full_data_dir}")

		return doc_statistics

	@staticmethod
	def build_corpus_statistics(doc_statistics: list[DocStatistics]) -> CorpusStatistics:
		if not doc_statistics:
			raise ValueError("doc_statistics 不能为空")

		longest_doc = max(doc_statistics, key=lambda item: item.content_char_count)
		doc_char_counts = [item.content_char_count for item in doc_statistics]
		total_content_char_count = sum(doc_char_counts)
		sorted_doc_char_counts = sorted(doc_char_counts)
		p95_index = min(len(sorted_doc_char_counts) - 1, max(0, int(len(sorted_doc_char_counts) * 0.95) - 1))
		return CorpusStatistics(
			doc_count=len(doc_statistics),
			total_content_char_count=total_content_char_count,
			average_doc_char_count=total_content_char_count / len(doc_statistics),
			median_doc_char_count=median(doc_char_counts),
			p95_doc_char_count=sorted_doc_char_counts[p95_index],
			max_doc_char_count=longest_doc.content_char_count,
			total_retrieval_document_count=sum(item.retrieval_document_count for item in doc_statistics),
			total_extra_chunk_count=sum(item.extra_chunk_count for item in doc_statistics),
			split_doc_count=sum(1 for item in doc_statistics if item.extra_chunk_count > 0),
			longest_doc_source=longest_doc.source,
			longest_doc_line_number=longest_doc.line_number,
			longest_doc_char_count=longest_doc.content_char_count,
		)

	@staticmethod
	def build_file_statistics(doc_statistics: list[DocStatistics]) -> list[FileStatistics]:
		file_groups: dict[str, list[DocStatistics]] = {}
		for item in doc_statistics:
			file_groups.setdefault(item.source, []).append(item)

		file_statistics: list[FileStatistics] = []
		for source, items in file_groups.items():
			total_content_char_count = sum(item.content_char_count for item in items)
			file_statistics.append(
				FileStatistics(
					source=source,
					doc_count=len(items),
					total_content_char_count=total_content_char_count,
					average_doc_char_count=total_content_char_count / len(items),
					max_doc_char_count=max(item.content_char_count for item in items),
					split_doc_count=sum(1 for item in items if item.extra_chunk_count > 0),
					total_extra_chunk_count=sum(item.extra_chunk_count for item in items),
				)
			)

		return sorted(file_statistics, key=lambda item: item.total_content_char_count, reverse=True)

	def generate_report(self) -> str:
		doc_statistics = sorted(
			self.collect_doc_statistics(),
			key=lambda item: item.content_char_count,
			reverse=True,
		)
		corpus_statistics = self.build_corpus_statistics(doc_statistics)
		file_statistics = self.build_file_statistics(doc_statistics)
		top_long_docs = doc_statistics[:10]

		lines = [
			f"全文目录: {self.full_data_dir}",
			f"文档数: {corpus_statistics.doc_count}",
			f"有效总字数（不含空白）: {corpus_statistics.total_content_char_count}",
			f"平均每个 doc 的有效字数: {corpus_statistics.average_doc_char_count:.1f}",
			f"doc 字数中位数: {corpus_statistics.median_doc_char_count:.1f}",
			f"doc 字数 P95: {corpus_statistics.p95_doc_char_count}",
			f"最长 doc 字数: {corpus_statistics.max_doc_char_count}",
			f"按当前切分配置预计入库片段数: {corpus_statistics.total_retrieval_document_count}",
			f"因切分新增的片段数: {corpus_statistics.total_extra_chunk_count}",
			f"触发切分的 doc 数: {corpus_statistics.split_doc_count}",
			f"最长 doc: {corpus_statistics.longest_doc_source}#L{corpus_statistics.longest_doc_line_number} ({corpus_statistics.longest_doc_char_count})",
			"",
			"每个文件的摘要:",
		]

		for item in file_statistics:
			lines.append(
				" | ".join(
					[
						item.source,
						f"doc 数 {item.doc_count}",
						f"有效字数 {item.total_content_char_count}",
						f"平均 doc 字数 {item.average_doc_char_count:.1f}",
						f"最长 doc {item.max_doc_char_count}",
						f"触发切分 doc {item.split_doc_count}",
						f"新增片段 {item.total_extra_chunk_count}",
					]
				)
			)

		lines.append("")
		lines.append("最长的 10 个 doc:")
		for item in top_long_docs:
			lines.append(
				" | ".join(
					[
						f"{item.source}#L{item.line_number}",
						f"字数 {item.content_char_count}",
						f"预计入库片段 {item.retrieval_document_count}",
						f"新增片段 {item.extra_chunk_count}",
					]
				)
			)

		return "\n".join(lines)


def main() -> None:
	statistician = FullDataStatistician()
	print(statistician.generate_report())


if __name__ == "__main__":
	main()
