from __future__ import annotations

import os
import re
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any, Iterable

import Config
from tqdm import tqdm


_CJK_CHAR_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_PAGE_MARKER_PATTERN = re.compile(r"^\s*(?:\d{1,4}\s+)?[\u4e00-\u9fffA-Za-z0-9·《》：:]+\s+\d{1,4}\s*$")
_PURE_PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])")


@dataclass
class CleanerChunk:
    index: int
    text: str


class DashScopeMarkdownFormatter:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen-turbo",
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or Config.dashscope_api_key
        self.model = model
        self.temperature = temperature

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and self._generation_class() is not None

    def format_chunk(self, chunk: CleanerChunk) -> str:
        if not self.enabled:
            raise RuntimeError("DashScope API Key 未配置，无法启用 LLM 清洗。")

        generation_class = self._generation_class()
        if generation_class is None:
            raise RuntimeError("当前 Python 环境未安装 dashscope，无法启用 LLM 清洗。")

        prompt = self._build_prompt(chunk)
        response = generation_class.call(
            model=self.model,
            api_key=self.api_key,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            temperature=self.temperature,
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(
                f"DashScope 调用失败: status={response.status_code}, code={response.code}, message={response.message}"
            )

        content = response.output.choices[0].message.content
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return str(content).strip()

    @staticmethod
    def _generation_class() -> Any | None:
        try:
            from dashscope import Generation
        except ImportError:
            return None
        return Generation

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是文档清洗助手。输入是一段 OCR/抽取后的中文书稿片段。"
            "你只能做版面清洗和轻微的显式 OCR 修复，不能创作、总结、补全事实，也不能改动原文顺序。"
            "输出必须是 Markdown。"
        )

    @staticmethod
    def _build_prompt(chunk: CleanerChunk) -> str:
        return (
            f"片段序号: {chunk.index}\n"
            "请按下面规则把输入片段整理成 Markdown：\n"
            "1. 删除页码、重复页眉/页脚、明显的扫描残留。\n"
            "2. 合并被错误换行打断的句子，按语义自然分段。\n"
            "3. 能明确判断是标题的小节，转成 Markdown 标题；标题层级只允许使用 # 和 ##。\n"
            "4. 不能补写原文没有的信息；不确定的字词保留原样；明显的错别字要修复。\n"
            "5. 只输出 Markdown 正文，不要解释。\n\n"
            "原始片段：\n"
            f"{chunk.text}"
        )


class MarkdownCleaner:
    def __init__(
        self,
        *,
        chunk_size: int = 2400,
        formatter: DashScopeMarkdownFormatter | None = None,
        use_llm: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.formatter = formatter or DashScopeMarkdownFormatter()
        self.use_llm = use_llm and self.formatter.enabled

    def clean_file(self, input_path: str | Path, output_path: str | Path | None = None) -> str:
        input_path = Path(input_path)
        markdown = self.clean_text(input_path.read_text(encoding="utf-8"), progress_desc=input_path.name)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
        return markdown

    def clean_text(self, raw_text: str, progress_desc: str | None = None) -> str:
        normalized_text = self._normalize_source_text(raw_text)
        chunks = self._build_chunks(normalized_text)
        cleaned_chunks = []
        chunk_iterable = tqdm(
            chunks,
            desc=f"清洗片段{f' - {progress_desc}' if progress_desc else ''}",
            unit="chunk",
            leave=False,
        )
        for chunk in chunk_iterable:
            cleaned_chunks.append(self._clean_chunk(chunk))
        return self._finalize_markdown(cleaned_chunks)

    def _clean_chunk(self, chunk: CleanerChunk) -> str:
        if self.use_llm:
            return self.formatter.format_chunk(chunk)
        return self._heuristic_markdown(chunk.text)

    def _normalize_source_text(self, raw_text: str) -> str:
        text = raw_text.replace("\ufeff", "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
        text = re.sub(r"(?<=[（《“‘])\s+", "", text)
        text = re.sub(r"\s+(?=[）》”’])", "", text)
        text = re.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
        text = text.replace("···", "……")

        cleaned_lines: list[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line:
                cleaned_lines.append("")
                continue
            if self._is_page_artifact(line):
                cleaned_lines.append("")
                continue
            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = self._inject_heading_breaks(text)
        return text.strip()

    def _build_chunks(self, text: str) -> list[CleanerChunk]:
        paragraphs = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
        if not paragraphs:
            return []

        chunks: list[CleanerChunk] = []
        current_parts: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current_parts, current_length
            if not current_parts:
                return
            chunks.append(CleanerChunk(index=len(chunks) + 1, text="\n\n".join(current_parts)))
            current_parts = []
            current_length = 0

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                flush()
                for piece in self._split_long_paragraph(paragraph):
                    chunks.append(CleanerChunk(index=len(chunks) + 1, text=piece))
                continue

            projected_length = current_length + len(paragraph) + (2 if current_parts else 0)
            if projected_length > self.chunk_size:
                flush()

            current_parts.append(paragraph)
            current_length += len(paragraph) + (2 if current_parts[:-1] else 0)

        flush()
        return chunks

    def _split_long_paragraph(self, paragraph: str) -> Iterable[str]:
        sentences = [segment.strip() for segment in _SENTENCE_SPLIT_PATTERN.split(paragraph) if segment.strip()]
        if len(sentences) <= 1:
            yield paragraph
            return

        current: list[str] = []
        current_length = 0
        for sentence in sentences:
            projected_length = current_length + len(sentence)
            if current and projected_length > self.chunk_size:
                yield "".join(current).strip()
                current = []
                current_length = 0
            current.append(sentence)
            current_length += len(sentence)
        if current:
            yield "".join(current).strip()

    def _heuristic_markdown(self, text: str) -> str:
        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        markdown_blocks: list[str] = []
        title_emitted = False
        for block in blocks:
            if self._looks_like_heading(block):
                prefix = "#" if not title_emitted else "##"
                markdown_blocks.append(f"{prefix} {block}")
                title_emitted = True
                continue
            markdown_blocks.append(self._wrap_as_paragraph(block))
        return "\n\n".join(markdown_blocks).strip()

    def _finalize_markdown(self, chunks: list[str]) -> str:
        markdown = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = re.sub(r"(?m)^\s+$", "", markdown)
        markdown = self._deduplicate_adjacent_headings(markdown)
        return markdown.strip() + "\n"

    def _inject_heading_breaks(self, text: str) -> str:
        text = re.sub(r"^(前言[^。！？\n]{0,24})", r"# \1\n\n", text, count=1)
        pattern = re.compile(
            r"([。！？!?])\s*"
            r"([\u4e00-\u9fffA-Za-z：:]{2,24})\s+"
            r"(?=[\u4e00-\u9fff“‘\"(（])"
        )
        return pattern.sub(self._replace_inline_heading, text)

    def _replace_inline_heading(self, match: re.Match[str]) -> str:
        sentence_end = match.group(1)
        heading = match.group(2).strip()
        if not self._looks_like_heading(heading):
            return match.group(0)
        return f"{sentence_end}\n\n## {heading}\n\n"

    def _deduplicate_adjacent_headings(self, markdown: str) -> str:
        lines = markdown.splitlines()
        result: list[str] = []
        previous_heading: str | None = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                normalized = re.sub(r"\s+", " ", stripped)
                if normalized == previous_heading:
                    continue
                previous_heading = normalized
            elif stripped:
                previous_heading = None
            result.append(line)
        return "\n".join(result)

    @staticmethod
    def _wrap_as_paragraph(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _is_page_artifact(line: str) -> bool:
        if _PURE_PAGE_NUMBER_PATTERN.match(line):
            return True
        if _PAGE_MARKER_PATTERN.match(line) and len(_CJK_CHAR_PATTERN.findall(line)) >= 3:
            return True
        return False

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        candidate = text.strip("# ").strip()
        if not candidate or len(candidate) > 30:
            return False
        if any(mark in candidate for mark in ("，", "。", "？", "！", "；", "“", "”")):
            return False
        if candidate.startswith(("我最好的朋友是", "我最喜欢做的是", "放学回家做的第一件事是", "我最希望实现的梦想是")):
            return False
        return len(_CJK_CHAR_PATTERN.findall(candidate)) >= max(2, len(candidate) // 2)