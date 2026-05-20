from __future__ import annotations

from pathlib import Path


class FullTextStorage:
    """负责把解析后的全文写入磁盘。"""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    def save(self, source_name: str, full_text: str) -> Path:
        """将全文写入目标目录，并返回落盘路径。"""
        output_path = self.build_path(source_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_text, encoding="utf-8")
        return output_path

    def build_path(self, source_name: str) -> Path:
        """根据源文件名生成对应的文本保存路径。"""
        file_name = Path(source_name)
        safe_name = file_name.stem or "uploaded_file"
        return self.base_path / f"{safe_name}.txt"