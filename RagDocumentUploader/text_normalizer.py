from __future__ import annotations

import re


class OCRTextNormalizer:
    """统一处理 OCR 文本清洗规则。"""

    def normalize(self, text: str) -> str:
        """修复 OCR 断开的文字：合并断行、保留段落、清理空格。"""
        normalized_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        normalized_text = re.sub(r'\n+', '\n\n', normalized_text)
        normalized_text = re.sub(r' +', ' ', normalized_text)
        return normalized_text.strip()