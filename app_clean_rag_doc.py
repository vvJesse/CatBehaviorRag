from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_document_uploader.markdown_cleaner import DashScopeMarkdownFormatter, MarkdownCleaner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "rag_document_uploader" / "data-cleaning-example" / "家有恶猫-片段.txt"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "rag_document_uploader" / "cleaned_data" / "家有恶猫-片段.cleaned.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 OCR/抽取后的猫行为文档清洗为 Markdown。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="输入文本路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 Markdown 路径")
    parser.add_argument("--chunk-size", type=int, default=2400, help="单个清洗片段的最大字符数")
    parser.add_argument("--model", default="qwen-turbo", help="DashScope 文本模型")
    parser.add_argument("--api-key", default=None, help="可选，显式指定 DashScope API Key")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用 DashScope，仅使用规则清洗和 Markdown 粗分段",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    formatter = DashScopeMarkdownFormatter(api_key=args.api_key, model=args.model)
    requested_llm = not args.no_llm
    if requested_llm and not formatter.enabled:
        print(
            "DashScope 不可用，已自动回退到规则清洗。"
            "如需启用 LLM，请确认当前环境已安装 dashscope 且已设置 DASHSCOPE_API_KEY。",
            file=sys.stderr,
        )

    cleaner = MarkdownCleaner(
        chunk_size=args.chunk_size,
        formatter=formatter,
        use_llm=requested_llm,
    )
    markdown = cleaner.clean_file(args.input, args.output)

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"llm_enabled: {cleaner.use_llm}")
    print(f"markdown_chars: {len(markdown)}")


if __name__ == "__main__":
    main()